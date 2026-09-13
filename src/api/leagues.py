"""Phase 5.2: where each league's ingested data lives, and making sure
it exists before the API answers for that league.

Phase 5.1 deliberately left the on-disk convention alone: one league in
the flat data/raw/sleeper/ + data/chroma/ that `python -m
src.ingest.sleeper` / `python -m src.rag.embed` write, verified by
league_id at query time. A multi-league server needs one directory per
league. This module is that mapping -- and nothing else in the project
has to know about it, because recommend()/generate_report() already
take raw_dir/persist_dir explicitly.

Layout:
  - The league named by SLEEPER_LEAGUE_ID (the developer's own, already
    ingested by the CLI convention) keeps using the flat dirs, so the
    CLI and the API server see the same data for it.
  - Every other league lives under data/raw/leagues/<league_id>/
    {sleeper,chroma}/ -- under data/raw/ so the existing .gitignore rule
    (raw pulls are reproducible, never committed) already covers it.

The signals table (data/processed/signals/) is shared by every league:
it is league-agnostic by construction (CLAUDE.md's key principle), so
it is computed once and embedded into each league's Chroma collection
alongside that league's own roster/matchup chunks -- the same thing
src/rag/embed.py's main() already does for the single flat league.
(Re-embedding the shared signal chunks per league is redundant work at
friend-group scale; splitting the collection is a src/rag/ change and
was flagged, not done, in Phase 5.1.)

ensure_league_data() runs the existing ingest + embed for a league the
first time it's needed (network: Sleeper + nflverse's player list),
then just verifies it via Phase 5.1's load_league(). Sleeper is blocked
in the sandbox this was built in, so the ingest path here is exercised
in tests only with the ingest/embed functions mocked -- see TODO.md's
Phase 5.2 entry for the live re-run.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import chromadb
from chromadb.api.shared_system_client import SharedSystemClient

from src.ingest import sleeper
from src.rag import embed
from src.rag.embed import CHROMA_DIR, RAW_DIR, SIGNALS_DIR
from src.reasoning.league import LeagueConfig, load_league

LEAGUES_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "leagues"

# ---- Chroma client warm-up (Phase 5.3 bugfix) ----
# src/rag/retrieve.py opens `chromadb.PersistentClient(path)` on every
# query. chromadb caches one shared System per path, but it registers the
# new System in its cache BEFORE `start()` finishes and takes no lock
# (SharedSystemClient._create_system_if_not_exists). So the first two
# requests to touch a league's Chroma index in a fresh server process can
# race: thread A is still starting the Rust bindings while thread B finds
# the cached-but-unstarted System and dies with
# "AttributeError: 'RustBindingsAPI' object has no attribute 'bindings'"
# -> "Could not connect to tenant default_tenant" -> HTTP 500. Observed
# for real: the Feed fires start_sit + drop + waiver_pickups together;
# start_sit and drop both read Chroma (per-player signal lookups) and
# 500'd, waiver_pickups reads only parquet and succeeded. A single curl
# never races. Once a path's System has started, later concurrent
# PersistentClient calls reuse it safely -- so the fix is to create the
# first client for each league path exactly once, under a lock, before
# any request-thread reaches retrieve.py. (The deeper fix, one shared
# client per path inside src/rag/retrieve.py, is a src/rag/ change and
# was out of scope for the bugfix pass that added this.)
#
# ---- ...and re-warming it when a refresh rewrites the index (Phase 5.7) ----
# That same process-wide System cache is also a staleness trap, which
# Phase 5.7's investigation reproduced through the real production
# functions (src/api/leagues.warm_chroma + src/rag/retrieve.query):
# once a process has warmed a path, chromadb keeps that path's in-memory
# vector index, and `collection.query()` is answered from it. So after
# `src/scheduler/refresh.py` re-embeds a league, a long-running server:
#   - never surfaces a chunk ADDED by the refresh (the new week's signal
#     chunks are invisible to semantic search), and
#   - returns phantom hits for chunks the refresh REMOVED -- same id,
#     document None, which is what retrieve.query() then hands the model.
# `collection.get()` (the metadata-filtered path: query_player_signal,
# so get_player_signals) reads SQLite directly and stays fresh, as do
# the parquet ranking tables and the raw Sleeper JSON. Only the semantic
# path goes stale -- measured, not assumed; see TODO.md's Phase 5.7 entry.
#
# The fix: remember the on-disk index's stamp when we warm a path and
# re-check it on every request (warm_chroma is already called per request
# by ensure_league_data). A changed stamp means someone re-embedded, so we
# forget chromadb's cached System for THAT path and build a new one, which
# reads the index off disk. Deliberately per-path rather than chromadb's
# public clear_system_cache(): that drops every path's System, and another
# league's request thread could then re-create its System outside this
# lock -- exactly the unsynchronized-start race the note above fixes.
# Nothing is stopped, only forgotten, so a query already in flight on the
# old System keeps working (verified under four concurrent query threads).
# This works whether the refresh runs inside this process (dev_server's
# background thread) or in a separate one (Phase 5.6's cron/worker).
_chroma_warm_lock = threading.Lock()
_chroma_warmed: dict[str, tuple[int, int] | None] = {}
_NEVER_WARMED = object()


def _index_stamp(persist_dir: Path) -> tuple[int, int] | None:
    """A cheap fingerprint of the on-disk Chroma index: the (mtime, size)
    of its SQLite file, which every add/delete touches. None when the
    index doesn't exist yet (an un-ingested league).

    Measured, because it decides where warm_chroma reads this: merely
    OPENING a PersistentClient bumps the SQLite mtime (creating the file
    on a fresh directory, touching it on an existing one), while reads
    afterwards do not. So the stamp to remember is the one taken AFTER
    the open -- a stamp taken before it is stale the moment we warm, and
    would make every single request rebuild the client.

    Known, accepted window: if a refresh's final commit lands inside our
    client construction, the stamp we record already covers a write our
    new System may have opened just before, and nothing would re-warm
    until the next refresh. Milliseconds wide, and invalidate_chroma()
    closes it outright for a refresh running in this process. Closing it
    for an out-of-process refresh too would mean a generation marker
    written by src/rag/embed.py itself -- a src/rag/ change, flagged in
    TODO.md rather than done here.
    """
    try:
        stat = (persist_dir / "chroma.sqlite3").stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _forget_cached_system(key: str) -> None:
    """Drop chromadb's cached System for this one path so the next
    PersistentClient(path=key) builds a fresh one. Uses chromadb
    internals on purpose (see the note above for why the public
    clear_system_cache() is the wrong tool here); falls back to it if a
    future chromadb drops these attributes, trading the race back for
    correctness rather than silently serving stale results."""
    try:
        SharedSystemClient._identifier_to_system.pop(key, None)
        SharedSystemClient._identifier_to_refcount.pop(key, None)
    except AttributeError:  # pragma: no cover - depends on the installed chromadb
        SharedSystemClient.clear_system_cache()


def warm_chroma(persist_dir: Path) -> None:
    """Open (and thereby fully start) the shared Chroma client for
    persist_dir, serialized across threads -- once per process, and again
    whenever the index on disk has been rewritten since. Idempotent and
    cheap after the first call (one stat() per call)."""
    key = str(persist_dir)
    if _chroma_warmed.get(key, _NEVER_WARMED) == _index_stamp(persist_dir):
        return
    with _chroma_warm_lock:
        if _chroma_warmed.get(key, _NEVER_WARMED) == _index_stamp(persist_dir):
            return
        if key in _chroma_warmed:
            _forget_cached_system(key)
        chromadb.PersistentClient(path=key)
        _chroma_warmed[key] = _index_stamp(persist_dir)  # after the open -- see _index_stamp


def invalidate_chroma(persist_dir: Path) -> None:
    """Force the next query against persist_dir to read the index from
    disk. warm_chroma()'s stamp check already catches an out-of-process
    refresh on the next request; src/scheduler/refresh.py calls this
    directly after re-embedding in-process so the very next query is
    fresh without depending on filesystem timestamp granularity."""
    key = str(persist_dir)
    with _chroma_warm_lock:
        if key in _chroma_warmed:
            _forget_cached_system(key)
            del _chroma_warmed[key]


def league_dirs(league_id: str) -> tuple[Path, Path]:
    """(raw_dir, persist_dir) for league_id -- the flat CLI dirs for the
    developer's own SLEEPER_LEAGUE_ID league, a per-league directory for
    everyone else's."""
    league_id = str(league_id)
    if league_id == (os.environ.get("SLEEPER_LEAGUE_ID") or "").strip():
        return RAW_DIR, CHROMA_DIR
    return LEAGUES_DIR / league_id / "sleeper", LEAGUES_DIR / league_id / "chroma"


def is_ingested(league_id: str) -> bool:
    raw_dir, persist_dir = league_dirs(league_id)
    return (raw_dir / "league.json").exists() and persist_dir.exists()


def ingested_league_ids() -> list[str]:
    """Every league whose data is actually on disk, newest-registered
    last: the per-league directories plus (when it's ingested) the flat
    SLEEPER_LEAGUE_ID league. This is what src/scheduler/refresh.py
    iterates over.

    "On disk" rather than "in the SQLite leagues table" on purpose. That
    table holds every league Sleeper listed for everyone who has logged
    in, including ones nobody ever opened; refreshing those would hit
    Sleeper for leagues with no local data to keep current. A league is
    ingested the first time a session selects it (ensure_league_data),
    and from then on it's here. It also means the scheduler needs no
    database access at all, which matters for Phase 5.6's cron/worker
    shape.
    """
    found: list[str] = []
    own = (os.environ.get("SLEEPER_LEAGUE_ID") or "").strip()
    if own and is_ingested(own):
        found.append(own)
    if LEAGUES_DIR.exists():
        for child in sorted(LEAGUES_DIR.iterdir()):
            if child.is_dir() and child.name not in found and is_ingested(child.name):
                found.append(child.name)
    return found


def ingest_league(league_id: str, week: int | None = None) -> tuple[Path, Path]:
    """Pull league_id from Sleeper and (re)build its Chroma collection,
    using the exact same functions the CLI convention uses
    (src/ingest/sleeper.py's run(), src/rag/embed.py's embed()) pointed
    at this league's directories. Network."""
    raw_dir, persist_dir = league_dirs(league_id)
    sleeper.run(league_id=str(league_id), week=week, out_dir=raw_dir)
    embed.embed(persist_dir=persist_dir, raw_dir=raw_dir, signals_dir=SIGNALS_DIR if SIGNALS_DIR.exists() else None)
    return raw_dir, persist_dir


def ensure_league_data(league_id: str, refresh: bool = False) -> LeagueConfig:
    """The verified LeagueConfig for league_id, ingesting it first if it
    isn't on disk yet (or if refresh=True). Raises LeagueMismatchError
    via load_league() if the directory somehow holds a different league."""
    if refresh or not is_ingested(league_id):
        ingest_league(league_id)
    raw_dir, persist_dir = league_dirs(league_id)
    config = load_league(str(league_id), raw_dir=raw_dir, persist_dir=persist_dir)
    warm_chroma(config.persist_dir)  # see the note above -- must happen before any request thread queries Chroma
    return config
