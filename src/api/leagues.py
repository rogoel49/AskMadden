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
then just verifies it via Phase 5.1's load_league(). It also re-embeds
that league when the shared signals table has changed on disk since the
league's Chroma collection was last built -- see the signals-resync note
below for why that is needed and what it does NOT fix. Sleeper is blocked
in the sandbox this was built in, so the ingest path here is exercised
in tests only with the ingest/embed functions mocked -- see TODO.md's
Phase 5.2 entry for the live re-run.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import chromadb

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
_chroma_warm_lock = threading.Lock()
_chroma_warmed: set[str] = set()


def warm_chroma(persist_dir: Path) -> None:
    """Open (and thereby fully start) the shared Chroma client for
    persist_dir once per process, serialized across threads. Idempotent
    and cheap after the first call."""
    key = str(persist_dir)
    if key in _chroma_warmed:
        return
    with _chroma_warm_lock:
        if key in _chroma_warmed:
            return
        chromadb.PersistentClient(path=key)
        _chroma_warmed.add(key)


# ---- Signals resync (investigated and added after a manual-refresh report) ----
# The signals table (data/processed/signals/*.parquet) is shared by every
# league and is refreshed OUT OF BAND, by re-running `python -m
# src.ingest.nflverse` + `python -m src.signals.matchup_signals` while the
# API server keeps running. Two different layers read it, and only one of
# them noticed the refresh:
#
#   - Reports read the parquet directly (ranking.load_signals_table
#     re-globs signals_dir on EVERY call), so a new week's file is picked
#     up with no restart and no cache to invalidate. What can still hide it
#     is the as-of-week bound: generate_report() infers as_of_week from
#     Sleeper's own nfl_state.json (recommend._infer_season_and_week), and
#     a signals-only refresh doesn't advance that file -- so the newer
#     week's rows are correctly filtered out as "future" until the Sleeper
#     ingest is re-run too. That is the as-of-date rule working, not a bug;
#     it is documented in README/TODO rather than "fixed" here, because
#     inferring the current week from the signals directory instead of the
#     league's own state would be a different (and weaker) definition.
#
#   - Chat reads the SAME numbers out of Chroma (get_player_signals ->
#     retrieve.query_player_signal), and Chroma is only ever written by
#     embed(). ensure_league_data() used to skip ingest entirely whenever
#     is_ingested() was true, so for an already-ingested league embed()
#     never ran again and the refreshed weeks were never embedded at all.
#     Restarting the server did NOT fix that -- is_ingested() is still
#     true on the next boot -- so the stale chunks were permanent until
#     someone re-ran the embed by hand or passed refresh=True.
#
# This gap is NOT warm_chroma()'s cached client going stale, though the two
# are easy to confuse and chromadb's two read paths behave differently
# (both verified in tests/test_api_signals_refresh.py):
#   - collection.get() -- metadata lookups, which is what
#     query_player_signal()/get_player_signals use -- reads SQLite directly
#     and DOES see another process's rewrite on the next call.
#   - collection.query() -- semantic search, which is what retrieve.query()/
#     search_league_info use -- answers from a per-process in-memory vector
#     index that a warmed process does NOT refresh after an out-of-process
#     re-embed. It returns removed ids with documents/metadatas of None,
#     which callers doing r["metadata"].get(...) turn into an AttributeError.
#     That is a real, separate defect; it is NOT introduced or fixed here
#     (the resync below re-embeds IN-process, which leaves that process's
#     own index correct -- verified), and PR #28's mtime/size stamp on
#     warm_chroma() is the fix for the out-of-process case.
# The gap this closes is purely "nothing re-ran embed", which
# _resync_signals_if_changed handles by fingerprinting the signals files the
# collection was built from. (Leagues ingested before that fingerprint
# existed adopt it on first sight rather than rebuilding -- see the
# function's docstring for why, and for the one-time migration that leaves
# behind.)
_SIGNALS_STAMP = ".askmadden_signals_fingerprint"
_signals_resync_lock = threading.Lock()


def signals_fingerprint(signals_dir: Path) -> str:
    """Identity of the signals table as it currently sits on disk: every
    parquet's name, size and mtime. Changes whenever matchup_signals.py
    adds a week or rewrites one; empty string when there are no signals
    at all (which is a valid state -- see ingest_league)."""
    if not signals_dir.exists():
        return ""
    parts = []
    for path in sorted(signals_dir.glob("signals_*.parquet")):
        try:
            stat = path.stat()
        except OSError:  # file vanished mid-scan (a refresh writing right now)
            continue
        parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(parts)


def _read_stamp(persist_dir: Path) -> str | None:
    try:
        return (persist_dir / _SIGNALS_STAMP).read_text()
    except OSError:
        return None


def _write_stamp(persist_dir: Path, fingerprint: str) -> None:
    try:
        persist_dir.mkdir(parents=True, exist_ok=True)
        (persist_dir / _SIGNALS_STAMP).write_text(fingerprint)
    except OSError:
        pass  # a stamp we can't persist just means we re-check next time


def _resync_signals_if_changed(league_id: str, raw_dir: Path, persist_dir: Path, signals_dir: Path) -> bool:
    """Re-embed league_id when signals_dir has changed since its Chroma
    collection was built. Returns True if a re-embed actually happened.

    Local only -- this re-runs embed() against the ALREADY-ingested
    raw_dir, so it never re-hits Sleeper; it is the embed half of
    ingest_league(), not the network half.

    Serialized on _signals_resync_lock, and called from
    ensure_league_data() -- which every league-scoped request goes through
    before it touches Chroma -- so a request arriving mid-resync waits for
    it rather than reading the half-rebuilt collection. (embed() clears the
    collection before re-adding, so a reader that got past
    ensure_league_data() before the resync started could still see a
    partial collection for that moment; at friend-group scale, once per
    refresh, that is a far smaller problem than serving permanently stale
    signals, but it is the reason this is a resync-on-demand and not a
    background job.)

    A league with NO stamp at all -- one ingested before this check
    existed -- adopts the current fingerprint WITHOUT re-embedding. The
    alternative (treat "unknown" as "stale" and rebuild) would make the
    first request after upgrading silently re-embed every existing
    league, and would turn a partially-written raw_dir -- which
    is_ingested() happily accepts, it only looks for league.json -- into
    a failed request where today it is merely incomplete. The cost is
    that a league whose signals were refreshed BEFORE this code landed
    stays stale until the next refresh; that one-time migration is
    documented in README/TODO as a single `refresh: true` session, not
    left for someone to discover."""
    fingerprint = signals_fingerprint(signals_dir)
    stamp = _read_stamp(persist_dir)
    if stamp == fingerprint:
        return False
    if stamp is None:
        _write_stamp(persist_dir, fingerprint)
        return False
    with _signals_resync_lock:
        if _read_stamp(persist_dir) == fingerprint:  # another thread just did it
            return False
        embed.embed(
            persist_dir=persist_dir,
            raw_dir=raw_dir,
            signals_dir=signals_dir if signals_dir.exists() else None,
        )
        _write_stamp(persist_dir, fingerprint)
        return True


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
    via load_league() if the directory somehow holds a different league.

    For a league that IS already ingested, re-embeds it when the shared
    signals table has changed on disk since its collection was built, so
    an out-of-band `matchup_signals` refresh reaches the chat path without
    a restart (which would not have helped anyway) -- see the
    signals-resync note above. SIGNALS_DIR is read through the module
    global at call time, not bound as a default, so tests and any future
    per-deployment override can point it elsewhere."""
    raw_dir, persist_dir = league_dirs(league_id)
    signals_dir = SIGNALS_DIR
    if refresh or not is_ingested(league_id):
        ingest_league(league_id)
        # Stamp here rather than inside ingest_league() so a fresh ingest
        # is never immediately followed by a redundant re-embed.
        _write_stamp(persist_dir, signals_fingerprint(signals_dir))
    else:
        _resync_signals_if_changed(league_id, raw_dir, persist_dir, signals_dir)
    config = load_league(str(league_id), raw_dir=raw_dir, persist_dir=persist_dir)
    warm_chroma(config.persist_dir)  # see the note above -- must happen before any request thread queries Chroma
    return config
