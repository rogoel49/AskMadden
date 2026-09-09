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
    via load_league() if the directory somehow holds a different league."""
    if refresh or not is_ingested(league_id):
        ingest_league(league_id)
    raw_dir, persist_dir = league_dirs(league_id)
    config = load_league(str(league_id), raw_dir=raw_dir, persist_dir=persist_dir)
    warm_chroma(config.persist_dir)  # see the note above -- must happen before any request thread queries Chroma
    return config
