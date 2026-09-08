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
from pathlib import Path

from src.ingest import sleeper
from src.rag import embed
from src.rag.embed import CHROMA_DIR, RAW_DIR, SIGNALS_DIR
from src.reasoning.league import LeagueConfig, load_league

LEAGUES_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "leagues"


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
    return load_league(str(league_id), raw_dir=raw_dir, persist_dir=persist_dir)
