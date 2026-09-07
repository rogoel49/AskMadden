"""Phase 5.1: the per-league configuration a recommend()/generate_report()
call operates on, resolved from an explicit Sleeper league_id.

Before this module, "which league" was never stated anywhere in the
reasoning layer: recommend() and generate_report() read whatever
`src.ingest.sleeper` last wrote to data/raw/sleeper/ and trusted it.
That was fine with one league and silently wrong with two -- ingest
league B, ask a question "about" league A, get league B's roster and
scoring settings back with no error. This module makes the league an
explicit, *verified* input: load_league() reads the ingested league.json
and refuses (LeagueMismatchError) if its Sleeper `league_id` isn't the
one the caller asked for, so a wrong-league answer becomes a loud error
instead of a quiet one.

**Where scoring settings come from -- and what Phase 5.1 actually found.**
PROJECT_SPEC.md's Phase 5 text says recommend.py "currently assumes
half-PPR". Traced in the Phase 5.1 session, that turned out to be a
single-league blind spot, not a magic number: recommend.py has read the
league's real Sleeper `scoring_settings` from league.json since Phase 3
(and put them in the system prompt verbatim), report.py never reads
scoring at all (its ranking is a scoring-format-independent weighted sum
of usage/efficiency signals), and evals/build_ground_truth.py reads the
same league.json key. Nothing anywhere had 0.5 (or any other scoring
weight) hardcoded -- the code was only ever *shown* one league's
settings. So `LeagueConfig.scoring_settings` below is just that same
real read, routed through one place so every consumer can't drift from
one another. See TODO.md's Phase 5.1 entry.

**On-disk convention (unchanged here, deliberately).** raw_dir /
persist_dir are still where a league's ingested Sleeper JSON and Chroma
collection live; they default to the same flat data/raw/sleeper/ and
data/chroma/ that `src.ingest.sleeper` / `src.rag.embed` write today,
so a single-league setup (Rohan's machine, with SLEEPER_LEAGUE_ID in
.env) keeps working unchanged. Per-league storage layout (one raw_dir/
persist_dir per league_id, and who decides which is active) is Phase
5.2's job -- this module only requires that whichever directory is
passed actually holds the league that was asked for.

**Still implicit after this module -- flagged, not fixed here:**
`MY_ROSTER_ID` (which roster in the league is "mine") is read from the
environment inside src/rag/lookup.py's current_roster(). A roster_id is
per-league, so in a multi-league world it belongs alongside league_id
as an explicit input; that change lives in src/rag/lookup.py, which
Phase 5.1 was scoped not to touch. Phase 5.2's "league_id + your team
within it" session model is where it gets parameterized.

Per CLAUDE.md's architectural principle this is the per-league join
layer: nothing here is imported by src/signals/ or src/rag/, and nothing
league-specific flows back into them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.rag.embed import CHROMA_DIR, RAW_DIR

INGEST_HINT = "run `python -m src.ingest.sleeper --league-id {league_id}` first"


class LeagueMismatchError(RuntimeError):
    """The ingested data in raw_dir belongs to a different Sleeper league
    than the one the caller asked for. Raised instead of silently
    answering from the wrong league's roster/scoring settings."""


@dataclass(frozen=True)
class LeagueConfig:
    league_id: str
    name: str | None
    season: str | None
    # The league's real Sleeper scoring_settings, exactly as ingested
    # (e.g. {"rec": 0.5, "pass_td": 4, ...}) -- never a default, never a
    # format label like "half-PPR" that something downstream has to
    # interpret.
    scoring_settings: dict[str, Any]
    roster_positions: list[str] = field(default_factory=list)
    raw_dir: Path = RAW_DIR
    persist_dir: Path = CHROMA_DIR


def read_league_json(path: Path) -> dict:
    """The raw Sleeper league object, unwrapped from the
    {"fetched_at", "source", "data"} envelope src/ingest/sleeper.py
    writes. Shared by load_league() and evals/build_ground_truth.py so
    both read scoring settings the same way from the same file."""
    return json.loads(path.read_text())["data"]


def load_league(
    league_id: str,
    raw_dir: Path = RAW_DIR,
    persist_dir: Path = CHROMA_DIR,
) -> LeagueConfig:
    """Resolve league_id to its ingested configuration in raw_dir.

    Raises RuntimeError if raw_dir has no ingested league at all, and
    LeagueMismatchError if it holds a *different* league than league_id
    -- the caller asked about one league and the data on disk is
    another's, which is exactly the silent failure an explicit league_id
    exists to prevent.
    """
    if league_id is None or not str(league_id).strip():
        raise ValueError("league_id is required -- pass the Sleeper league ID recommend()/generate_report() should answer for.")
    league_id = str(league_id).strip()

    league_path = raw_dir / "league.json"
    if not league_path.exists():
        raise RuntimeError(
            f"{league_path} doesn't exist -- {INGEST_HINT.format(league_id=league_id)}."
        )
    league = read_league_json(league_path)

    on_disk_id = league.get("league_id")
    if on_disk_id is None:
        raise LeagueMismatchError(
            f"{league_path} has no league_id field, so it can't be confirmed as league {league_id} -- "
            f"{INGEST_HINT.format(league_id=league_id)} to re-ingest it."
        )
    if str(on_disk_id) != league_id:
        raise LeagueMismatchError(
            f"{league_path} holds Sleeper league {on_disk_id!r} ({league.get('name')!r}), not the requested "
            f"league {league_id!r} -- refusing to answer from another league's roster/scoring settings. "
            f"Either pass league_id={on_disk_id!r}, or {INGEST_HINT.format(league_id=league_id)} "
            "(ingest writes one league per raw_dir; per-league storage is Phase 5.2)."
        )

    return LeagueConfig(
        league_id=league_id,
        name=league.get("name"),
        season=league.get("season"),
        scoring_settings=dict(league.get("scoring_settings") or {}),
        roster_positions=list(league.get("roster_positions") or []),
        raw_dir=raw_dir,
        persist_dir=persist_dir,
    )
