"""Generate evals/ground_truth.jsonl: real fantasy points per player per
eval week, computed from nflverse's measured box-score stats using the
league's *actual* Sleeper scoring settings (whatever that league
configured -- for Victorious Secret 3.0 that's half-PPR plus its
bonuses) -- not a generic PPR estimate. Never hand-authored; every row
traces back to a measured nflverse stat line.

Scoring settings are read from the same ingested league.json, via the
same shared reader (src/reasoning/league.py), that recommend() and
generate_report() use -- so the points a dilemma is graded against and
the scoring settings the agent is shown can't quietly come from two
different leagues. Pass league_id (CLI: --league-id, defaulting to
SLEEPER_LEAGUE_ID) to have that checked; this is the only place in the
project that applies scoring settings numerically (STAT_TO_SCORING_KEY
below), and it has always read them from the league's real data, never
a hardcoded format -- see TODO.md's Phase 5.1 entry.

Scope: offensive skill positions (QB/RB/WR/TE) only. Defense/kicker
scoring is out of scope for v1 -- see PROJECT_SPEC.md's phased plan.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

from src.ingest import nflverse
from src.reasoning.league import load_league, read_league_json

LEAGUE_JSON = Path(__file__).resolve().parents[1] / "data" / "raw" / "sleeper" / "league.json"
GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "ground_truth.jsonl"

# The stat -> Sleeper scoring mapping now lives in src/reasoning/points_proxy.py
# (Phase 6: the same mapping scores the per-game points proxy the ranking
# uses), re-exported here so ground truth and the product can never
# disagree about what a stat line is worth.
from src.reasoning.points_proxy import FUMBLE_LOST_COLUMNS, STAT_TO_SCORING_KEY, fantasy_points  # noqa: E402

SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]


def load_scoring_settings(league_json: Path = LEAGUE_JSON, league_id: str | None = None) -> dict:
    """The league's real scoring_settings from its ingested league.json.

    With league_id given, goes through src/reasoning/league.py's
    load_league() -- the exact same resolution (and same
    LeagueMismatchError if league_json belongs to another league) that
    recommend()/generate_report() perform, so ground truth and the agent
    are guaranteed to be scored/shown the same league's settings. Without
    it, reads the file as-is (the pre-Phase-5.1 behavior, kept for
    callers that only have a path)."""
    if league_id is not None:
        return load_league(league_id, raw_dir=league_json.parent).scoring_settings
    return read_league_json(league_json)["scoring_settings"]


def compute_points(row: dict, scoring: dict) -> float:
    return fantasy_points(row, scoring)


def build(season: int, weeks: list[int], scoring: dict | None = None, league_id: str | None = None) -> list[dict]:
    if scoring is None:
        scoring = load_scoring_settings(league_id=league_id)

    df = nflverse.fetch_weekly_stats(season)
    df = df.filter(pl.col("week").is_in(weeks) & pl.col("position").is_in(SKILL_POSITIONS))

    entries = []
    for row in df.iter_rows(named=True):
        entries.append(
            {
                "season": season,
                "week": int(row["week"]),
                "player_name": row["player_display_name"],
                "position": row["position"],
                "nfl_team": row["team"],
                "fantasy_points": compute_points(row, scoring),
                "scoring_format": "league_actual",
                "source": "nflverse",
                "measured": True,
            }
        )
    return entries


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate evals/ground_truth.jsonl from nflverse")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--weeks", type=int, nargs="+", required=True)
    parser.add_argument(
        "--league-id",
        default=os.environ.get("SLEEPER_LEAGUE_ID"),
        help=(
            "the Sleeper league whose scoring settings to apply (defaults to SLEEPER_LEAGUE_ID; verified "
            "against data/raw/sleeper/league.json when given)"
        ),
    )
    args = parser.parse_args()

    entries = build(args.season, args.weeks, league_id=args.league_id)
    with GROUND_TRUTH_PATH.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"wrote {len(entries)} ground truth rows -> {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()
