"""Generate evals/ground_truth.jsonl: real fantasy points per player per
eval week, computed from nflverse's measured box-score stats using this
league's *actual* Sleeper scoring settings (half-PPR + whatever bonuses
this league configured) -- not a generic PPR estimate. Never hand-
authored; every row traces back to a measured nflverse stat line.

Scope: offensive skill positions (QB/RB/WR/TE) only. Defense/kicker
scoring is out of scope for v1 -- see PROJECT_SPEC.md's phased plan.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ingest import nflverse

LEAGUE_JSON = Path(__file__).resolve().parents[1] / "data" / "raw" / "sleeper" / "league.json"
GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "ground_truth.jsonl"

# Sleeper scoring_settings key each nflverse weekly-stats column feeds.
STAT_TO_SCORING_KEY = {
    "passing_yards": "pass_yd",
    "passing_tds": "pass_td",
    "interceptions": "pass_int",
    "passing_2pt_conversions": "pass_2pt",
    "rushing_yards": "rush_yd",
    "rushing_tds": "rush_td",
    "rushing_2pt_conversions": "rush_2pt",
    "receptions": "rec",
    "receiving_yards": "rec_yd",
    "receiving_tds": "rec_td",
    "receiving_2pt_conversions": "rec_2pt",
}
FUMBLE_LOST_COLUMNS = ["rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost"]
SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]


def load_scoring_settings(league_json: Path = LEAGUE_JSON) -> dict:
    payload = json.loads(league_json.read_text())
    return payload["data"]["scoring_settings"]


def compute_points(row: dict, scoring: dict) -> float:
    points = 0.0
    for stat_col, scoring_key in STAT_TO_SCORING_KEY.items():
        points += float(row.get(stat_col) or 0) * scoring.get(scoring_key, 0)
    fumbles_lost = sum(float(row.get(col) or 0) for col in FUMBLE_LOST_COLUMNS)
    points += fumbles_lost * scoring.get("fum_lost", 0)
    return round(points, 2)


def build(season: int, weeks: list[int], scoring: dict | None = None) -> list[dict]:
    if scoring is None:
        scoring = load_scoring_settings()

    df = nflverse.fetch_weekly_stats(season)
    df = df[df["week"].isin(weeks) & df["position"].isin(SKILL_POSITIONS)]

    entries = []
    for _, row in df.iterrows():
        entries.append(
            {
                "season": season,
                "week": int(row["week"]),
                "player_name": row["player_display_name"],
                "position": row["position"],
                "nfl_team": row["recent_team"],
                "fantasy_points": compute_points(row, scoring),
                "scoring_format": "league_actual",
                "source": "nflverse",
                "measured": True,
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evals/ground_truth.jsonl from nflverse")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--weeks", type=int, nargs="+", required=True)
    args = parser.parse_args()

    entries = build(args.season, args.weeks)
    with GROUND_TRUTH_PATH.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"wrote {len(entries)} ground truth rows -> {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()
