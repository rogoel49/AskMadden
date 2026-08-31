"""nflverse ingest: real weekly player box-score stats via nflreadpy
(free, pulls from nflverse's public GitHub data releases, no API key).

Used to build evals/ground_truth.jsonl -- the actual, measured outcomes
eval questions get graded against. Never used to power recommendations
directly; that's the signals layer (Phase 2) and reasoning layer
(Phase 3), both out of scope here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nflreadpy as nfl
import polars as pl

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "nflverse"


def fetch_weekly_stats(season: int) -> pl.DataFrame:
    """Real, measured weekly player stats for `season` (reg + postseason)."""
    return nfl.load_player_stats(seasons=[season], summary_level="week")


def save_weekly_stats(season: int, out_dir: Path = RAW_DIR) -> Path:
    df = fetch_weekly_stats(season)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"weekly_{season}.parquet"
    df.write_parquet(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull nflverse weekly player stats")
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()
    path = save_weekly_stats(args.season)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
