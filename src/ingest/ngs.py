"""NGS (Next Gen Stats) ingest via nflreadpy -- free, no API key, pulls
from nflverse's mirror of the public Next Gen Stats site.

Feeds three of the Phase 2 signals (see PROJECT_SPEC.md's signals
table): aDOT / target depth distribution, RYOE, and CROE. League-agnostic
NFL-wide data -- no Sleeper league or scoring settings involved here.

Note on CROE: NGS does not publish a stat literally named "catch rate
over expected." The closest direct NGS receiving fields are
avg_yac_above_expectation (YAC earned vs. expectation) and avg_separation
(route-winning proxy) -- src/signals/matchup_signals.py uses those as the
CROE proxy and labels it as such rather than claiming a stat that isn't
actually published.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import nflreadpy as nfl
import polars as pl

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "ngs"
STAT_TYPES = ("passing", "receiving", "rushing")
StatType = Literal["passing", "receiving", "rushing"]


def fetch_ngs(season: int, stat_type: StatType) -> pl.DataFrame:
    """Weekly NGS rows for `season` and `stat_type` ("passing", "receiving",
    or "rushing")."""
    if stat_type not in STAT_TYPES:
        raise ValueError(f"stat_type must be one of {STAT_TYPES}, got {stat_type!r}")
    return nfl.load_nextgen_stats(seasons=[season], stat_type=stat_type)


def save_ngs(season: int, stat_type: StatType, out_dir: Path = RAW_DIR) -> Path:
    df = fetch_ngs(season, stat_type)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ngs_{stat_type}_{season}.parquet"
    df.write_parquet(path)
    return path


def save_all(season: int, out_dir: Path = RAW_DIR) -> dict[str, Path]:
    return {stat_type: save_ngs(season, stat_type, out_dir) for stat_type in STAT_TYPES}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull nflverse Next Gen Stats")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--stat-type", choices=[*STAT_TYPES, "all"], default="all")
    args = parser.parse_args()

    if args.stat_type == "all":
        for stat_type, path in save_all(args.season).items():
            print(f"wrote {stat_type} -> {path}")
    else:
        path = save_ngs(args.season, args.stat_type)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
