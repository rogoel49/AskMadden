"""League-agnostic weekly stat lines (Phase 6, 2026-09-21).

One row per player per regular-season week with the raw counting stats a
fantasy league scores off (passing/rushing/receiving yards and TDs,
receptions, interceptions, 2-pt conversions, fumbles lost), from
nflverse's weekly player stats. No fantasy points in it: those depend on
the league, and this layer never sees a league (CLAUDE.md's one
architectural rule). Written to
data/processed/player_stats/player_stats_<season>.parquet by the refresh,
one file per season, regenerated each cycle. The per-league half -- turning
a stat line into points under a league's settings and rolling it up as-of
a week -- is src/reasoning/points_proxy.py.

Why this exists: the first real decision eval (52% on 400 dilemmas) used
a ranking built only from usage shares and an efficiency trend -- the
single strongest predictor of who scores more next week, how many points
each player has actually been scoring, was not in it at all. It is also
PROJECT_SPEC.md's Phase 6 signal: the cheapest defensible proxy for
"how good has this player been this season", named as a proxy
everywhere (`season_points_so_far_proxy`, never a bare `value`), and
explicitly NOT a projection, NOT position-scarcity-adjusted, NOT a trade
value model.

The as-of rule applies here exactly as everywhere else: a proxy computed
as of week N uses weeks 1..N-1 only.
"""
from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl
import polars as pl

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "player_stats"
REG_SEASON_TYPE = "REG"
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")

# The counting stats a fantasy league scores off. Which Sleeper key each
# one feeds, and what it is worth, is the per-league join's business
# (src/reasoning/points_proxy.py), not this layer's.
STAT_COLUMNS = [
    "passing_yards", "passing_tds", "passing_interceptions", "passing_2pt_conversions",
    "rushing_yards", "rushing_tds", "rushing_2pt_conversions",
    "receptions", "receiving_yards", "receiving_tds", "receiving_2pt_conversions",
    "rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost",
]
ID_COLUMNS = ["player_id", "player_display_name", "position", "team", "opponent_team", "season", "week"]


def fetch_weekly_stats(season: int) -> pl.DataFrame:
    """Regular-season weekly stat lines for skill-position players, from
    nflverse (network). Columns: ID_COLUMNS + STAT_COLUMNS, nulls as 0."""
    df = nfl.load_player_stats([season])
    df = df.filter(
        (pl.col("season_type") == REG_SEASON_TYPE) & pl.col("position").is_in(SKILL_POSITIONS)
    )
    return df.select(ID_COLUMNS + [pl.col(c).fill_null(0).cast(pl.Float64) for c in STAT_COLUMNS])


def stats_path(season: int, stats_dir: Path | None = None) -> Path:
    return (stats_dir or PROCESSED_DIR) / f"player_stats_{season}.parquet"


def write_weekly_stats(season: int, stats_dir: Path | None = None, df: pl.DataFrame | None = None) -> Path:
    path = stats_path(season, stats_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    (df if df is not None else fetch_weekly_stats(season)).write_parquet(path)
    return path


def load_weekly_stats(season: int, stats_dir: Path | None = None) -> pl.DataFrame | None:
    path = stats_path(season, stats_dir)
    return pl.read_parquet(path) if path.exists() else None
