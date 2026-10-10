"""League-agnostic weekly stat lines for the two roster units the skill-
position signals never covered: kickers and team defenses (2026-10-06,
"what about the Bucs defense? that surely can be updated").

One row per kicker per regular-season week (nflverse weekly player
stats, position K: field goals made by distance, misses, blocks, PATs)
and one row per team defense per week (nflverse weekly team stats:
sacks, interceptions, forced fumbles, defensive and special-teams
touchdowns, safeties, blocked kicks; fumbles recovered from the
opponent's offensive line; points allowed from the schedule's final
score). No fantasy points here: what each count is worth is the per-
league join's business (src/reasoning/points_proxy.py), exactly as for
skill players. Written to data/processed/unit_stats/{kickers,defenses}_
<season>.parquet by the refresh, one file per season, regenerated each
cycle. A defense's id is its team abbreviation -- the same id Sleeper
uses for a DEF on a roster -- so no name resolution is needed.

Approximations, stated: points allowed is the opponent's final score
(Sleeper's own pts_allow excludes points the offense gives up on
returns; nflverse doesn't split that out), a blocked kick counts as a
miss, and yards-allowed scoring keys are not computed.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "unit_stats"
REG_SEASON_TYPE = "REG"

KICKER_ID_COLUMNS = ["player_id", "player_display_name", "position", "team", "opponent_team", "season", "week"]
KICKER_STAT_COLUMNS = [
    "fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49", "fg_made_50_59", "fg_made_60_",
    "fg_made", "fg_missed", "fg_blocked", "pat_made", "pat_missed", "pat_blocked",
]
DEFENSE_ID_COLUMNS = ["player_id", "player_display_name", "position", "team", "opponent_team", "season", "week"]
DEFENSE_STAT_COLUMNS = [
    "sacks", "interceptions", "fumbles_forced", "fumbles_recovered", "defensive_tds", "special_teams_tds",
    "safeties", "blocked_kicks", "points_allowed",
]


def fetch_kicker_stats(season: int) -> pl.DataFrame:
    """Regular-season weekly kicking lines (network)."""
    import nflreadpy as nfl

    df = nfl.load_player_stats([season])
    df = df.filter((pl.col("season_type") == REG_SEASON_TYPE) & (pl.col("position") == "K"))
    return df.select(KICKER_ID_COLUMNS + [pl.col(c).fill_null(0).cast(pl.Float64) for c in KICKER_STAT_COLUMNS])


def fetch_defense_stats(season: int) -> pl.DataFrame:
    """Regular-season weekly team-defense lines (network): the team's own
    defensive counts, the opponent's fumbles lost (= this defense's
    recoveries) and the opponent's final score."""
    import nflreadpy as nfl

    ts = nfl.load_team_stats([season], summary_level="week").filter(pl.col("season_type") == REG_SEASON_TYPE)
    sched = nfl.load_schedules([season]).select("game_id", "home_team", "away_team", "home_score", "away_score")
    return build_defense_stats(ts, sched)


def build_defense_stats(team_stats: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """Pure: team_stats (nflverse weekly team stats, one row per team per
    game) + schedules (final scores) -> DEFENSE columns."""
    own = team_stats.select(
        "season", "week", "game_id", "team", "opponent_team",
        pl.col("def_sacks").fill_null(0).cast(pl.Float64).alias("sacks"),
        pl.col("def_interceptions").fill_null(0).cast(pl.Float64).alias("interceptions"),
        pl.col("def_fumbles_forced").fill_null(0).cast(pl.Float64).alias("fumbles_forced"),
        pl.col("def_tds").fill_null(0).cast(pl.Float64).alias("defensive_tds"),
        pl.col("special_teams_tds").fill_null(0).cast(pl.Float64).alias("special_teams_tds"),
        pl.col("def_safeties").fill_null(0).cast(pl.Float64).alias("safeties"),
        (pl.col("def_punt_blocks").fill_null(0) + pl.col("def_pat_blocks").fill_null(0) + pl.col("def_fg_blocks").fill_null(0))
        .cast(pl.Float64).alias("blocked_kicks"),
    )
    # The opponent's offensive line, keyed so it joins onto OUR row.
    opp = team_stats.select(
        pl.col("game_id"), pl.col("team").alias("opponent_team"),
        (pl.col("rushing_fumbles_lost").fill_null(0) + pl.col("receiving_fumbles_lost").fill_null(0)
         + pl.col("sack_fumbles_lost").fill_null(0)).cast(pl.Float64).alias("fumbles_recovered"),
    )
    scores = pl.concat([
        schedules.select(pl.col("game_id"), pl.col("home_team").alias("team"), pl.col("away_score").cast(pl.Float64).alias("points_allowed")),
        schedules.select(pl.col("game_id"), pl.col("away_team").alias("team"), pl.col("home_score").cast(pl.Float64).alias("points_allowed")),
    ])
    out = own.join(opp, on=["game_id", "opponent_team"], how="left").join(scores, on=["game_id", "team"], how="left")
    return out.select(
        pl.col("team").alias("player_id"),
        pl.col("team").alias("player_display_name"),
        pl.lit("DEF").alias("position"),
        "team", "opponent_team", "season", "week",
        *[pl.col(c).fill_null(0).cast(pl.Float64) for c in DEFENSE_STAT_COLUMNS],
    )


def kickers_path(season: int, stats_dir: Path | None = None) -> Path:
    return (stats_dir or PROCESSED_DIR) / f"kickers_{season}.parquet"


def defenses_path(season: int, stats_dir: Path | None = None) -> Path:
    return (stats_dir or PROCESSED_DIR) / f"defenses_{season}.parquet"


def write_unit_stats(season: int, stats_dir: Path | None = None) -> tuple[Path, Path]:
    kp, dp = kickers_path(season, stats_dir), defenses_path(season, stats_dir)
    kp.parent.mkdir(parents=True, exist_ok=True)
    fetch_kicker_stats(season).write_parquet(kp)
    fetch_defense_stats(season).write_parquet(dp)
    return kp, dp


def load_kicker_stats(season: int, stats_dir: Path | None = None) -> pl.DataFrame | None:
    path = kickers_path(season, stats_dir)
    return pl.read_parquet(path) if path.exists() else None


def load_defense_stats(season: int, stats_dir: Path | None = None) -> pl.DataFrame | None:
    path = defenses_path(season, stats_dir)
    return pl.read_parquet(path) if path.exists() else None
