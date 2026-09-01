"""Odds / game-script ingest.

PROJECT_SPEC.md's signals table lists "game script / implied total" as
sourced from "odds API (free tier)". In practice nflverse's own
schedules dataset (src/ingest/nflverse.fetch_schedules) already carries
closing Vegas lines (spread_line, total_line, moneylines) for every game,
free and with no API key or rate limit -- so this module derives implied
team totals from that instead of adding a second, redundant data source.

Known limitation: nflverse schedules only carry the *closing* line, not
a time series of line movement. Real line movement (opening vs. current,
tracked through the week) would need a live odds API with historical
snapshots -- out of scope for now; src/ingest/realtime.py documents this
gap rather than faking movement data.

League-agnostic: this is NFL-wide schedule data, no Sleeper league or
scoring settings involved.
"""
from __future__ import annotations

import polars as pl


def implied_team_totals(schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per team per game: season, week, team, opponent, is_home,
    spread_line, total_line, and implied_total (the team's Vegas-implied
    point total, derived from the game total and spread).

    nflverse convention (verified against actual 2024 results): spread_line
    is the home team's expected margin -- positive means the home team is
    favored by that many points, negative means the home team is the
    underdog. Implied totals: home = total/2 + spread/2, away =
    total/2 - spread/2.
    """
    home = schedules.select(
        pl.col("season"),
        pl.col("week"),
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.lit(True).alias("is_home"),
        pl.col("spread_line"),
        pl.col("total_line"),
        (pl.col("total_line") / 2 + pl.col("spread_line") / 2).alias("implied_total"),
    )
    away = schedules.select(
        pl.col("season"),
        pl.col("week"),
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.lit(False).alias("is_home"),
        pl.col("spread_line"),
        pl.col("total_line"),
        (pl.col("total_line") / 2 - pl.col("spread_line") / 2).alias("implied_total"),
    )
    return pl.concat([home, away]).sort(["season", "week", "team"])
