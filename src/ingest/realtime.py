"""Fast-refresh-tier ingest: data that changes multiple times a week (or
day-of) rather than once a week like play-by-play/NGS.

Covers two of the three items PROJECT_SPEC.md's Phase 2 checklist lists
for this module:
- Injury/inactive status: nflverse's injury report (updated throughout
  the week from official team reports), via nflreadpy.load_injuries.
- Weather: nflverse schedules' temp/wind/roof columns.

Line movement (opening vs. current spread/total, tracked through the
week) is NOT implemented here. nflverse's schedules dataset (see
src/ingest/odds.py) only carries the closing line, not a time series --
genuine line-movement tracking needs a live odds API with historical
snapshots (e.g. polling The Odds API on a schedule), which is a real
external dependency this project doesn't currently have. Flagging this
explicitly rather than faking movement data from a single closing-line
snapshot.

League-agnostic: NFL-wide injury/weather data, no Sleeper league or
scoring settings involved.
"""
from __future__ import annotations

import nflreadpy as nfl
import polars as pl


def fetch_injuries(season: int) -> pl.DataFrame:
    """Weekly injury report rows for `season`: player, team, week, and
    report_status ("Out", "Doubtful", "Questionable", or null if not
    listed)."""
    return nfl.load_injuries(seasons=[season])


def current_injury_status(injuries: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Each player's most recently reported status for `season`/`week`
    (the injury report is updated multiple times during the week; this
    keeps only the latest row per player for that week)."""
    return (
        injuries.filter((pl.col("season") == season) & (pl.col("week") == week))
        .sort("date_modified")
        .group_by("gsis_id", maintain_order=True)
        .last()
    )


def fetch_weather(schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per game: season, week, home/away team, temp (F), wind
    (mph), and roof (indoor games have null temp/wind since it isn't
    a game-script-relevant condition there)."""
    return schedules.select(
        "season", "week", "home_team", "away_team", "roof", "surface", "temp", "wind"
    )
