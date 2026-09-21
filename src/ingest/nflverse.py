"""nflverse ingest: weekly player box-score stats, play-by-play (with
EPA/WPA), and game schedules (with Vegas closing lines), all via
nflreadpy (free, pulls from nflverse's public GitHub data releases, no
API key).

Weekly stats build evals/ground_truth.jsonl -- the actual, measured
outcomes eval questions get graded against. Play-by-play and schedules
feed the Phase 2 signals layer (src/signals/matchup_signals.py) and the
Phase 2 odds ingest (src/ingest/odds.py). This module is league-agnostic
by construction: it just wraps nflverse's NFL-wide data, no Sleeper
league or scoring settings involved anywhere here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nflreadpy as nfl
import polars as pl

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "nflverse"


def bye_weeks(season: int, schedules=None) -> dict[str, int]:
    """Each team's regular-season bye week: the one week 1-18 the team
    has no game on the schedule. Empty dict if the schedule isn't there."""
    import polars as pl

    schedules = schedules if schedules is not None else fetch_schedules(season)
    reg = schedules.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
    weeks = set(reg["week"].unique().to_list())
    byes: dict[str, int] = {}
    for team in set(reg["home_team"].to_list()) | set(reg["away_team"].to_list()):
        played = set(reg.filter((pl.col("home_team") == team) | (pl.col("away_team") == team))["week"].to_list())
        off = sorted(weeks - played)
        if len(off) == 1:
            byes[team] = int(off[0])
    return byes


def fetch_weekly_stats(season: int) -> pl.DataFrame:
    """Real, measured weekly player stats for `season` (reg + postseason)."""
    return nfl.load_player_stats(seasons=[season], summary_level="week")


def save_weekly_stats(season: int, out_dir: Path = RAW_DIR) -> Path:
    df = fetch_weekly_stats(season)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"weekly_{season}.parquet"
    df.write_parquet(path)
    return path


def fetch_pbp(season: int) -> pl.DataFrame:
    """Play-by-play for `season`, including EPA/WPA and per-play personnel/
    formation columns. This is the raw input the signals layer aggregates
    into defense run-funnel rate, red zone role share, recent efficiency
    trend, and opponent-adjusted target share."""
    return nfl.load_pbp(seasons=[season])


def save_pbp(season: int, out_dir: Path = RAW_DIR) -> Path:
    df = fetch_pbp(season)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pbp_{season}.parquet"
    df.write_parquet(path)
    return path


def current_season() -> int:
    """nflverse's own notion of the current NFL season -- the calendar
    year from the Thursday after Labor Day onward, the previous year
    before that (so January playoffs still resolve to the right season).
    Used by src/scheduler/refresh.py to decide which season to refresh
    without hardcoding a year."""
    return int(nfl.get_current_season())


def fetch_schedules(season: int) -> pl.DataFrame:
    """Game schedules for `season`, including closing Vegas lines
    (spread_line, total_line, moneylines) and weather fields (temp, wind,
    roof). Feeds src/ingest/odds.py (game script / implied totals) and
    src/ingest/realtime.py (weather)."""
    return nfl.load_schedules(seasons=[season])


def save_schedules(season: int, out_dir: Path = RAW_DIR) -> Path:
    df = fetch_schedules(season)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"schedules_{season}.parquet"
    df.write_parquet(path)
    return path


def fetch_players() -> pl.DataFrame:
    """nflverse's full player reference table: gsis_id, display_name,
    position, latest_team, rookie_season/last_season, etc. -- not scoped
    to a single season (it's the whole historical database). Feeds
    src/rag/player_index.py's name -> gsis_id resolution, which filters
    to recent/relevant players itself; this function just wraps the raw
    pull."""
    return nfl.load_players()


def save_players(out_dir: Path = RAW_DIR) -> Path:
    df = fetch_players()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "players.parquet"
    df.write_parquet(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull nflverse weekly stats, play-by-play, schedules, and players")
    parser.add_argument("--season", type=int, help="required for weekly/pbp/schedules; ignored for players")
    parser.add_argument(
        "--what",
        choices=["weekly", "pbp", "schedules", "players", "all"],
        default="all",
        help="which dataset(s) to pull (default: all)",
    )
    args = parser.parse_args()

    savers = {"weekly": save_weekly_stats, "pbp": save_pbp, "schedules": save_schedules}
    targets = dict(savers) if args.what == "all" else ({} if args.what == "players" else {args.what: savers[args.what]})
    if targets and args.season is None:
        raise SystemExit("--season is required for weekly/pbp/schedules")
    for name, saver in targets.items():
        path = saver(args.season)
        print(f"wrote {name} -> {path}")

    if args.what in ("players", "all"):
        path = save_players()
        print(f"wrote players -> {path}")


if __name__ == "__main__":
    main()
