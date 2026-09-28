"""Availability: how likely a player with an injury designation is to
play this week, from the official NFL injury report (nflverse's
`injuries` dataset: the final report_status -- Questionable / Doubtful /
Out -- and the practice_status behind it).

Why this exists (2026-09-27): "Questionable" is close to a coin flip on
its own, and the practice report moves it a lot -- over 2023-24 a
Questionable skill player who practiced in full played 70% of the time,
limited 59%, did not practice 42%. A lineup that treats every Q player
as healthy starts the wrong one every third time.

What this is: a historical play rate looked up by (designation,
practice bucket, position), fitted by evals/fit_play_rates.py on
2020-2024 and tested on 2025 (numbers in play_rates.json and TODO.md).
It is a base rate, not a read on the specific injury -- no beat
reporters, no per-player model -- and everything downstream says so.

League-agnostic by construction (NFL-wide report, NFL-wide rates); the
per-league use -- discounting a player's points by it in a lineup --
lives in src/reasoning/ranking.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "injuries"
PLAY_RATES_PATH = Path(__file__).resolve().parent / "play_rates.json"
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
REG_SEASON_TYPE = "REG"
COLUMNS = ["season", "week", "team", "gsis_id", "full_name", "position", "report_status", "practice_status",
           "report_primary_injury"]

# Designations this module assigns a probability to. Out/IR/etc. are
# "cannot play" and handled as such upstream (report.UNAVAILABLE_STATUSES).
PROBABILISTIC_STATUSES = ("Questionable", "Doubtful")
MIN_GROUP_N = 30  # below this the position-specific rate falls back to the all-position rate


def fetch_injury_reports(season: int) -> pl.DataFrame:
    """Regular-season injury reports for skill positions, one row per
    player-week (the final report). Network."""
    import nflreadpy as nfl  # lazy: the ranking imports this module and must stay cheap to import

    df = nfl.load_injuries([season])
    df = df.filter((pl.col("game_type") == REG_SEASON_TYPE) & pl.col("position").is_in(SKILL_POSITIONS))
    return (
        df.select([pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64)] + COLUMNS[2:])
        .unique(subset=["season", "week", "gsis_id"], keep="last")
    )


def reports_path(season: int, injuries_dir: Path | None = None) -> Path:
    return (injuries_dir or PROCESSED_DIR) / f"injuries_{season}.parquet"


def write_injury_reports(season: int, injuries_dir: Path | None = None, df: pl.DataFrame | None = None) -> Path:
    path = reports_path(season, injuries_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    (df if df is not None else fetch_injury_reports(season)).write_parquet(path)
    return path


def load_injury_reports(season: int, injuries_dir: Path | None = None) -> pl.DataFrame | None:
    path = reports_path(season, injuries_dir)
    return pl.read_parquet(path) if path.exists() else None


def practice_bucket(practice_status: str | None) -> str:
    """'full' / 'limited' / 'dnp' / 'unknown' from the report's wording."""
    s = (practice_status or "").lower()
    if s.startswith("full"):
        return "full"
    if s.startswith("limited"):
        return "limited"
    if s.startswith("did not"):
        return "dnp"
    return "unknown"


def practice_label(practice_status: str | None) -> str | None:
    return {"full": "full practice", "limited": "limited practice", "dnp": "did not practice", "unknown": None}[
        practice_bucket(practice_status)
    ]


_RATES: dict | None = None


def load_play_rates(path: Path = PLAY_RATES_PATH) -> dict:
    global _RATES
    if _RATES is None:
        _RATES = json.loads(path.read_text()) if path.exists() else {"rates": {}}
    return _RATES


def rate_key(report_status: str, bucket: str, position: str | None) -> str:
    return f"{report_status}|{bucket}|{position or '*'}"


def play_probability(
    report_status: str | None, practice_status: str | None, position: str | None, rates: dict | None = None
) -> float | None:
    """Historical share of players with this designation + practice status
    (+ position, when that group is big enough) who took an offensive snap
    that week. None when there is no designation to speak of (healthy) or
    the designation is not one this table covers."""
    if report_status not in PROBABILISTIC_STATUSES:
        return None
    table = (rates or load_play_rates()).get("rates") or {}
    bucket = practice_bucket(practice_status)
    for key in (rate_key(report_status, bucket, position), rate_key(report_status, bucket, None),
                rate_key(report_status, "unknown", None)):
        entry = table.get(key)
        if entry and entry.get("n", 0) >= MIN_GROUP_N:
            return round(float(entry["rate"]), 2)
    return None


def output_factor(report_status: str | None, practice_status: str | None, rates: dict | None = None) -> float | None:
    """When a player with this designation does play, the share of his
    season-to-date average he has historically scored (fitted alongside
    the play rates; None when the table has no such entry)."""
    if report_status not in PROBABILISTIC_STATUSES:
        return None
    table = (rates or load_play_rates()).get("factors") or {}
    for key in (rate_key(report_status, practice_bucket(practice_status), None), rate_key(report_status, "unknown", None)):
        entry = table.get(key)
        if entry and entry.get("n", 0) >= MIN_GROUP_N:
            return round(float(entry["factor"]), 2)
    return None


def expected_output(play_probability: float | None, factor: float | None) -> float | None:
    """Play rate x share-of-usual-output when playing: the multiplier a
    lineup applies to a designated player's usual production."""
    if play_probability is None:
        return None
    return round(play_probability * (factor if factor is not None else 1.0), 2)


def fields_for(report_status: str | None, practice_status: str | None, position: str | None, injury: str | None = None) -> dict:
    """The availability fields every ranking row / tool output carries."""
    p = play_probability(report_status, practice_status, position)
    k = output_factor(report_status, practice_status)
    return {
        "report_status": report_status,
        "practice_status": practice_label(practice_status),
        "injury": injury,
        "play_probability": p,
        "played_output_factor": k if p is not None else None,
        "expected_output": expected_output(p, k),
    }


AVAILABILITY_FIELDS = ("report_status", "practice_status", "injury", "play_probability", "played_output_factor", "expected_output")


def availability_for(df: pl.DataFrame | None, week: int) -> dict[str, dict]:
    """gsis_id -> {report_status, practice_status, injury, play_probability}
    for one week's report. Players not on the report are simply absent
    (healthy as far as the league office is concerned)."""
    if df is None or df.is_empty():
        return {}
    out: dict[str, dict] = {}
    for row in df.filter(pl.col("week") == week).iter_rows(named=True):
        if not row.get("gsis_id"):
            continue
        out[row["gsis_id"]] = fields_for(row.get("report_status"), row.get("practice_status"), row.get("position"),
                                         row.get("report_primary_injury"))
    return out


if __name__ == "__main__":  # python -m src.signals.availability --season 2026
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    print(write_injury_reports(args.season, args.out))
