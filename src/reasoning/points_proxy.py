"""The per-league join for Phase 6's points proxy.

Takes the league-agnostic weekly stat lines (src/signals/player_stats.py)
and one league's real Sleeper scoring settings, and produces the single
new signal PROJECT_SPEC.md's Phase 6 describes: fantasy points scored so
far this season, per player, under THIS league's scoring, strictly as-of
a week -- plus the same per game, and last season's. Named a proxy
everywhere (`season_points_so_far_proxy`, never a bare `value`): it is
how good a player has BEEN, not a projection, not position-scarcity
adjusted, not a trade value.

Why it exists: the first real decision eval (52% on 400 dilemmas) ran a
ranking built only from usage shares and an efficiency trend; points per
game -- the single strongest predictor of who scores more next week --
was not in it. evals/fit_ranking_weights.py measured that: ppg alone
scores 61.7% on held-out 2024 pairs, the old score 57.6%.

The scoring mapping here is the one evals/build_ground_truth.py has used
since Phase 1 (it imports it from here now), so ground truth and the
product can never disagree about what a stat line is worth. The as-of
rule applies exactly as everywhere else: a proxy as of week N uses weeks
1..N-1 only.
"""
from __future__ import annotations

import polars as pl

# Sleeper scoring_settings key each nflverse weekly-stats column feeds.
STAT_TO_SCORING_KEY = {
    "passing_yards": "pass_yd",
    "passing_tds": "pass_td",
    "passing_interceptions": "pass_int",
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


def fantasy_points(row: dict, scoring: dict) -> float:
    """One stat line scored under one league's Sleeper scoring_settings."""
    points = 0.0
    for stat_col, scoring_key in STAT_TO_SCORING_KEY.items():
        points += float(row.get(stat_col) or 0) * scoring.get(scoring_key, 0)
    fumbles_lost = sum(float(row.get(col) or 0) for col in FUMBLE_LOST_COLUMNS)
    points += fumbles_lost * scoring.get("fum_lost", 0)
    return round(points, 2)


def _points_expr(scoring: dict) -> pl.Expr:
    expr = pl.lit(0.0)
    for stat_col, scoring_key in STAT_TO_SCORING_KEY.items():
        weight = float(scoring.get(scoring_key, 0) or 0)
        if weight:
            expr = expr + pl.col(stat_col).fill_null(0) * weight
    fum = float(scoring.get("fum_lost", 0) or 0)
    if fum:
        expr = expr + sum(pl.col(c).fill_null(0) for c in FUMBLE_LOST_COLUMNS) * fum
    return expr.alias("points")


def points_by_week(stats: pl.DataFrame, scoring: dict) -> pl.DataFrame:
    """player_id, week, points -- every stat line scored under `scoring`."""
    return stats.select("player_id", "week", _points_expr(scoring))


def season_points_proxy(stats: pl.DataFrame | None, scoring: dict, as_of_week: int | None = None) -> dict[str, dict]:
    """Per player: {games_played, season_points_so_far_proxy, ppg} from the
    weeks strictly BEFORE as_of_week (None = the whole table, i.e. a
    finished season). games_played counts weeks with a stat line, which
    nflverse only emits for players who recorded a stat -- a player who
    dressed but never touched the ball that week is not counted as a
    game, so ppg is "per game with a touch". Empty dict when there is no
    table."""
    if stats is None or stats.is_empty():
        return {}
    scored = points_by_week(stats, scoring)
    if as_of_week is not None:
        scored = scored.filter(pl.col("week") < as_of_week)
    agg = scored.group_by("player_id").agg(
        pl.len().alias("games_played"), pl.col("points").sum().alias("season_points_so_far_proxy")
    )
    out: dict[str, dict] = {}
    for row in agg.to_dicts():
        games = int(row["games_played"])
        total = round(float(row["season_points_so_far_proxy"]), 2)
        out[row["player_id"]] = {
            "games_played": games,
            "season_points_so_far_proxy": total,
            "ppg": round(total / games, 2) if games else None,
        }
    return out


def points_allowed_by_position(stats: pl.DataFrame | None, scoring: dict, as_of_week: int | None = None) -> dict[tuple[str, str], dict]:
    """Per (defense, position): fantasy points allowed per game, under
    `scoring`, from weeks strictly before as_of_week -- the sum of every
    opposing player's points at that position in each game, averaged over
    the defense's games. Also a league-wide average per position, keyed
    ("*", position), so a defense can be read relative to it. Empty when
    there is no table or the stat lines don't carry opponent_team."""
    if stats is None or stats.is_empty() or "opponent_team" not in stats.columns:
        return {}
    scored = stats.select("opponent_team", "position", "week", _points_expr(scoring))
    if as_of_week is not None:
        scored = scored.filter(pl.col("week") < as_of_week)
    scored = scored.filter(pl.col("opponent_team").is_not_null())
    if scored.is_empty():
        return {}
    per_game = scored.group_by(["opponent_team", "position", "week"]).agg(pl.col("points").sum().alias("pts"))
    out: dict[tuple[str, str], dict] = {}
    for row in per_game.group_by(["opponent_team", "position"]).agg(pl.col("pts").mean().alias("allowed_pg"), pl.len().alias("games")).to_dicts():
        out[(row["opponent_team"], row["position"])] = {"allowed_pg": round(float(row["allowed_pg"]), 2), "games": int(row["games"])}
    for row in per_game.group_by("position").agg(pl.col("pts").mean().alias("allowed_pg")).to_dicts():
        out[("*", row["position"])] = {"allowed_pg": round(float(row["allowed_pg"]), 2), "games": None}
    return out


def trailing_ppg(stats: pl.DataFrame | None, scoring: dict, as_of_week: int, n_weeks: int = 3) -> dict[str, dict]:
    """Per player: points per game over the last n_weeks weeks before
    as_of_week (games with a stat line only)."""
    if stats is None or stats.is_empty():
        return {}
    scored = points_by_week(stats, scoring).filter((pl.col("week") < as_of_week) & (pl.col("week") >= as_of_week - n_weeks))
    return {
        r["player_id"]: {"ppg_last": round(float(r["ppg"]), 2), "games_last": int(r["games"])}
        for r in scored.group_by("player_id").agg(pl.col("points").mean().alias("ppg"), pl.len().alias("games")).to_dicts()
    }
