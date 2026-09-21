"""src/signals/player_stats.py: league-agnostic stat lines, scored per
league at query time, rolled up strictly as-of a week."""
from __future__ import annotations

import polars as pl
import pytest

from evals import build_ground_truth as bgt
from src.reasoning import points_proxy as pp
from src.signals import player_stats as ps

HALF_PPR = {"rec": 0.5, "pass_td": 4, "pass_yd": 0.04, "pass_int": -1, "rush_td": 6, "rec_td": 6, "rec_yd": 0.1, "rush_yd": 0.1, "fum_lost": -2}


def _line(pid, week, **stats):
    base = {c: 0.0 for c in ps.STAT_COLUMNS}
    base.update(stats)
    return {"player_id": pid, "player_display_name": pid, "position": "RB", "team": "PHI", "season": 2024, "week": week, **base}


@pytest.fixture
def stats():
    return pl.DataFrame([
        _line("a", 1, rushing_yards=100, rushing_tds=1, receptions=2, receiving_yards=20),   # 10+6+1+2 = 19.0
        _line("a", 2, rushing_yards=50, rushing_fumbles_lost=1),                             # 5-2 = 3.0
        _line("a", 3, rushing_yards=80, rushing_tds=2),                                       # 8+12 = 20.0 (week 3 -- must be excluded as of week 3)
        _line("b", 1, receptions=8, receiving_yards=90, receiving_tds=1),                    # 4+9+6 = 19.0
    ])


def test_fantasy_points_is_the_same_mapping_ground_truth_uses(stats):
    row = stats.row(0, named=True)
    assert pp.fantasy_points(row, HALF_PPR) == 19.0
    assert bgt.compute_points(row, HALF_PPR) == pp.fantasy_points(row, HALF_PPR)
    assert bgt.STAT_TO_SCORING_KEY is pp.STAT_TO_SCORING_KEY


def test_points_by_week_matches_the_row_by_row_scoring(stats):
    scored = {(r["player_id"], r["week"]): r["points"] for r in pp.points_by_week(stats, HALF_PPR).to_dicts()}
    for row in stats.to_dicts():
        assert scored[(row["player_id"], row["week"])] == pytest.approx(pp.fantasy_points(row, HALF_PPR))


def test_season_points_proxy_only_sees_weeks_before_the_as_of_week(stats):
    proxy = pp.season_points_proxy(stats, HALF_PPR, as_of_week=3)
    assert proxy["a"] == {"games_played": 2, "season_points_so_far_proxy": 22.0, "ppg": 11.0}
    assert proxy["b"] == {"games_played": 1, "season_points_so_far_proxy": 19.0, "ppg": 19.0}
    whole = pp.season_points_proxy(stats, HALF_PPR)  # a finished season: everything counts
    assert whole["a"]["games_played"] == 3 and whole["a"]["season_points_so_far_proxy"] == 42.0
    assert pp.season_points_proxy(None, HALF_PPR, 3) == {}
    assert pp.season_points_proxy(stats, HALF_PPR, as_of_week=1) == {}


def test_full_ppr_scores_the_same_lines_differently(stats):
    half = pp.season_points_proxy(stats, HALF_PPR, as_of_week=2)["b"]["ppg"]
    full = pp.season_points_proxy(stats, {**HALF_PPR, "rec": 1.0}, as_of_week=2)["b"]["ppg"]
    assert full == half + 4.0  # 8 receptions x 0.5 more each


def test_write_and_load_roundtrip(tmp_path, stats, monkeypatch):
    monkeypatch.setattr(ps, "fetch_weekly_stats", lambda season: stats)
    path = ps.write_weekly_stats(2024, stats_dir=tmp_path)
    assert path == tmp_path / "player_stats_2024.parquet"
    assert ps.load_weekly_stats(2024, stats_dir=tmp_path).shape == stats.shape
    assert ps.load_weekly_stats(2023, stats_dir=tmp_path) is None
