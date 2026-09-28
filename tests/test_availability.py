"""Availability (2026-09-27): play probability from the injury report,
folded into the ranking, the start/sit report and Chat's tools."""
from __future__ import annotations

import polars as pl

from src.reasoning import ranking
from src.reasoning.ranking import rank_candidates
from src.signals import availability as av

RATES = {
    "rates": {
        "Questionable|limited|*": {"n": 800, "rate": 0.645},
        "Questionable|limited|QB": {"n": 150, "rate": 0.41},
        "Questionable|limited|TE": {"n": 12, "rate": 0.9},  # too small: falls back to the pooled rate
        "Questionable|unknown|*": {"n": 34, "rate": 0.676},
        "Doubtful|dnp|*": {"n": 200, "rate": 0.01},
    },
    "factors": {"Questionable|limited|*": {"n": 700, "factor": 0.85}, "Questionable|unknown|*": {"n": 1000, "factor": 0.86}},
}


def test_practice_bucket_and_label():
    assert av.practice_bucket("Limited Participation in Practice") == "limited"
    assert av.practice_bucket("Did Not Participate In Practice") == "dnp"
    assert av.practice_bucket("Full Participation in Practice") == "full"
    assert av.practice_bucket(None) == "unknown"
    assert av.practice_label("Did Not Participate In Practice") == "did not practice"


def test_play_probability_uses_the_position_rate_only_when_the_group_is_big_enough():
    assert av.play_probability("Questionable", "Limited Participation in Practice", "QB", RATES) == 0.41
    assert av.play_probability("Questionable", "Limited Participation in Practice", "TE", RATES) == 0.65  # pooled
    assert av.play_probability("Questionable", None, "RB", RATES) == 0.68  # Sleeper designation, no practice report
    assert av.play_probability("Doubtful", "Did Not Participate In Practice", "WR", RATES) == 0.01
    assert av.play_probability(None, "Full Participation in Practice", "WR", RATES) is None  # healthy
    assert av.play_probability("Out", None, "WR", RATES) is None  # handled as unavailable upstream


def test_expected_output_is_play_rate_times_output_when_playing():
    assert av.expected_output(0.645, 0.85) == 0.55
    assert av.expected_output(0.01, None) == 0.01
    assert av.expected_output(None, 0.85) is None


def test_availability_for_reads_one_weeks_report(monkeypatch):
    monkeypatch.setattr(av, "load_play_rates", lambda path=None: RATES)
    df = pl.DataFrame({
        "season": [2026, 2026], "week": [3, 4], "team": ["BUF", "BUF"], "gsis_id": ["00-1", "00-1"],
        "full_name": ["DJ Moore"] * 2, "position": ["WR"] * 2,
        "report_status": ["Questionable", None], "practice_status": ["Limited Participation in Practice", "Full Participation in Practice"],
        "report_primary_injury": ["Ankle", None],
    })
    week3 = av.availability_for(df, 3)["00-1"]
    assert week3["play_probability"] == 0.65 and week3["practice_status"] == "limited practice" and week3["injury"] == "Ankle"
    assert week3["expected_output"] == 0.55
    assert av.availability_for(df, 4)["00-1"]["play_probability"] is None  # off the designation list
    assert av.availability_for(None, 3) == {}


def _row(pid, ppg, red_zone=0.2, **extra):
    return {"player_id": pid, "position": "RB", "ppg": ppg, "games_played": 3, "target_share": 0.15, "targets": 12, "team_targets": 80,
            "red_zone_share": red_zone, "epa_trend": None, "epa_trend_plays": 0, "epa_baseline_plays": 0,
            "season_plays": 100, "stale": False, "source_season": None, "source_as_of_week": None, **extra}


def test_ranking_discounts_a_designated_player_and_leaves_healthy_order_alone(monkeypatch):
    monkeypatch.setattr(av, "load_play_rates", lambda path=None: RATES)
    healthy = [{"player_id": "a", "name": "A", "position": "RB", "team": "X", "row": _row("a", 14.0)},
               {"player_id": "b", "name": "B", "position": "RB", "team": "X", "row": _row("b", 12.0)},
               {"player_id": "c", "name": "C", "position": "RB", "team": "X", "row": _row("c", 7.0)}]
    base = rank_candidates(healthy)
    assert [e["name"] for e in base["ranked"]] == ["A", "B", "C"]
    assert all(e["play_probability"] is None for e in base["ranked"])
    assert base["ranked"][0]["win_probability_vs_next"] == ranking.pairwise_confidence(
        base["ranked"][0]["opportunity_score"], base["ranked"][1]["opportunity_score"])

    # A on this week's report, Questionable after a limited week: counted at 55% of himself, i.e. like a
    # 7.7-point back -> behind B (12) and still ahead of C (7).
    q_row = {**_row("a", 14.0), "report_status": "Questionable", "practice_status": "limited practice", "injury": "Knee",
             "play_probability": 0.65, "played_output_factor": 0.85, "expected_output": 0.55}
    out = rank_candidates([{**healthy[0], "row": q_row}, healthy[1], healthy[2]])
    assert [e["name"] for e in out["ranked"]] == ["B", "A", "C"]
    a = out["ranked"][1]
    assert (a["play_probability"], a["expected_output"], a["practice_status"]) == (0.65, 0.55, "limited practice")
    assert a["key_stats"]["play_probability"] == 0.65
    # B's edge over a discounted A is larger than over a healthy A.
    assert out["ranked"][0]["win_probability_vs_next"] > base["ranked"][0]["win_probability_vs_next"]
    assert "availability_description" in out

    # Doubtful and did not practice: counted at ~1% (like a 0.1-point back) -> last, whatever his numbers.
    d_row = {**q_row, "report_status": "Doubtful", "play_probability": 0.01, "played_output_factor": None, "expected_output": 0.01}
    assert [e["name"] for e in rank_candidates([{**healthy[0], "row": d_row}, healthy[1], healthy[2]])["ranked"]] == ["B", "C", "A"]


def test_sleepers_designation_alone_gives_the_pooled_rate(monkeypatch):
    monkeypatch.setattr(av, "load_play_rates", lambda path=None: RATES)
    out = rank_candidates([
        {"player_id": "a", "name": "A", "position": "RB", "team": "X", "row": _row("a", 14.0), "injury_status": "Questionable"},
        {"player_id": "b", "name": "B", "position": "RB", "team": "X", "row": _row("b", 12.0)},
    ])
    a = next(e for e in out["ranked"] if e["name"] == "A")
    assert a["play_probability"] == 0.68 and a["practice_status"] is None and a["expected_output"] == 0.58


def test_pairwise_confidence_with_availability():
    # equal standing: healthy vs a player who plays 60% of the time -> the healthy one is favored
    assert ranking.pairwise_confidence(1.0, 1.0) == 0.5
    assert ranking.pairwise_confidence(1.0, 1.0, 1.0, 0.6) == 0.7   # 0.6*0.5 + 0.4
    assert ranking.pairwise_confidence(1.0, 1.0, 0.6, 1.0) == 0.3   # 0.6*0.5
    assert ranking.pairwise_confidence(9.0, 0.0, 0.5, 1.0) == 0.5   # a great player who plays half the time = a coin flip


def test_adjusted_score_removes_the_points_not_expected():
    row = _row("a", 14.0)
    assert ranking.availability_adjusted_score(1.0, row, 1.0) == 1.0
    # 45% of 14 points removed at the fitted points weight
    assert abs(ranking.availability_adjusted_score(1.0, row, 0.55) - (1.0 - ranking.PPG_WEIGHT * 0.45 * 14.0)) < 1e-9
    # no points on record: scaled in log space instead, still a penalty
    assert ranking.availability_adjusted_score(1.0, {"player_id": "x"}, 0.55) < 1.0
