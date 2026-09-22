"""Step 1 of the accuracy plan (confidence on every verdict) and the
Malachi Fields RCA (the model passed a bare first name; tools now settle
an ambiguity from the user's own question)."""
from __future__ import annotations

import polars as pl
import pytest

from src.rag import player_index
from src.reasoning import ranking, recommend, report
from tests.test_league import _SEASON, _WEEK, VS30_ID, _setup


def test_pairwise_confidence_is_tempered_capped_and_labeled():
    assert ranking.pairwise_confidence(1.0, 1.0) == 0.5
    import math
    assert ranking.pairwise_confidence(1.5, 1.0) == pytest.approx(1 / (1 + math.exp(-0.85 * 0.5)), abs=1e-3)
    assert ranking.pairwise_confidence(10.0, 0.0) == ranking.CONFIDENCE_CAP  # never more than 85% on a one-week call
    assert ranking.pairwise_confidence(0.0, 10.0) < 0.01
    assert [ranking.confidence_label(p) for p in (0.51, 0.6, 0.7, 0.85, None)] == ["coin flip", "lean", "clear", "strong", None]


def test_rank_candidates_emits_a_probability_for_each_player_over_the_next():
    a = {"player_id": "a", "name": "A", "position": "RB", "team": "X", "row": {"ppg": 20.0, "games_played": 4}}
    b = {"player_id": "b", "name": "B", "position": "RB", "team": "Y", "row": {"ppg": 12.0, "games_played": 4}}
    c = {"player_id": "c", "name": "C", "position": "RB", "team": "Z", "row": {"ppg": 11.5, "games_played": 4}}
    out = ranking.rank_candidates([c, a, b])
    assert [e["name"] for e in out["ranked"]] == ["A", "B", "C"]
    assert out["confidence"] == out["ranked"][0]["win_probability_vs_next"] > 0.6  # 8 ppg edge
    assert out["ranked"][1]["confidence_label"] == "coin flip"  # 0.5 ppg edge
    assert out["ranked"][2]["win_probability_vs_next"] is None
    assert "capped at 85%" in out["confidence_description"]


def test_start_sit_entries_and_rank_players_carry_the_confidence(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch)
    rep = report.generate_report("start_sit", VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir)
    starter = rep["entries"][0]["recommended_starter"]
    assert 0.5 <= starter["win_probability_vs_next"] <= ranking.CONFIDENCE_CAP and starter["confidence_label"]
    from src.reasoning.league import load_league

    ctx = recommend.RecommendContext(raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK,
                                     player_idx=player_index.build_player_index(_SEASON),
                                     league=load_league(VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir), roster_id="1", signals_dir=signals_dir)
    tool = recommend.dispatch_tool("rank_players", {"player_names": ["Saquon Barkley", "James Cook"]}, ctx)
    assert tool["confidence"] == starter["win_probability_vs_next"]  # Chat and Feed agree on the number too
    assert tool["confidence_label"] == starter["confidence_label"]


def test_an_ambiguous_first_name_resolves_to_the_player_the_user_actually_typed(monkeypatch):
    idx = pl.DataFrame([
        {"player_id": "00-0039920", "player_name": "Malachi Corley", "position": "WR", "team": "CLE"},
        {"player_id": "00-0041042", "player_name": "Malachi Fields", "position": "WR", "team": "NYG"},
        {"player_id": "00-0036849", "player_name": "Tutu Atwell", "position": "WR", "team": "LA"},
    ])
    assert player_index.resolve_player("Malachi", idx).match_type == "ambiguous"  # the tool alone can't know
    ctx = recommend.RecommendContext(raw_dir=None, persist_dir=None, season=2026, as_of_week=2, player_idx=idx,
                                     question="Because Puka is out, should I hypothetically start Tutu or Malachi Fields?")
    r = recommend.resolve_named_player("Malachi", ctx)
    assert r.match_type == "exact" and r.candidates[0].player_name == "Malachi Fields"
    # no help from the question -> still ambiguous, never a guess
    ctx.question = "who should I start, Tutu or Malachi?"
    assert recommend.resolve_named_player("Malachi", ctx).match_type == "ambiguous"
    ctx.question = "Malachi Corley or Malachi Fields?"
    assert recommend.resolve_named_player("Malachi", ctx).match_type == "ambiguous"  # both named: still ask


def test_prompt_demands_full_names_and_stated_confidence():
    prompt = recommend._build_system_prompt({"name": "L"}, {"rec": 0.5}, season=2026, as_of_week=3)
    assert "EXACTLY as the user wrote it, surname included" in prompt
    assert "never run the comparison with a candidate you picked yourself" in prompt
    assert "State the confidence with every start/sit verdict" in prompt
