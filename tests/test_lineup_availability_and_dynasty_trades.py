"""2026-09-23, from a dynasty-league user: chat started a player Sleeper
lists as NA on last season's numbers, and proposed a 32-year-old WR plus
a 33-year-old backup QB for a 26-year-old RB1 because the ppg added up."""
from __future__ import annotations

import json

from src.rag import player_index
from src.reasoning import ranking, recommend
from src.reasoning.league import load_league
from tests.test_league import _SEASON, _WEEK, VS30_ID, _setup


def test_current_season_players_outrank_stale_ones_once_the_season_is_underway():
    live = {"player_id": "a", "name": "Played", "position": "RB", "team": "X", "row": {"ppg": 6.0, "games_played": 2, "stale": False}}
    stale = {"player_id": "b", "name": "Sat out", "position": "RB", "team": "Y",
             "row": {"ppg": None, "ppg_prior_season": 20.0, "red_zone_share": 0.4, "stale": True, "source_season": _SEASON - 1}}
    out = ranking.rank_candidates([stale, live])
    assert [e["name"] for e in out["ranked"]] == ["Played", "Sat out"] and out["verdict"] == "clear"
    # week 1: everyone is stale -> the score decides as before
    other = {"player_id": "c", "name": "Also sat", "position": "RB", "team": "Z", "row": {"ppg_prior_season": 8.0, "stale": True}}
    assert [e["name"] for e in ranking.rank_candidates([other, stale])["ranked"]] == ["Sat out", "Also sat"]


def test_rank_players_leaves_out_players_sleeper_marks_unavailable(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch)
    players = json.loads((raw_dir / "players.json").read_text()); players["data"]["s_cook"]["injury_status"] = "NA"
    (raw_dir / "players.json").write_text(json.dumps(players))
    ctx = recommend.RecommendContext(raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK,
                                     player_idx=player_index.build_player_index(_SEASON),
                                     league=load_league(VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir), roster_id="1", signals_dir=signals_dir)
    out = recommend.dispatch_tool("rank_players", {"player_names": ["Saquon Barkley", "James Cook"]}, ctx)
    assert out["verdict"] == "insufficient_data"  # only Barkley is left to rank
    assert out["unavailable"] == [{"name": "James Cook", "reason": "unavailable", "injury_status": "NA",
                                   "note": "James Cook is listed NA by Sleeper -- not available to start this week."}]


def test_asset_stage_bands_by_position_and_experience():
    assert recommend.asset_stage("WR", 32, 11) == "declining"
    assert recommend.asset_stage("QB", 33, 10) == "aging"
    assert recommend.asset_stage("RB", 26, 4) == "prime"
    assert recommend.asset_stage("RB", 23, 1) == "young"
    assert recommend.asset_stage("RB", 28, 7) == "aging" and recommend.asset_stage("RB", 29, 8) == "declining"
    assert recommend.asset_stage("K", 30, 5) is None and recommend.asset_stage("RB", None, 5) is None


def test_league_rosters_carry_age_depth_and_stage_in_dynasty_leagues(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch)
    lg = json.loads((raw_dir / "league.json").read_text()); lg["data"].setdefault("settings", {})["type"] = 2
    (raw_dir / "league.json").write_text(json.dumps(lg))
    players = json.loads((raw_dir / "players.json").read_text())
    players["data"]["s_barkley"].update({"age": 29, "years_exp": 8, "depth_chart_order": 1}); players["data"]["s_cook"].update({"age": 26, "years_exp": 4, "depth_chart_order": 2})
    (raw_dir / "players.json").write_text(json.dumps(players))
    monkeypatch.setattr(recommend.nflverse, "bye_weeks", lambda season, schedules=None: {})
    ctx = recommend.RecommendContext(raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK,
                                     player_idx=player_index.build_player_index(_SEASON),
                                     league=load_league(VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir), roster_id="1", signals_dir=signals_dir)
    by_name = {p["name"]: p for p in recommend.dispatch_tool("get_league_rosters", {}, ctx)["teams"][0]["players"]}
    assert (by_name["Saquon Barkley"]["age"], by_name["Saquon Barkley"]["asset_stage"], by_name["Saquon Barkley"]["depth_chart_order"]) == (29, "declining", 1)
    assert by_name["James Cook"]["asset_stage"] == "prime" and by_name["James Cook"]["depth_chart_order"] == 2


def test_prompt_carries_the_trade_hard_rules_and_the_availability_rule():
    prompt = recommend._build_system_prompt({"name": "L"}, {"rec": 0.5}, season=2026, as_of_week=3, league_type="dynasty")
    for phrase in ("never add two players' points per game together", "why THAT manager says yes", "`asset_stage`",
                   "do not buy a 26-year-old RB1", "`depth_chart_order` 2+ means a backup", "under `unavailable`", "no snaps this"):
        assert phrase in prompt, phrase
