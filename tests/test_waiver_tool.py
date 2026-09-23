"""Chat can answer 'who should I pick up and how much should I bid' (2026-09-22)."""
from __future__ import annotations

import json
from pathlib import Path

from src.rag import lookup, player_index
from src.reasoning import recommend
from src.reasoning.league import load_league
from tests.test_league import _SEASON, _WEEK, VS30_ID, _setup


def _faab(raw_dir: Path, budget: int, used: int) -> None:
    lg = json.loads((raw_dir / "league.json").read_text()); lg["data"].setdefault("settings", {}).update({"waiver_type": 2, "waiver_budget": budget, "waiver_bid_min": 0, "waiver_day_of_week": 2})
    (raw_dir / "league.json").write_text(json.dumps(lg))
    teams = json.loads((raw_dir / "teams.json").read_text()); teams["data"][0]["settings"]["waiver_budget_used"] = used; teams["data"][0]["settings"]["waiver_position"] = 5
    (raw_dir / "teams.json").write_text(json.dumps(teams))


def test_waiver_status_reads_the_leagues_rules_and_this_teams_faab(tmp_path, monkeypatch):
    raw_dir, _, _ = _setup(tmp_path, monkeypatch)
    _faab(raw_dir, 200, 45)
    w = lookup.waiver_status("1", raw_dir)
    assert (w["waiver_type"], w["faab_budget"], w["faab_used"], w["faab_remaining"], w["waiver_position"]) == ("faab", 200, 45, 155, 5)


def test_bid_guide_is_a_labeled_rule_of_thumb_on_the_remaining_budget():
    waivers = {"waiver_type": "faab", "faab_remaining": 100}
    top = {"position": "TE", "position_rank": 1, "key_stats": {"ppg": 14.0}}
    g = recommend.bid_guide(top, waivers, needs=["TE"])
    assert g["amount"] == [30, 50] and "not a market model" in g["basis"]  # 15-25%, doubled for a need
    assert recommend.bid_guide(top, waivers, needs=[])["amount"] == [15, 25]
    assert recommend.bid_guide({"position": "RB", "position_rank": 4, "key_stats": {}}, waivers, [])["amount"] == [1, 3]
    assert recommend.bid_guide(top, {"waiver_type": "rolling"}, ["TE"]) is None


def test_get_waiver_targets_tool_returns_targets_needs_rules_and_bids(tmp_path, monkeypatch):
    from tests.test_report import _UNROSTERED_WR_ROW, _CHASE_SIGNAL_ROW  # an unrostered WR with signals
    import polars as pl
    from tests.test_league import _ALL_INDEX_ROWS, _ALL_SIGNALS
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch)
    _faab(raw_dir, 200, 45)
    import src.rag.player_index as pim
    monkeypatch.setattr(pim.nflverse, "fetch_players", lambda: pl.DataFrame(_ALL_INDEX_ROWS + [_UNROSTERED_WR_ROW]))
    ctx = recommend.RecommendContext(raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK,
                                     player_idx=player_index.build_player_index(_SEASON),
                                     league=load_league(VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir), roster_id="1", signals_dir=signals_dir)
    out = recommend.dispatch_tool("get_waiver_targets", {}, ctx)
    assert out["waivers"]["faab_remaining"] == 155
    assert "WR" in out["your_needs"]  # the fixture roster has no WR
    names = [t["name"] for t in out["targets"]]
    assert "Ja'Marr Chase" in names
    chase = next(t for t in out["targets"] if t["name"] == "Ja'Marr Chase")
    assert chase["bid_guide"]["amount"][1] > chase["bid_guide"]["amount"][0] > 0
    assert chase["key_stats"]["red_zone_share"] is not None
    assert recommend.dispatch_tool("get_waiver_targets", {"position": "TE"}, ctx)["targets"] == []


def test_prompt_routes_pickup_questions_to_the_tool():
    prompt = recommend._build_system_prompt({"name": "L"}, {"rec": 0.5}, season=2026, as_of_week=3)
    assert "call get_waiver_targets" in prompt and "Never invent a bid amount" in prompt
