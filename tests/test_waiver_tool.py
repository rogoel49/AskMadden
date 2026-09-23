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


def test_bid_guide_tiers_on_the_points_added_over_the_displaced_starter_and_caps_early():
    waivers = {"waiver_type": "faab", "faab_remaining": 100}
    te = {"position": "TE", "position_rank": 1, "key_stats": {"ppg": 10.8, "games_played": 2, "ppg_prior_season": None}}
    # the 2026-09-22 case: a 10.8 ppg TE over a 5.3 ppg starter in week 3 -> +5.5, a clear upgrade, but capped early
    g = recommend.bid_guide(te, waivers, {"name": "Tucker Kraft", "ppg": 5.3}, as_of_week=3)
    assert g["marginal_ppg"] == 5.5 and g["amount"] == [10, 15] and "capped through week 4" in g["tier"]
    # same pickup in week 8: the clear-upgrade tier uncapped
    assert recommend.bid_guide(te, waivers, {"name": "Tucker Kraft", "ppg": 5.3}, as_of_week=8)["amount"] == [10, 18]
    # not an upgrade over what you start -> minimum or pass
    g = recommend.bid_guide(te, waivers, {"name": "Trey McBride", "ppg": 14.0}, as_of_week=8)
    assert g["amount"] == [0, 2] and g["tier"].startswith("not an upgrade")
    # nobody at the position: the whole ppg is the gain
    assert recommend.bid_guide(te, waivers, None, as_of_week=8)["marginal_ppg"] == 10.8
    assert recommend.bid_guide(te, {"waiver_type": "rolling"}, None, 8) is None
    assert "not a market model" in recommend.bid_guide(te, waivers, None, 8)["basis"]


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
    assert chase["bid_guide"] is None  # the fixture has no stat lines, so no points on record -> no bid guide, never a guess
    assert out["your_weakest_starters"] == {}  # no points on record in this fixture -> nothing to displace
    assert chase["key_stats"]["red_zone_share"] is not None
    assert recommend.dispatch_tool("get_waiver_targets", {"position": "TE"}, ctx)["targets"] == []


def test_prompt_routes_pickup_questions_to_the_tool():
    prompt = recommend._build_system_prompt({"name": "L"}, {"rec": 0.5}, season=2026, as_of_week=3)
    assert "call get_waiver_targets" in prompt and "Never invent a bid amount" in prompt


# ---- competition-aware bids (2026-09-22): "everyone will want to hop on the backup" ----


def test_competition_counts_needing_teams_their_faab_and_sleeper_trending(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch)
    _faab(raw_dir, 200, 45)
    # a second team with no TE and lots of budget, and a third with a TE already
    teams = json.loads((raw_dir / "teams.json").read_text())
    teams["data"] += [
        {"roster_id": 2, "owner_id": "u2", "display_name": "rival", "team_name": "Rival", "players": ["s_barkley"], "starters": [], "settings": {"waiver_budget_used": 20}},
        {"roster_id": 3, "owner_id": "u3", "display_name": "set", "team_name": "Set", "players": ["s_te"], "starters": [], "settings": {"waiver_budget_used": 150}},
    ]
    (raw_dir / "teams.json").write_text(json.dumps(teams))
    players = json.loads((raw_dir / "players.json").read_text())
    players["data"]["s_te"] = {"full_name": "Some Tight End", "position": "TE", "team": "X"}
    players["data"]["s_hot"] = {"full_name": "Kaelon Black", "position": "RB", "team": "SF"}
    (raw_dir / "players.json").write_text(json.dumps(players))
    monkeypatch.setattr(recommend.sleeper, "fetch_trending_adds", lambda **kw: [{"player_id": "s_hot", "count": 3000000}, {"player_id": "zzz", "count": 10}])
    ctx = recommend.RecommendContext(raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, player_idx=None,
                                     league=load_league(VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir), roster_id="1", signals_dir=signals_dir)
    trending = recommend._trending_adds_by_name(ctx)
    assert trending == {"Kaelon Black": {"adds_24h": 3000000, "rank": 1}}

    # TE: the rival (no TE, $180 left) needs one; 'set' does not; my own team is excluded
    c = recommend.competition_for("TE", "Some Tight End", ctx, trending)
    assert c["teams_needing_position"] == 1 and c["max_competitor_faab"] == 180 and c["level"] == "medium"
    # the hot backup: top of Sleeper's most-added list -> high, whatever the league's needs
    c = recommend.competition_for("RB", "Kaelon Black", ctx, trending)
    assert c["level"] == "high" and c["trending_rank"] == 1 and c["trending_adds_24h"] == 3000000
    # a fetch failure degrades to no demand signal, never an error
    monkeypatch.setattr(recommend.sleeper, "fetch_trending_adds", lambda **kw: (_ for _ in ()).throw(RuntimeError("down")))
    assert recommend._trending_adds_by_name(ctx) == {}


def test_suggested_bid_moves_with_competition():
    guide = {"amount": [10, 15]}
    assert recommend.suggested_bid(guide, {"level": "low"})["amount"] == 10
    assert recommend.suggested_bid(guide, {"level": "medium"})["amount"] == 12
    hot = recommend.suggested_bid(guide, {"level": "high"})
    assert hot["amount"] == 22 and hot["why"].startswith("contested")  # 1.5x the top of the range
