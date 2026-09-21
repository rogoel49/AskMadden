"""A pre-draft league says so instead of ranking the whole NFL; the
league-roster tool carries the other manager's situation (needs/surplus
vs. real starting slots, injury status, bye week)."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.ingest import nflverse
from src.rag import player_index
from src.reasoning import recommend, report
from src.reasoning.league import load_league
from tests.test_league import VS30_ID, _SEASON, _WEEK, _setup


def _set_status(raw_dir: Path, status: str) -> None:
    envelope = json.loads((raw_dir / "league.json").read_text())
    envelope["data"]["status"] = status
    (raw_dir / "league.json").write_text(json.dumps(envelope))


def test_pre_draft_league_reports_say_so_instead_of_ranking(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch)
    _set_status(raw_dir, "pre_draft")
    assert load_league(VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir).status == "pre_draft"

    for kind in ("start_sit", "drop"):
        rep = report.generate_report(kind, VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir)
        assert rep["entries"] == [] and rep["notes"] == [report.PRE_DRAFT_NOTE]
    waivers = report.generate_report("waiver_pickups", VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir)
    assert waivers["notes"][0].startswith(report.PRE_DRAFT_NOTE) and "draft board" in waivers["notes"][0]


def test_in_season_league_is_unaffected(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch)
    _set_status(raw_dir, "in_season")
    rep = report.generate_report("start_sit", VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir)
    assert rep["entries"] and report.PRE_DRAFT_NOTE not in rep["notes"]


def test_bye_weeks_come_from_the_schedule():
    rows = []
    for week in range(1, 4):
        for home, away in (("PHI", "DAL"), ("BUF", "NYJ")):
            if week == 2 and home == "BUF":
                continue  # BUF and NYJ off in week 2
            rows.append({"season": 2026, "game_type": "REG", "week": week, "home_team": home, "away_team": away})
    byes = nflverse.bye_weeks(2026, schedules=pl.DataFrame(rows))
    assert byes == {"BUF": 2, "NYJ": 2}  # PHI/DAL play every week: no bye in this slice


def test_league_rosters_carry_needs_surplus_injury_and_bye(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch)
    # the fixture league: QB, RB, WR, WR, TE, FLEX, BN; roster = Barkley (RB, PHI) + Cook (RB, BUF)
    players = json.loads((raw_dir / "players.json").read_text())
    players["data"]["s_cook"]["injury_status"] = "Out"
    (raw_dir / "players.json").write_text(json.dumps(players))
    monkeypatch.setattr(nflverse, "bye_weeks", lambda season, schedules=None: {"PHI": _WEEK, "BUF": 9})

    ctx = recommend.RecommendContext(
        raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK,
        player_idx=player_index.build_player_index(_SEASON), league=load_league(VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir),
        roster_id="1", signals_dir=signals_dir,
    )
    out = recommend.dispatch_tool("get_league_rosters", {}, ctx)
    team = out["teams"][0]
    by_name = {p["name"]: p for p in team["players"]}
    assert by_name["James Cook"]["injury_status"] == "Out" and by_name["James Cook"]["bye_week"] == 9
    assert by_name["Saquon Barkley"]["on_bye_this_week"] is True
    assert team["starting_slots"] == {"QB": 1, "RB": 1, "WR": 2, "TE": 1}  # FLEX is not dedicated to anyone
    assert team["healthy_by_position"] == {"RB": 1}
    assert team["needs"] == ["QB", "RB", "TE", "WR"] and team["surplus"] == []
    assert "needs / surplus" in out["points_proxy_note"]


def test_prompt_tells_the_model_to_pitch_with_reasons_and_invite_refinement():
    prompt = recommend._build_system_prompt({"name": "L"}, {"rec": 0.5}, season=2026, as_of_week=3)
    for phrase in ("`needs` and", "`injury_status`", "`bye_week`", "on 1 game", "invite refinement explicitly", "no tight ends"):
        assert phrase in prompt, phrase
