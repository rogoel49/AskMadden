"""Phase 5.2: roster_id as an explicit parameter, threaded from
recommend()/generate_report() down to src/rag/lookup.py, with the
MY_ROSTER_ID environment fallback preserved for the single-league CLI.
Plus recommend.load_dotenv_once()'s once-per-process behavior.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.rag import lookup
from src.reasoning import recommend, report
from tests.test_embed import _seed_raw_dir
from tests.test_league import VS30_ID, _setup, _submit_only_client, _tool_use_block, _RecordingClient, _SEASON, _WEEK

# ---- lookup.py: explicit roster_id beats the environment; None falls back to it ----


def test_explicit_roster_id_wins_over_MY_ROSTER_ID(tmp_path, monkeypatch):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)
    monkeypatch.setenv("MY_ROSTER_ID", "1")
    assert lookup.current_roster(raw_dir, roster_id=2)["display_name"] == "otheruser"
    assert lookup.current_roster(raw_dir, roster_id="2")["display_name"] == "otheruser"  # str or int
    assert {p["full_name"] for p in lookup.my_players(raw_dir, roster_id=2)} == {
        "Christian McCaffrey", "Brock Purdy", "George Kittle"
    }
    assert [p["full_name"] for p in lookup.my_players_by_position("TE", raw_dir, roster_id=2)] == ["George Kittle"]
    assert lookup.my_team_record(raw_dir, roster_id=2)["wins"] == 1


def test_none_roster_id_falls_back_to_MY_ROSTER_ID_exactly_as_before(tmp_path, monkeypatch):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)
    monkeypatch.setenv("MY_ROSTER_ID", "1")
    assert lookup.current_roster(raw_dir)["display_name"] == "rogoel49"
    assert lookup.current_roster(raw_dir, roster_id=None)["display_name"] == "rogoel49"
    assert lookup.current_roster(raw_dir, roster_id="")["display_name"] == "rogoel49"  # blank == not given


def test_explicit_roster_id_does_not_need_the_env_var_at_all(tmp_path, monkeypatch):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)
    monkeypatch.delenv("MY_ROSTER_ID", raising=False)
    assert lookup.current_roster(raw_dir, roster_id=1)["display_name"] == "rogoel49"


def test_unknown_explicit_roster_id_raises_the_same_error(tmp_path, monkeypatch):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)
    with pytest.raises(RuntimeError, match="No team found with roster_id='999'"):
        lookup.current_roster(raw_dir, roster_id=999)


def test_team_roster_for_roster_id(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)
    team = lookup.team_roster_for_roster_id(2, raw_dir)
    assert team["team_name"] == "Team Two" and team["counts_by_position"] == {"RB": 1, "QB": 1, "TE": 1}
    assert lookup.team_roster_for_roster_id(999, raw_dir) is None


# ---- recommend()/generate_report(): roster_id reaches the tools; CLI path unchanged ----


def test_recommend_passes_roster_id_to_the_my_tools_and_echoes_it(tmp_path, monkeypatch):
    raw_dir, persist_dir, _ = _setup(tmp_path, monkeypatch)  # _setup sets MY_ROSTER_ID=1
    monkeypatch.setenv("MY_ROSTER_ID", "999")  # would fail if the env var were consulted
    seen = {}
    real_dispatch = recommend.dispatch_tool

    def spy(name, tool_input, ctx):
        result = real_dispatch(name, tool_input, ctx)
        seen[name] = (ctx.roster_id, result)
        return result

    monkeypatch.setattr(recommend, "dispatch_tool", spy)
    client = _RecordingClient(
        [
            SimpleNamespace(content=[_tool_use_block("get_my_roster", {}), _tool_use_block("get_team_record", {}, "t2")]),
            SimpleNamespace(content=[_tool_use_block("submit_recommendation", {"recommendation": "ok", "reasoning": "ok"}, "t3")]),
        ]
    )

    result = recommend.recommend(
        "q", VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, client=client, roster_id=1
    )

    assert result["roster_id"] == "1"
    assert seen["get_my_roster"][0] == "1"
    assert {p["name"] for p in seen["get_my_roster"][1]["players"]} == {"Saquon Barkley", "James Cook"}
    assert seen["get_team_record"][1]["wins"] == 3


def test_recommend_without_roster_id_still_uses_MY_ROSTER_ID(tmp_path, monkeypatch):
    """The single-league CLI convention: no roster_id passed, env var set -- unchanged."""
    raw_dir, persist_dir, _ = _setup(tmp_path, monkeypatch)  # MY_ROSTER_ID=1
    client = _RecordingClient(
        [
            SimpleNamespace(content=[_tool_use_block("get_my_roster", {})]),
            SimpleNamespace(content=[_tool_use_block("submit_recommendation", {"recommendation": "ok", "reasoning": "ok"}, "t2")]),
        ]
    )
    result = recommend.recommend("q", VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, client=client)
    assert result["roster_id"] is None  # nothing explicit was passed...
    roster = next(c["result"] for c in result["tool_calls"] if c["name"] == "get_my_roster")
    assert {p["name"] for p in roster["players"]} == {"Saquon Barkley", "James Cook"}  # ...env fallback answered


def test_generate_report_uses_explicit_roster_id_and_ignores_env(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch)
    monkeypatch.delenv("MY_ROSTER_ID", raising=False)
    result = report.generate_report(
        "drop", VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK,
        signals_dir=signals_dir, roster_id=1,
    )
    assert result["roster_id"] == 1
    assert {e["name"] for e in result["entries"]} == {"Saquon Barkley", "James Cook"}


def test_generate_report_without_roster_id_still_requires_MY_ROSTER_ID(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch)
    monkeypatch.delenv("MY_ROSTER_ID", raising=False)
    with pytest.raises(RuntimeError, match="MY_ROSTER_ID is not set"):
        report.generate_report(
            "drop", VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
        )


# ---- load_dotenv_once(): once per process, not per call ----


def test_load_dotenv_runs_once_across_many_calls_in_one_process(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(recommend, "load_dotenv", lambda *a, **kw: calls.append(1))

    for _ in range(3):
        recommend.recommend("q", VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, client=_submit_only_client())
    for _ in range(2):
        report.generate_report("drop", VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir)

    assert len(calls) == 1  # five requests, one .env load


def test_load_dotenv_once_reports_whether_it_loaded():
    assert recommend.load_dotenv_once() is True  # conftest resets the memo per test
    assert recommend.load_dotenv_once() is False


def test_load_dotenv_once_still_honors_direct_import_first_call_guarantee(monkeypatch):
    """The PR #18 guarantee -- a direct import loads .env on first use --
    is preserved: the first call always loads."""
    calls = []
    monkeypatch.setattr(recommend, "load_dotenv", lambda *a, **kw: calls.append(1))
    recommend.load_dotenv_once()
    assert calls == [1]
