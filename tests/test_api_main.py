"""The FastAPI app end to end through TestClient -- no network.

What's real here: the app's routing/validation, SQLite storage (a temp
file), Phase 5.1's league verification, the actual recommend() tool-use
loop and its real tools, and generate_report()'s real ranking, all
against real-shaped fixture data (real nflverse identities, real 2024
week-5 signal values). What's faked, and only at these boundaries:
  - Sleeper login (auth.resolve_user_leagues) -- Sleeper is blocked here.
  - League ingest (leagues.ingest_league) -- seeds fixture data instead.
  - The Claude model -- a scripted client that decides which tools to
    call; the tools themselves run for real.
  - nflverse's player list -- the same monkeypatch every other test uses.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from anthropic.types import TextBlock, ToolUseBlock
from fastapi.testclient import TestClient

from src.api import auth, leagues, main
from src.api.storage import Storage
from src.rag import embed
from tests.test_league import (
    _ALL_INDEX_ROWS,
    _ALL_SIGNALS,
    _BARKLEY_SIGNAL,
    _CHASE_SIGNAL,
    _COOK_SIGNAL,
    _SEASON,
    _WEEK,
    HALF_PPR,
    VS30_ID,
    _write,
)

FRIEND_LEAGUE_ID = "555"


def _seed_two_team_league(raw_dir: Path, league_id: str, name: str) -> None:
    _write(
        raw_dir,
        "league.json",
        {"league_id": league_id, "name": name, "season": str(_SEASON), "settings": {"num_teams": 2},
         "scoring_settings": HALF_PPR, "roster_positions": ["QB", "RB", "RB", "WR", "TE", "BN"]},
    )
    _write(
        raw_dir,
        "teams.json",
        [
            {"roster_id": 1, "owner_id": "112233", "display_name": "rogoel49", "team_name": "Victorious Secret",
             "players": ["s_barkley", "s_cook"], "starters": ["s_barkley"], "settings": {"wins": 3, "losses": 1, "ties": 0}},
            {"roster_id": 2, "owner_id": "444", "display_name": "friend", "team_name": "Friend Team",
             "players": ["s_chase"], "starters": ["s_chase"], "settings": {"wins": 1, "losses": 3, "ties": 0}},
        ],
    )
    _write(
        raw_dir,
        "players.json",
        {
            "s_barkley": {"full_name": "Saquon Barkley", "position": "RB", "team": "PHI"},
            "s_cook": {"full_name": "James Cook", "position": "RB", "team": "BUF"},
            "s_chase": {"full_name": "Ja'Marr Chase", "position": "WR", "team": "CIN"},
        },
    )
    _write(raw_dir, "nfl_state.json", {"season": str(_SEASON), "display_week": _WEEK})
    _write(raw_dir, f"matchups_week_{_WEEK}.json",
           [{"roster_id": 1, "matchup_id": 1, "points": 0}, {"roster_id": 2, "matchup_id": 1, "points": 0}])


class FakeClaude:
    """Scripted model: pops one response per create() and records every
    kwargs it was called with (so multi-turn `messages` threading can be
    asserted on)."""

    def __init__(self):
        self.responses: list = []
        self.calls: list[dict] = []
        outer = self

        class _Messages:
            def create(self_inner, **kwargs):
                outer.calls.append(kwargs)
                return outer.responses.pop(0)

        self.messages = _Messages()


def _resp(*blocks):
    class _R:
        content = list(blocks)

    return _R()


def _tool(name, input_, id_):
    return ToolUseBlock(type="tool_use", id=id_, name=name, input=input_)


@pytest.fixture
def api(tmp_path, monkeypatch):
    # Storage: temp SQLite. Signals: fixture parquet. Player index: fixture rows.
    storage = Storage(tmp_path / "api.sqlite3")
    signals_dir = tmp_path / "signals"
    signals_dir.mkdir()
    pl.DataFrame(_ALL_SIGNALS).write_parquet(signals_dir / f"signals_{_SEASON}_week{_WEEK}.parquet")
    monkeypatch.setattr(main, "SIGNALS_DIR", signals_dir)
    import src.rag.player_index as player_index_module

    monkeypatch.setattr(player_index_module.nflverse, "fetch_players", lambda: pl.DataFrame(_ALL_INDEX_ROWS))

    # League data: per-league dirs under tmp; "ingest" seeds fixture data + a real Chroma index.
    monkeypatch.setattr(leagues, "LEAGUES_DIR", tmp_path / "leagues")
    monkeypatch.delenv("SLEEPER_LEAGUE_ID", raising=False)
    ingested: list[str] = []

    def fake_ingest(league_id, week=None):
        raw_dir, persist_dir = leagues.league_dirs(league_id)
        name = {VS30_ID: "Victorious Secret 3.0", FRIEND_LEAGUE_ID: "Friends PPR"}[league_id]
        _seed_two_team_league(raw_dir, league_id, name)
        embed.embed(embed.build_chunks(raw_dir) + embed.build_signal_chunks(_ALL_SIGNALS), persist_dir=persist_dir)
        ingested.append(league_id)
        return raw_dir, persist_dir

    monkeypatch.setattr(leagues, "ingest_league", fake_ingest)

    # Sleeper login: documented shape, no network.
    def fake_login(username, season=None):
        if username.lower() != "rogoel49":
            raise auth.UnknownSleeperUser(f"No Sleeper user named {username!r}.")
        return {
            "user": {"user_id": "112233", "username": "rogoel49", "display_name": "Rohan"},
            "leagues": [
                {"league_id": VS30_ID, "name": "Victorious Secret 3.0", "season": str(_SEASON), "roster_id": 1},
                {"league_id": FRIEND_LEAGUE_ID, "name": "Friends PPR", "season": str(_SEASON), "roster_id": 2},
            ],
        }

    monkeypatch.setattr(auth, "resolve_user_leagues", fake_login)

    claude = FakeClaude()
    main.app.dependency_overrides[main.get_storage] = lambda: storage
    main.app.dependency_overrides[main.anthropic_client] = lambda: claude
    main.app.dependency_overrides[main.daily_query_cap] = lambda: 3
    client = TestClient(main.app)
    yield {"client": client, "claude": claude, "storage": storage, "ingested": ingested, "tmp": tmp_path}
    main.app.dependency_overrides.clear()


def _login_and_session(api, league_id=VS30_ID, roster_id=None) -> str:
    client = api["client"]
    assert client.post("/api/leagues", json={"username": "rogoel49"}).status_code == 200
    body = {"username": "rogoel49", "league_id": league_id}
    if roster_id is not None:
        body["roster_id"] = roster_id
    resp = client.post("/api/sessions", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()["session_id"]


# ---- login + leagues ----


def test_login_lists_leagues_with_the_users_roster_in_each_and_stores_them(api):
    resp = api["client"].post("/api/leagues", json={"username": "rogoel49"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["user"]["user_id"] == "112233"
    assert {(l["league_id"], l["roster_id"]) for l in body["leagues"]} == {(VS30_ID, "1"), (FRIEND_LEAGUE_ID, "2")}
    assert api["client"].get("/api/leagues/rogoel49").json()["leagues"] == body["leagues"]


def test_login_unknown_sleeper_user_is_404(api):
    assert api["client"].post("/api/leagues", json={"username": "ghost"}).status_code == 404
    assert api["client"].get("/api/leagues/ghost").status_code == 404


def test_sleeper_unreachable_is_a_502_not_a_500(api, monkeypatch):
    """What the sandbox actually produces (Sleeper is blocked here): a
    clear upstream error, not an opaque server error."""
    import requests

    def down(username, season=None):
        raise requests.ConnectionError("Tunnel connection failed: 403 Forbidden")

    monkeypatch.setattr(auth, "resolve_user_leagues", down)
    resp = api["client"].post("/api/leagues", json={"username": "rogoel49"})
    assert resp.status_code == 502
    assert "Sleeper API unreachable" in resp.json()["detail"]


def test_ingest_failure_on_first_use_is_a_502(api, monkeypatch):
    import requests

    api["client"].post("/api/leagues", json={"username": "rogoel49"})

    def down(league_id, week=None):
        raise requests.ConnectionError("Tunnel connection failed: 403 Forbidden")

    monkeypatch.setattr(leagues, "ingest_league", down)
    resp = api["client"].post("/api/sessions", json={"username": "rogoel49", "league_id": VS30_ID})
    assert resp.status_code == 502
    assert "ingesting league" in resp.json()["detail"]


# ---- sessions ----


def test_session_ingests_the_league_on_first_use_only(api):
    session_id = _login_and_session(api)
    assert api["ingested"] == [VS30_ID]
    resp = api["client"].get(f"/api/sessions/{session_id}")
    assert resp.status_code == 200
    assert (resp.json()["league_id"], resp.json()["roster_id"]) == (VS30_ID, "1")

    _login_and_session(api)  # a second session for the same league
    assert api["ingested"] == [VS30_ID]  # not re-ingested


def test_session_returns_the_leagues_real_scoring_settings(api):
    api["client"].post("/api/leagues", json={"username": "rogoel49"})
    resp = api["client"].post("/api/sessions", json={"username": "rogoel49", "league_id": VS30_ID})
    assert resp.json()["league_name"] == "Victorious Secret 3.0"
    assert resp.json()["scoring_settings"] == HALF_PPR


def test_session_for_a_league_the_user_is_not_in_is_404(api):
    api["client"].post("/api/leagues", json={"username": "rogoel49"})
    resp = api["client"].post("/api/sessions", json={"username": "rogoel49", "league_id": "not-mine"})
    assert resp.status_code == 404
    assert api["ingested"] == []  # never ingests a league the user isn't in


def test_unknown_session_is_404_everywhere(api):
    client = api["client"]
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.get("/api/roster", params={"session_id": "nope"}).status_code == 404
    assert client.get("/api/reports/drop", params={"session_id": "nope"}).status_code == 404
    assert client.post("/api/chat", json={"session_id": "nope", "question": "q"}).status_code == 404


# ---- roster ----


def test_roster_is_the_sessions_own_roster_grouped_by_position(api):
    session_id = _login_and_session(api)
    resp = api["client"].get("/api/roster", params={"session_id": session_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["league_id"] == VS30_ID and body["roster_id"] == 1
    assert body["counts_by_position"] == {"RB": 2}
    assert {p["name"] for p in body["players"]} == {"Saquon Barkley", "James Cook"}


def test_roster_comes_from_the_session_not_from_MY_ROSTER_ID(api, monkeypatch):
    """The server never uses the process-wide env var: even with
    MY_ROSTER_ID pointing at the friend's roster, the session's roster
    wins (Phase 5.2's whole point)."""
    monkeypatch.setenv("MY_ROSTER_ID", "2")
    session_id = _login_and_session(api)
    body = api["client"].get("/api/roster", params={"session_id": session_id}).json()
    assert body["roster_id"] == 1
    assert {p["name"] for p in body["players"]} == {"Saquon Barkley", "James Cook"}


def test_switching_league_switches_roster(api):
    session_id = _login_and_session(api, league_id=FRIEND_LEAGUE_ID)
    body = api["client"].get("/api/roster", params={"session_id": session_id}).json()
    assert body["league_id"] == FRIEND_LEAGUE_ID and body["roster_id"] == 2
    assert {p["name"] for p in body["players"]} == {"Ja'Marr Chase"}


# ---- reports ----


def test_report_is_generate_report_output_verbatim_with_stale_markers_per_entry(api):
    session_id = _login_and_session(api)
    resp = api["client"].get("/api/reports/drop", params={"session_id": session_id, "season": _SEASON, "as_of_week": _WEEK})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["report_type"] == "drop"
    assert body["league_id"] == VS30_ID and body["roster_id"] == 1
    assert {e["name"] for e in body["entries"]} == {"Saquon Barkley", "James Cook"}
    for entry in body["entries"]:  # Phase 3.6 markers, as distinct fields, on every entry
        assert entry["stale"] is False and entry["source_season"] is None and entry["source_as_of_week"] is None
    assert "notes" in body


def test_report_stale_fallback_markers_pass_through(api):
    """A season with no current signals falls back to the prior season
    (Phase 3.6); the API must surface stale/source_season per entry, not
    a summary flag."""
    session_id = _login_and_session(api)
    resp = api["client"].get(
        "/api/reports/drop", params={"session_id": session_id, "season": _SEASON + 1, "as_of_week": 1}
    )
    assert resp.status_code == 200, resp.text
    entries = resp.json()["entries"]
    assert entries and all(e["stale"] is True and e["source_season"] == _SEASON for e in entries)
    assert any("stale" in note for note in resp.json()["notes"])


def test_unknown_report_type_is_400(api):
    session_id = _login_and_session(api)
    assert api["client"].get("/api/reports/trade_suggestions", params={"session_id": session_id}).status_code == 400


def test_reports_are_not_counted_against_the_query_cap(api):
    session_id = _login_and_session(api)
    for _ in range(5):  # cap is 3 in this fixture
        assert api["client"].get("/api/reports/waiver_pickups", params={"session_id": session_id}).status_code == 200
    assert api["storage"].queries_today("rogoel49") == 0


# ---- chat ----


def _script_grounded_answer(claude: FakeClaude) -> None:
    claude.responses = [
        _resp(_tool("get_my_roster", {}, "t1"), _tool("get_player_signals", {"player_name": "Saquon Barkley"}, "t2")),
        _resp(
            _tool(
                "submit_recommendation",
                {
                    "recommendation": "Start Saquon Barkley.",
                    "reasoning": "48% red zone share.",
                    "player_id": "00-0034844",
                    "data_gaps": [
                        {"reason": "out_of_scope_capability", "detail": "No tool values trades."},
                    ],
                },
                "t3",
            )
        ),
    ]


def test_chat_exposes_data_gaps_and_per_player_stale_markers_as_distinct_fields(api):
    session_id = _login_and_session(api)
    _script_grounded_answer(api["claude"])

    resp = api["client"].post(
        "/api/chat", json={"session_id": session_id, "question": "Start Barkley?", "season": _SEASON, "as_of_week": _WEEK}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["recommendation"] == "Start Saquon Barkley."
    assert body["player_id"] == "00-0034844"
    # Phase 3.7, verbatim from submit_recommendation:
    assert body["data_gaps"] == [{"reason": "out_of_scope_capability", "detail": "No tool values trades."}]
    # Phase 3.6, per player the agent actually looked up, fields unmodified:
    assert body["signals_consulted"] == [
        {
            "player_id": "00-0034844", "player_name": "Saquon Barkley", "position": "RB", "team": "PHI",
            "has_signals": True, "stale": False, "source_season": None, "source_as_of_week": None,
        }
    ]
    assert body["league_id"] == VS30_ID and body["roster_id"] == "1"
    assert body["error"] is None
    assert (body["queries_used_today"], body["daily_query_cap"]) == (1, 3)
    json.dumps(body["messages"])  # fully JSON-serializable, SDK content blocks converted


def test_chat_stale_signal_markers_pass_through(api):
    session_id = _login_and_session(api)
    _script_grounded_answer(api["claude"])
    body = api["client"].post(
        "/api/chat", json={"session_id": session_id, "question": "q", "season": _SEASON + 1, "as_of_week": 1}
    ).json()
    consulted = body["signals_consulted"][0]
    assert consulted["has_signals"] is True and consulted["stale"] is True
    assert consulted["source_season"] == _SEASON and consulted["source_as_of_week"] == _WEEK


def test_chat_no_signal_data_player_is_visible_in_signals_consulted(api):
    session_id = _login_and_session(api)
    api["claude"].responses = [
        _resp(_tool("get_player_signals", {"player_name": "James Cook"}, "t1")),
        _resp(_tool("submit_recommendation", {"recommendation": "x", "reasoning": "y", "data_gaps": []}, "t2")),
    ]
    # Cook is in the player index but give him no Chroma chunk: has_signals false.
    raw_dir, persist_dir = leagues.league_dirs(VS30_ID)
    embed.embed(embed.build_chunks(raw_dir) + embed.build_signal_chunks([_BARKLEY_SIGNAL, _CHASE_SIGNAL]), persist_dir=persist_dir)

    body = api["client"].post(
        "/api/chat", json={"session_id": session_id, "question": "q", "season": _SEASON, "as_of_week": _WEEK}
    ).json()
    assert body["signals_consulted"] == [
        {"player_id": _COOK_SIGNAL["player_id"], "player_name": "James Cook", "position": "RB", "team": "BUF",
         "has_signals": False, "stale": None, "source_season": None, "source_as_of_week": None}
    ]


def test_chat_uses_the_sessions_roster_for_my_questions(api, monkeypatch):
    monkeypatch.setenv("MY_ROSTER_ID", "2")  # must be ignored
    session_id = _login_and_session(api)
    _script_grounded_answer(api["claude"])
    api["client"].post("/api/chat", json={"session_id": session_id, "question": "q", "season": _SEASON, "as_of_week": _WEEK})
    # The get_my_roster tool result the model saw (recommend() keeps appending
    # to the same messages list after each create(), so search every user
    # turn rather than assuming which one is last):
    tool_results = [
        block
        for message in api["claude"].calls[1]["messages"]
        if message["role"] == "user" and isinstance(message["content"], list)
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    roster_result = json.loads(next(r["content"] for r in tool_results if r["tool_use_id"] == "t1"))
    assert {p["name"] for p in roster_result["players"]} == {"Saquon Barkley", "James Cook"}


def test_chat_multi_turn_threads_messages_back_in(api):
    session_id = _login_and_session(api)
    claude = api["claude"]
    claude.responses = [_resp(TextBlock(type="text", text="Which Barkley do you mean?"))]
    first = api["client"].post(
        "/api/chat", json={"session_id": session_id, "question": "Start Barkley?", "season": _SEASON, "as_of_week": _WEEK}
    ).json()
    assert first["recommendation"] == "Which Barkley do you mean?"
    assert first["messages"][-1] == {"role": "assistant", "content": [{"type": "text", "text": "Which Barkley do you mean?"}]}

    _script_grounded_answer(claude)
    second = api["client"].post(
        "/api/chat",
        json={"session_id": session_id, "question": "Saquon", "messages": first["messages"], "season": _SEASON, "as_of_week": _WEEK},
    ).json()
    assert second["recommendation"] == "Start Saquon Barkley."
    # The model saw the whole conversation: turn 1 question, its own clarifying answer, then "Saquon".
    seen = claude.calls[1]["messages"]
    assert [m["role"] for m in seen[:3]] == ["user", "assistant", "user"]
    assert seen[0]["content"] == "Start Barkley?" and seen[2]["content"] == "Saquon"


def test_chat_daily_cap_returns_429_and_counts_only_successful_calls(api):
    session_id = _login_and_session(api)
    for _ in range(3):
        _script_grounded_answer(api["claude"])
        assert api["client"].post(
            "/api/chat", json={"session_id": session_id, "question": "q", "season": _SEASON, "as_of_week": _WEEK}
        ).status_code == 200
    resp = api["client"].post("/api/chat", json={"session_id": session_id, "question": "q"})
    assert resp.status_code == 429
    assert "3 of 3" in resp.json()["detail"]
    assert api["claude"].responses == []  # the model was never called for the refused request


def test_chat_cap_is_per_user(api):
    """Another user's queries don't consume rogoel49's cap (cap counts key
    on the session's username, not on the process)."""
    session_id = _login_and_session(api)
    api["storage"].record_query("someone_else", cap=3)
    api["storage"].record_query("someone_else", cap=3)
    api["storage"].record_query("someone_else", cap=3)
    _script_grounded_answer(api["claude"])
    assert api["client"].post(
        "/api/chat", json={"session_id": session_id, "question": "q", "season": _SEASON, "as_of_week": _WEEK}
    ).status_code == 200
