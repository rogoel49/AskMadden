import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from src.rag import embed
from src.reasoning import recommend


def _write(raw_dir: Path, filename: str, data) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / filename).write_text(json.dumps({"fetched_at": "2026-01-01T00:00:00Z", "source": "sleeper", "data": data}))


def _seed_league(raw_dir: Path) -> None:
    _write(
        raw_dir,
        "league.json",
        {
            "name": "Victorious Secret 3.0",
            "season": "2024",
            "settings": {"num_teams": 12},
            "scoring_settings": {"rec": 0.5, "pass_td": 4, "rush_td": 6, "rec_td": 6},
        },
    )
    _write(
        raw_dir,
        "teams.json",
        [
            {
                "roster_id": 1,
                "owner_id": "u1",
                "display_name": "rogoel49",
                "team_name": "Victorious Secret",
                "players": ["sleeper_cmc"],
                "starters": ["sleeper_cmc"],
                "settings": {},
            }
        ],
    )
    _write(
        raw_dir,
        "players.json",
        {"sleeper_cmc": {"full_name": "Christian McCaffrey", "position": "RB", "team": "SF"}},
    )
    _write(raw_dir, "nfl_state.json", {"season": "2024", "display_week": 8})


# Real nflverse identities from the reported bug (see tests/test_player_index.py).
_CHRISTIAN_ROW = {
    "gsis_id": "00-0033280", "display_name": "Christian McCaffrey", "position": "RB",
    "latest_team": "SF", "last_season": 2026,
}
_LUKE_ROW = {
    "gsis_id": "00-0039355", "display_name": "Luke McCaffrey", "position": "WR",
    "latest_team": "WAS", "last_season": 2026,
}
_MIKE_WILLIAMS_RECENT = {
    "gsis_id": "00-0033536", "display_name": "Mike Williams", "position": "WR",
    "latest_team": "LAC", "last_season": 2025,
}
_MIKE_WILLIAMS_STALE = {
    "gsis_id": "00-0027702", "display_name": "Mike Williams", "position": "WR",
    "latest_team": "BUF", "last_season": 2014,
}

_CHRISTIAN_SIGNAL_ROW = {
    "player_id": "00-0033280", "player_name": "C.McCaffrey", "team": "SF", "season": 2024, "as_of_week": 8,
    "epa_trend": 0.2, "red_zone_share": 0.4, "target_share": 0.15, "opponent": "SEA",
    "run_funnel_rate_vs_avg": 0.05, "implied_total": 24.0, "adot": None, "ryoe_per_att": 1.1, "cpoe": None,
}
_LUKE_SIGNAL_ROW = {
    "player_id": "00-0039355", "player_name": "L.McCaffrey", "team": "WAS", "season": 2024, "as_of_week": 8,
    "epa_trend": -0.1, "red_zone_share": 0.1, "target_share": 0.08, "opponent": "NYG",
    "run_funnel_rate_vs_avg": -0.05, "implied_total": 18.0, "adot": None, "ryoe_per_att": None, "cpoe": None,
}


def _make_ctx(tmp_path, players_df) -> recommend.RecommendContext:
    raw_dir = tmp_path / "sleeper"
    persist_dir = tmp_path / "chroma"
    _seed_league(raw_dir)
    chunks = embed.build_signal_chunks([_CHRISTIAN_SIGNAL_ROW, _LUKE_SIGNAL_ROW])
    embed.embed(chunks, persist_dir=persist_dir)

    from src.rag import player_index

    idx = player_index.build_player_index(2024, players=players_df)
    return recommend.RecommendContext(raw_dir=raw_dir, persist_dir=persist_dir, season=2024, as_of_week=8, player_idx=idx)


# ---- individual tool dispatch tests (no network, no Claude API) ----


def test_get_player_signals_resolves_christian_not_luke(tmp_path):
    players = pl.DataFrame([_CHRISTIAN_ROW, _LUKE_ROW])
    ctx = _make_ctx(tmp_path, players)

    result = recommend.dispatch_tool("get_player_signals", {"player_name": "Christian McCaffrey"}, ctx)

    assert result["resolved"] is True
    assert result["player_id"] == "00-0033280"
    assert "L.McCaffrey" not in result["signals"]


def test_get_player_signals_ambiguous_name_reports_all_candidates_not_a_guess(tmp_path):
    players = pl.DataFrame([_CHRISTIAN_ROW, _LUKE_ROW])
    ctx = _make_ctx(tmp_path, players)

    result = recommend.dispatch_tool("get_player_signals", {"player_name": "McCaffrey"}, ctx)

    assert result["resolved"] is False
    assert result["ambiguous"] is True
    ids = {c["player_id"] for c in result["candidates"]}
    assert ids == {"00-0033280", "00-0039355"}
    assert "signals" not in result


def test_get_player_signals_mike_williams_resolves_to_the_current_one(tmp_path):
    players = pl.DataFrame([_MIKE_WILLIAMS_RECENT, _MIKE_WILLIAMS_STALE])
    ctx = _make_ctx(tmp_path, players)

    result = recommend.dispatch_tool("get_player_signals", {"player_name": "Mike Williams"}, ctx)

    assert result["resolved"] is True
    assert result["player_id"] == "00-0033536"


def test_get_player_signals_unknown_player_reports_not_resolved(tmp_path):
    players = pl.DataFrame([_CHRISTIAN_ROW])
    ctx = _make_ctx(tmp_path, players)

    result = recommend.dispatch_tool("get_player_signals", {"player_name": "Nobody Realname"}, ctx)

    assert result["resolved"] is False
    assert "ambiguous" not in result


def test_get_my_roster_uses_structured_sleeper_lookup(tmp_path, monkeypatch):
    players = pl.DataFrame([_CHRISTIAN_ROW])
    ctx = _make_ctx(tmp_path, players)
    monkeypatch.setenv("MY_ROSTER_ID", "1")

    result = recommend.dispatch_tool("get_my_roster", {}, ctx)

    assert [p["name"] for p in result["players"]] == ["Christian McCaffrey"]


def test_find_owner_returns_team_that_rosters_the_player(tmp_path):
    players = pl.DataFrame([_CHRISTIAN_ROW])
    ctx = _make_ctx(tmp_path, players)

    result = recommend.dispatch_tool("find_owner", {"player_name": "Christian McCaffrey"}, ctx)

    assert result["owner_team_name"] == "Victorious Secret"


def test_search_league_info_falls_back_to_semantic_query(tmp_path):
    players = pl.DataFrame([_CHRISTIAN_ROW])
    ctx = _make_ctx(tmp_path, players)

    result = recommend.dispatch_tool("search_league_info", {"query": "league settings"}, ctx)

    assert isinstance(result["results"], list)


def _make_ctx_with_record_and_matchup(tmp_path, players_df) -> recommend.RecommendContext:
    """Like _make_ctx, but with a second team and a real matchup so
    get_team_record / get_current_matchup have something to find."""
    ctx = _make_ctx(tmp_path, players_df)
    _write(
        ctx.raw_dir,
        "teams.json",
        [
            {
                "roster_id": 1,
                "owner_id": "u1",
                "display_name": "rogoel49",
                "team_name": "Victorious Secret",
                "players": ["sleeper_cmc"],
                "starters": ["sleeper_cmc"],
                "settings": {"wins": 5, "losses": 2, "ties": 0},
            },
            {
                "roster_id": 2,
                "owner_id": "u2",
                "display_name": "rival",
                "team_name": "Rival Team",
                "players": [],
                "starters": [],
                "settings": {"wins": 3, "losses": 4, "ties": 0},
            },
        ],
    )
    _write(
        ctx.raw_dir,
        "matchups_week_8.json",
        [
            {"roster_id": 1, "matchup_id": 1, "points": 0},
            {"roster_id": 2, "matchup_id": 1, "points": 0},
        ],
    )
    return ctx


def test_get_team_record_reports_my_record(tmp_path, monkeypatch):
    players = pl.DataFrame([_CHRISTIAN_ROW])
    ctx = _make_ctx_with_record_and_matchup(tmp_path, players)
    monkeypatch.setenv("MY_ROSTER_ID", "1")

    result = recommend.dispatch_tool("get_team_record", {}, ctx)

    assert result == {"roster_id": 1, "team_name": "Victorious Secret", "wins": 5, "losses": 2, "ties": 0}


def test_get_team_record_for_another_owner(tmp_path, monkeypatch):
    players = pl.DataFrame([_CHRISTIAN_ROW])
    ctx = _make_ctx_with_record_and_matchup(tmp_path, players)
    monkeypatch.setenv("MY_ROSTER_ID", "1")

    result = recommend.dispatch_tool("get_team_record", {"owner_display_name": "rival"}, ctx)

    assert result["wins"] == 3


def test_get_current_matchup_reports_this_weeks_opponent(tmp_path, monkeypatch):
    players = pl.DataFrame([_CHRISTIAN_ROW])
    ctx = _make_ctx_with_record_and_matchup(tmp_path, players)
    monkeypatch.setenv("MY_ROSTER_ID", "1")

    result = recommend.dispatch_tool("get_current_matchup", {}, ctx)

    assert result["opponent_roster_id"] == 2
    assert result["opponent_team_name"] == "Rival Team"


def test_get_current_matchup_reports_a_note_when_week_not_ingested(tmp_path, monkeypatch):
    players = pl.DataFrame([_CHRISTIAN_ROW])
    ctx = _make_ctx(tmp_path, players)  # no matchups_week_8.json seeded
    monkeypatch.setenv("MY_ROSTER_ID", "1")

    result = recommend.dispatch_tool("get_current_matchup", {}, ctx)

    assert "note" in result
    assert "has been ingested" in result["note"]  # "No matchup data ... has been ingested" (not a guess)


# ---- orchestration loop, driven by a fake Anthropic client ----


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name, input_, id_="tool_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_)


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        # Snapshot the messages list at call time -- recommend() keeps
        # appending to the same list object after this call returns, so
        # storing the bare reference would make later assertions see the
        # *final* state of the conversation instead of what was actually
        # sent for this particular call.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def test_recommend_calls_get_player_signals_then_submits(tmp_path, monkeypatch):
    raw_dir = tmp_path / "sleeper"
    persist_dir = tmp_path / "chroma"
    _seed_league(raw_dir)
    chunks = embed.build_signal_chunks([_CHRISTIAN_SIGNAL_ROW, _LUKE_SIGNAL_ROW])
    embed.embed(chunks, persist_dir=persist_dir)

    responses = [
        SimpleNamespace(content=[_tool_use_block("get_player_signals", {"player_name": "Christian McCaffrey"})]),
        SimpleNamespace(
            content=[
                _tool_use_block(
                    "submit_recommendation",
                    {
                        "recommendation": "Start Christian McCaffrey.",
                        "reasoning": "Positive efficiency trend and a strong red zone share.",
                        "player_id": "00-0033280",
                    },
                    id_="tool_2",
                )
            ]
        ),
    ]
    client = _FakeClient(responses)

    # avoid a real network call to nflreadpy inside build_player_index()
    import src.rag.player_index as player_index_module

    monkeypatch.setattr(player_index_module.nflverse, "fetch_players", lambda: pl.DataFrame([_CHRISTIAN_ROW, _LUKE_ROW]))

    result = recommend.recommend(
        "Should I start Christian McCaffrey?",
        raw_dir=raw_dir,
        persist_dir=persist_dir,
        season=2024,
        as_of_week=8,
        client=client,
    )

    assert result["recommendation"] == "Start Christian McCaffrey."
    assert result["player_id"] == "00-0033280"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "get_player_signals"
    assert result["tool_calls"][0]["result"]["player_id"] == "00-0033280"


def test_recommend_falls_back_to_plain_text_if_model_never_calls_a_tool(tmp_path, monkeypatch):
    raw_dir = tmp_path / "sleeper"
    persist_dir = tmp_path / "chroma"
    _seed_league(raw_dir)
    embed.embed([], persist_dir=persist_dir)

    client = _FakeClient([SimpleNamespace(content=[_text_block("I don't have enough information.")])])

    import src.rag.player_index as player_index_module

    monkeypatch.setattr(player_index_module.nflverse, "fetch_players", lambda: pl.DataFrame([_CHRISTIAN_ROW]))

    result = recommend.recommend(
        "What's up?", raw_dir=raw_dir, persist_dir=persist_dir, season=2024, as_of_week=8, client=client
    )

    assert result["recommendation"] == "I don't have enough information."
    assert result["reasoning"] is None


def test_recommend_returns_gracefully_if_max_turns_exceeded_without_submitting(tmp_path, monkeypatch):
    """Regression test: this used to raise an unhandled RuntimeError,
    which crashed the CLI ungracefully. recommend() must never raise on
    a question the tools couldn't resolve -- it should return a clear
    "not enough information" result instead, with error set so a caller
    can detect this specific failure mode."""
    raw_dir = tmp_path / "sleeper"
    persist_dir = tmp_path / "chroma"
    _seed_league(raw_dir)
    embed.embed([], persist_dir=persist_dir)

    # Always calls a non-terminal tool, never submit_recommendation.
    responses = [
        SimpleNamespace(content=[_tool_use_block("search_league_info", {"query": "settings"})]) for _ in range(3)
    ]
    client = _FakeClient(responses)

    import src.rag.player_index as player_index_module

    monkeypatch.setattr(player_index_module.nflverse, "fetch_players", lambda: pl.DataFrame([_CHRISTIAN_ROW]))

    result = recommend.recommend(
        "loop forever",
        raw_dir=raw_dir,
        persist_dir=persist_dir,
        season=2024,
        as_of_week=8,
        client=client,
        max_turns=3,
    )

    assert result["error"] == "max_turns_exceeded"
    assert "don't have enough information" in result["recommendation"]
    assert result["player_id"] is None
    assert len(result["tool_calls"]) == 3
    assert isinstance(result["messages"], list) and result["messages"]


def test_recommend_requires_sleeper_ingest_to_have_run(tmp_path):
    with pytest.raises(RuntimeError, match="run `python -m src.ingest.sleeper`"):
        recommend.recommend("anything", raw_dir=tmp_path / "nonexistent", client=_FakeClient([]))


def _seed_league_with_record_and_matchup(raw_dir: Path) -> None:
    """Same shape as _make_ctx_with_record_and_matchup, but for a full
    recommend() call (raw_dir directly, not a RecommendContext)."""
    _seed_league(raw_dir)
    _write(
        raw_dir,
        "teams.json",
        [
            {
                "roster_id": 1, "owner_id": "u1", "display_name": "rogoel49", "team_name": "Victorious Secret",
                "players": ["sleeper_cmc"], "starters": ["sleeper_cmc"], "settings": {"wins": 5, "losses": 2, "ties": 0},
            },
            {
                "roster_id": 2, "owner_id": "u2", "display_name": "rival", "team_name": "Rival Team",
                "players": [], "starters": [], "settings": {"wins": 3, "losses": 4, "ties": 0},
            },
        ],
    )
    _write(
        raw_dir,
        "matchups_week_8.json",
        [{"roster_id": 1, "matchup_id": 1, "points": 0}, {"roster_id": 2, "matchup_id": 1, "points": 0}],
    )


# ---- regression test for the reported crash: "what's my team's record
# and who do i play this week?" used to burn through max_turns (nothing
# could answer it) and raise an unhandled RuntimeError. ----

_RECORD_AND_MATCHUP_QUESTION = "what's my team's record and who do i play this week?"


def test_recommend_answers_record_and_matchup_question_with_the_new_tools(tmp_path, monkeypatch):
    raw_dir = tmp_path / "sleeper"
    persist_dir = tmp_path / "chroma"
    _seed_league_with_record_and_matchup(raw_dir)
    embed.embed([], persist_dir=persist_dir)
    monkeypatch.setenv("MY_ROSTER_ID", "1")

    import src.rag.player_index as player_index_module

    monkeypatch.setattr(player_index_module.nflverse, "fetch_players", lambda: pl.DataFrame([_CHRISTIAN_ROW]))

    responses = [
        SimpleNamespace(content=[_tool_use_block("get_team_record", {}, id_="t1")]),
        SimpleNamespace(content=[_tool_use_block("get_current_matchup", {}, id_="t2")]),
        SimpleNamespace(
            content=[
                _tool_use_block(
                    "submit_recommendation",
                    {
                        "recommendation": "You're 5-2 and play Rival Team this week.",
                        "reasoning": "get_team_record reported 5 wins/2 losses; get_current_matchup reported Rival Team as this week's opponent.",
                    },
                    id_="t3",
                )
            ]
        ),
    ]
    client = _FakeClient(responses)

    result = recommend.recommend(
        _RECORD_AND_MATCHUP_QUESTION, raw_dir=raw_dir, persist_dir=persist_dir, season=2024, as_of_week=8, client=client
    )

    assert result["error"] is None
    assert "5-2" in result["recommendation"]
    tool_names = [c["name"] for c in result["tool_calls"]]
    assert tool_names == ["get_team_record", "get_current_matchup"]
    assert result["tool_calls"][0]["result"]["wins"] == 5
    assert result["tool_calls"][1]["result"]["opponent_team_name"] == "Rival Team"


def test_recommend_never_crashes_on_the_record_and_matchup_question_even_if_model_stalls(tmp_path, monkeypatch):
    """Safety net: even if a model somehow never converges on this
    question, recommend() must return gracefully, never raise or crash
    the process (the originally reported behavior)."""
    raw_dir = tmp_path / "sleeper"
    persist_dir = tmp_path / "chroma"
    _seed_league_with_record_and_matchup(raw_dir)
    embed.embed([], persist_dir=persist_dir)
    monkeypatch.setenv("MY_ROSTER_ID", "1")

    import src.rag.player_index as player_index_module

    monkeypatch.setattr(player_index_module.nflverse, "fetch_players", lambda: pl.DataFrame([_CHRISTIAN_ROW]))

    # A model that keeps re-searching instead of using the new tools or submitting.
    responses = [
        SimpleNamespace(content=[_tool_use_block("search_league_info", {"query": "record"})]) for _ in range(3)
    ]
    client = _FakeClient(responses)

    result = recommend.recommend(
        _RECORD_AND_MATCHUP_QUESTION,
        raw_dir=raw_dir,
        persist_dir=persist_dir,
        season=2024,
        as_of_week=8,
        client=client,
        max_turns=3,
    )

    assert result["error"] == "max_turns_exceeded"
    assert result["recommendation"]  # a real string, not an exception


# ---- multi-turn conversation ----


def test_recommend_returns_messages_that_can_be_continued(tmp_path, monkeypatch):
    raw_dir = tmp_path / "sleeper"
    persist_dir = tmp_path / "chroma"
    _seed_league(raw_dir)
    chunks = embed.build_signal_chunks([_CHRISTIAN_SIGNAL_ROW, _LUKE_SIGNAL_ROW])
    embed.embed(chunks, persist_dir=persist_dir)

    import src.rag.player_index as player_index_module

    monkeypatch.setattr(player_index_module.nflverse, "fetch_players", lambda: pl.DataFrame([_CHRISTIAN_ROW, _LUKE_ROW]))

    responses = [
        SimpleNamespace(content=[_tool_use_block("get_player_signals", {"player_name": "Christian McCaffrey"})]),
        SimpleNamespace(
            content=[
                _tool_use_block(
                    "submit_recommendation",
                    {"recommendation": "Start him.", "reasoning": "Good signals.", "player_id": "00-0033280"},
                    id_="t2",
                )
            ]
        ),
    ]
    client = _FakeClient(responses)

    result = recommend.recommend(
        "Should I start Christian McCaffrey?",
        raw_dir=raw_dir,
        persist_dir=persist_dir,
        season=2024,
        as_of_week=8,
        client=client,
    )

    # The last message must be a resolved tool_result turn (not a dangling
    # tool_use), so a caller can immediately continue the conversation
    # with a plain follow-up question.
    last = result["messages"][-1]
    assert last["role"] == "user"
    assert all(block["type"] == "tool_result" for block in last["content"])


def test_recommend_second_call_sends_the_full_prior_history_to_the_client(tmp_path, monkeypatch):
    """Mechanical proof that context threads across turns: the second
    recommend() call's messages= must be exactly the first call's
    returned history plus the new question -- never a fresh, isolated
    conversation."""
    raw_dir = tmp_path / "sleeper"
    persist_dir = tmp_path / "chroma"
    _seed_league(raw_dir)
    embed.embed([], persist_dir=persist_dir)

    import src.rag.player_index as player_index_module

    monkeypatch.setattr(player_index_module.nflverse, "fetch_players", lambda: pl.DataFrame([_CHRISTIAN_ROW]))

    # Turn 1: the agent asks a clarifying question instead of calling a tool.
    turn1_responses = [SimpleNamespace(content=[_text_block("Which of your flex-eligible players do you mean?")])]
    client = _FakeClient(turn1_responses)
    turn1 = recommend.recommend(
        "Who should I start at flex?", raw_dir=raw_dir, persist_dir=persist_dir, season=2024, as_of_week=8, client=client
    )

    assert turn1["recommendation"] == "Which of your flex-eligible players do you mean?"

    # Turn 2: continue the same conversation with the clarification.
    turn2_responses = [
        SimpleNamespace(
            content=[
                _tool_use_block(
                    "submit_recommendation",
                    {"recommendation": "Start Christian McCaffrey then.", "reasoning": "You clarified it's him."},
                )
            ]
        )
    ]
    client2 = _FakeClient(turn2_responses)
    turn2 = recommend.recommend(
        "I meant Christian McCaffrey",
        messages=turn1["messages"],
        raw_dir=raw_dir,
        persist_dir=persist_dir,
        season=2024,
        as_of_week=8,
        client=client2,
    )

    sent_messages = client2.messages.calls[0]["messages"]
    # turn1's user question + turn1's assistant clarification + turn2's new user question
    assert len(sent_messages) == 3
    assert sent_messages[0] == {"role": "user", "content": "Who should I start at flex?"}
    assert sent_messages[1]["role"] == "assistant"
    assert sent_messages[2] == {"role": "user", "content": "I meant Christian McCaffrey"}
    assert turn2["recommendation"] == "Start Christian McCaffrey then."


def test_eval_style_calls_never_carry_state_between_independent_questions(tmp_path, monkeypatch):
    """The eval harness (run_eval.py, run_decision_eval.py) never passes
    messages= -- confirm that omitting it (the default) means each call
    is a genuinely fresh, independent conversation, never leaking a
    prior question's context."""
    raw_dir = tmp_path / "sleeper"
    persist_dir = tmp_path / "chroma"
    _seed_league(raw_dir)
    embed.embed([], persist_dir=persist_dir)

    import src.rag.player_index as player_index_module

    monkeypatch.setattr(player_index_module.nflverse, "fetch_players", lambda: pl.DataFrame([_CHRISTIAN_ROW]))

    def _one_shot_client():
        return _FakeClient(
            [SimpleNamespace(content=[_tool_use_block("submit_recommendation", {"recommendation": "ok", "reasoning": "ok"})])]
        )

    client_a = _one_shot_client()
    recommend.recommend(
        "first independent question", raw_dir=raw_dir, persist_dir=persist_dir, season=2024, as_of_week=8, client=client_a
    )
    client_b = _one_shot_client()
    recommend.recommend(
        "second independent question", raw_dir=raw_dir, persist_dir=persist_dir, season=2024, as_of_week=8, client=client_b
    )

    # Each call's client only ever saw its own single question -- no
    # trace of the other question's text anywhere in what was sent.
    sent_to_b = client_b.messages.calls[0]["messages"]
    assert len(sent_to_b) == 1
    assert sent_to_b[0]["content"] == "second independent question"
