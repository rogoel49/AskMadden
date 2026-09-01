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
        self.calls.append(kwargs)
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


def test_recommend_raises_if_max_turns_exceeded_without_submitting(tmp_path, monkeypatch):
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

    with pytest.raises(RuntimeError, match="exceeded max_turns"):
        recommend.recommend(
            "loop forever",
            raw_dir=raw_dir,
            persist_dir=persist_dir,
            season=2024,
            as_of_week=8,
            client=client,
            max_turns=3,
        )


def test_recommend_requires_sleeper_ingest_to_have_run(tmp_path):
    with pytest.raises(RuntimeError, match="run `python -m src.ingest.sleeper`"):
        recommend.recommend("anything", raw_dir=tmp_path / "nonexistent", client=_FakeClient([]))
