"""Chat vs. Feed verdict alignment: recommend()'s rank_players tool and
generate_report()'s start_sit report must reach the same start/sit
verdict on the same signals, because they now call the same code
(src/reasoning/ranking.py's rank_candidates). See ranking.py's docstring
for the real usage-testing case (Herbert vs. Mahomes) that motivated it.

Fixtures reuse tests/test_report.py's league seeding and real 2024
week-5 signal values so every "clear-cut" case here is the same
real-data case the report is already validated against. The Herbert /
Mahomes rows are the real 2025 season-end values from the committed
data/processed/signals/signals_2025_week19.parquet.

What these tests can and can't prove, honestly: a fake Anthropic client
proves (1) the tool returns the same verdict the report does, (2) the
orchestration loop bounces a submitted verdict that contradicts the
ranking and passes a consistent one through, and (3) the system prompt
carries the rule. None of that proves a real model *follows* the rule
when it sets no player_id and contradicts the ranking in prose only --
that needs real-model re-validation on Rohan's machine, same convention
as Phase 3.7/3.8 (flagged in TODO.md).
"""
from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from src.reasoning import ranking, recommend, report
from tests.test_report import (
    _BARKLEY_ROW,
    _BARKLEY_SIGNAL_ROW,
    _CHRISTIAN_ROW,
    _COOK_ROW,
    _COOK_SIGNAL_ROW,
    _LEAGUE_ID,
    _LUKE_ROW,
    _SEASON,
    _WEEK,
    _setup,
)

# Real nflverse identities for the two QBs from the real usage-testing case.
_HERBERT_ROW = {
    "gsis_id": "00-0036355", "display_name": "Justin Herbert", "position": "QB",
    "latest_team": "LAC", "last_season": 2026,
}
_MAHOMES_ROW = {
    "gsis_id": "00-0033873", "display_name": "Patrick Mahomes", "position": "QB",
    "latest_team": "KC", "last_season": 2026,
}
# Real 2025 season-end signal rows (data/processed/signals/signals_2025_week19.parquet).
_HERBERT_2025_SIGNAL_ROW = {
    "player_id": "00-0036355", "player_name": "J.Herbert", "team": "LAC", "season": 2025, "as_of_week": 19,
    "season_plays": 591, "epa_trend": 0.18788404515740742, "red_zone_share": 0.027586206896551724,
    "target_share": None, "target_share_adjusted": None, "opponent": "NE",
    "run_funnel_rate_vs_avg": -0.008378604710922277, "implied_total": 21.0,
}
_MAHOMES_2025_SIGNAL_ROW = {
    "player_id": "00-0033873", "player_name": "P.Mahomes", "team": "KC", "season": 2025, "as_of_week": 19,
    "season_plays": 541, "epa_trend": None, "red_zone_share": 0.019230769230769232,
    "target_share": 0.0018115942028985507, "target_share_adjusted": None, "opponent": None,
    "run_funnel_rate_vs_avg": None, "implied_total": None,
}

_RB_ROSTER = {
    "sleeper_barkley": {"full_name": "Saquon Barkley", "position": "RB", "team": "PHI"},
    "sleeper_cook": {"full_name": "James Cook", "position": "RB", "team": "BUF"},
}
_QB_ROSTER = {
    "sleeper_herbert": {"full_name": "Justin Herbert", "position": "QB", "team": "LAC"},
    "sleeper_mahomes": {"full_name": "Patrick Mahomes", "position": "QB", "team": "KC"},
}


def _ctx(raw_dir, persist_dir, signals_dir, season=_SEASON, as_of_week=_WEEK):
    from src.rag import player_index

    return recommend.RecommendContext(
        raw_dir=raw_dir,
        persist_dir=persist_dir,
        season=season,
        as_of_week=as_of_week,
        player_idx=player_index.build_player_index(season),
        signals_dir=signals_dir,
    )


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name, input_, id_="tool_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_)


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def _submit(recommendation, reasoning, player_id=None, data_gaps=None, id_="tool_submit"):
    payload = {"recommendation": recommendation, "reasoning": reasoning}
    if player_id is not None:
        payload["player_id"] = player_id
    if data_gaps is not None:
        payload["data_gaps"] = data_gaps
    return SimpleNamespace(content=[_tool_use_block("submit_recommendation", payload, id_=id_)])


def _start_sit_starter(raw_dir, persist_dir, signals_dir, season=_SEASON, as_of_week=_WEEK):
    result = report.generate_report(
        "start_sit", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=season, as_of_week=as_of_week,
        signals_dir=signals_dir,
    )
    assert len(result["entries"]) == 1
    return result["entries"][0]


# ---- (a) clear-cut: the tool's verdict is the Feed's verdict, exactly ----


def test_rank_players_matches_the_start_sit_report_on_a_clear_cut_case(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, _RB_ROSTER, [_BARKLEY_SIGNAL_ROW, _COOK_SIGNAL_ROW], pl.DataFrame([_BARKLEY_ROW, _COOK_ROW])
    )
    ctx = _ctx(raw_dir, persist_dir, signals_dir)

    tool = recommend.dispatch_tool("rank_players", {"player_names": ["James Cook", "Saquon Barkley"]}, ctx)
    entry = _start_sit_starter(raw_dir, persist_dir, signals_dir)

    assert tool["verdict"] == "clear"
    assert tool["recommended"]["player_id"] == entry["recommended_starter"]["player_id"] == "00-0034844"
    assert tool["recommended"]["name"] == entry["recommended_starter"]["name"] == "Saquon Barkley"
    # Same ordering, same cited numbers, same (non-)stale markers as the report.
    assert [e["player_id"] for e in tool["ranked"]] == [
        entry["recommended_starter"]["player_id"],
        *(alt["player_id"] for alt in entry["alternatives_considered"]),
    ]
    assert tool["ranked"][1]["signals_summary"] == entry["alternatives_considered"][0]["signals_summary"]
    assert tool["recommended"]["stale"] is False and entry["recommended_starter"]["stale"] is False
    assert tool["same_position"] is True
    assert tool["unranked"] == []
    assert "48%" in tool["recommended"]["signals_summary"]  # Barkley's real red zone share


def test_recommend_passes_a_verdict_consistent_with_rank_players_straight_through(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, _RB_ROSTER, [_BARKLEY_SIGNAL_ROW, _COOK_SIGNAL_ROW], pl.DataFrame([_BARKLEY_ROW, _COOK_ROW])
    )
    client = _FakeClient(
        [
            SimpleNamespace(content=[_tool_use_block("rank_players", {"player_names": ["Saquon Barkley", "James Cook"]})]),
            _submit("Start Saquon Barkley over James Cook.", "48% red zone share vs. 26%.", player_id="00-0034844"),
        ]
    )

    result = recommend.recommend(
        "Should I start Saquon Barkley or James Cook?", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir,
        season=_SEASON, as_of_week=_WEEK, client=client, signals_dir=signals_dir,
    )

    assert result["error"] is None
    assert result["player_id"] == "00-0034844"
    assert result["recommendation"] == "Start Saquon Barkley over James Cook."
    assert [c["name"] for c in result["tool_calls"]] == ["rank_players"]
    assert result["tool_calls"][0]["result"]["recommended"]["player_id"] == result["player_id"]
    assert result["data_gaps"] == []
    # And the Feed agrees.
    assert _start_sit_starter(raw_dir, persist_dir, signals_dir)["recommended_starter"]["player_id"] == result["player_id"]


def test_recommend_bounces_a_verdict_that_contradicts_rank_players_until_it_matches(tmp_path, monkeypatch):
    """The code-level half of the rule: the model retrieved the ranking,
    then submitted the OTHER player. The loop hands the contradiction
    back as a tool error instead of returning it; the corrected resubmit
    goes through."""
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, _RB_ROSTER, [_BARKLEY_SIGNAL_ROW, _COOK_SIGNAL_ROW], pl.DataFrame([_BARKLEY_ROW, _COOK_ROW])
    )
    client = _FakeClient(
        [
            SimpleNamespace(content=[_tool_use_block("rank_players", {"player_names": ["Saquon Barkley", "James Cook"]})]),
            _submit("Start James Cook.", "Cook has the better efficiency trend.", player_id="00-0037248", id_="bad_submit"),
            _submit("Start Saquon Barkley.", "Ranking favors Barkley's 48% red zone share.", player_id="00-0034844"),
        ]
    )

    result = recommend.recommend(
        "Should I start Saquon Barkley or James Cook?", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir,
        season=_SEASON, as_of_week=_WEEK, client=client, signals_dir=signals_dir,
    )

    assert result["error"] is None
    assert result["player_id"] == "00-0034844"
    assert result["recommendation"] == "Start Saquon Barkley."
    # The rejected submit is on the record, with why.
    rejected = [c for c in result["tool_calls"] if c["name"] == "submit_recommendation"]
    assert len(rejected) == 1
    assert rejected[0]["input"]["player_id"] == "00-0037248"
    assert "contradicts rank_players" in rejected[0]["result"]["error"]
    # ...and was fed back to the model as an is_error tool_result for the bad submit's tool_use id.
    third_call_messages = client.messages.calls[2]["messages"]
    bounce = third_call_messages[-1]["content"][0]
    assert bounce["tool_use_id"] == "bad_submit" and bounce["is_error"] is True
    assert "Saquon Barkley" in bounce["content"]


def test_a_contradicting_model_ends_in_the_graceful_max_turns_result_never_a_wrong_verdict(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, _RB_ROSTER, [_BARKLEY_SIGNAL_ROW, _COOK_SIGNAL_ROW], pl.DataFrame([_BARKLEY_ROW, _COOK_ROW])
    )
    stubborn = _submit("Start James Cook.", "Trust me.", player_id="00-0037248")
    client = _FakeClient(
        [SimpleNamespace(content=[_tool_use_block("rank_players", {"player_names": ["Saquon Barkley", "James Cook"]})])]
        + [stubborn] * 3
    )

    result = recommend.recommend(
        "Barkley or Cook?", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK,
        client=client, signals_dir=signals_dir, max_turns=4,
    )

    assert result["error"] == "max_turns_exceeded"
    assert result["player_id"] is None
    assert "James Cook" not in result["recommendation"]


# ---- (b) genuinely ambiguous: tie / insufficient data are stated, never resolved by picking ----


def test_rank_players_reports_a_tie_and_recommends_nobody(tmp_path, monkeypatch):
    cook_twin = {**_COOK_SIGNAL_ROW, "player_id": "00-0034844", "player_name": "S.Barkley", "team": "PHI"}
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, _RB_ROSTER, [cook_twin, _COOK_SIGNAL_ROW], pl.DataFrame([_BARKLEY_ROW, _COOK_ROW])
    )
    ctx = _ctx(raw_dir, persist_dir, signals_dir)

    tool = recommend.dispatch_tool("rank_players", {"player_names": ["Saquon Barkley", "James Cook"]}, ctx)

    assert tool["verdict"] == "tied"
    assert tool["recommended"] is None
    assert sorted(e["name"] for e in tool["tied_at_top"]) == ["James Cook", "Saquon Barkley"]
    assert len(tool["ranked"]) == 2


def test_recommend_bounces_a_pick_on_a_tie_and_accepts_an_honest_tie_answer(tmp_path, monkeypatch):
    cook_twin = {**_COOK_SIGNAL_ROW, "player_id": "00-0034844", "player_name": "S.Barkley", "team": "PHI"}
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, _RB_ROSTER, [cook_twin, _COOK_SIGNAL_ROW], pl.DataFrame([_BARKLEY_ROW, _COOK_ROW])
    )
    client = _FakeClient(
        [
            SimpleNamespace(content=[_tool_use_block("rank_players", {"player_names": ["Saquon Barkley", "James Cook"]})]),
            _submit("Start Saquon Barkley.", "He's the bigger name.", player_id="00-0034844", id_="bad_submit"),
            _submit(
                "The ranking can't separate Saquon Barkley and James Cook this week -- it's a genuine tie.",
                "Both score identically on the computed signals; no grounded basis to prefer either.",
            ),
        ]
    )

    result = recommend.recommend(
        "Barkley or Cook?", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK,
        client=client, signals_dir=signals_dir,
    )

    assert result["error"] is None
    assert result["player_id"] is None
    assert "tie" in result["recommendation"]
    rejected = [c for c in result["tool_calls"] if c["name"] == "submit_recommendation"]
    assert len(rejected) == 1 and "tied" in rejected[0]["result"]["error"]


def test_rank_players_reports_insufficient_data_when_only_one_side_has_signals(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, _RB_ROSTER, [_BARKLEY_SIGNAL_ROW], pl.DataFrame([_BARKLEY_ROW, _COOK_ROW])
    )
    ctx = _ctx(raw_dir, persist_dir, signals_dir)

    tool = recommend.dispatch_tool("rank_players", {"player_names": ["Saquon Barkley", "James Cook"]}, ctx)

    assert tool["verdict"] == "insufficient_data"
    assert tool["recommended"] is None
    assert [e["name"] for e in tool["ranked"]] == ["Saquon Barkley"]
    assert tool["unranked"] == [
        {
            "player_id": "00-0037248", "name": "James Cook", "position": "RB", "team": "BUF",
            "reason": "no_signal_data", "has_signals": False,
        }
    ]
    # Same condition under which the Feed skips the position entirely.
    feed = report.generate_report(
        "start_sit", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK,
        signals_dir=signals_dir,
    )
    assert feed["entries"] == []
    assert any("fewer than 2 had computed signals" in note for note in feed["notes"])


def test_recommend_bounces_picking_the_only_player_with_data(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, _RB_ROSTER, [_BARKLEY_SIGNAL_ROW], pl.DataFrame([_BARKLEY_ROW, _COOK_ROW])
    )
    client = _FakeClient(
        [
            SimpleNamespace(content=[_tool_use_block("rank_players", {"player_names": ["Saquon Barkley", "James Cook"]})]),
            _submit("Start Saquon Barkley.", "He's the only one with signals.", player_id="00-0034844", id_="bad_submit"),
            _submit(
                "I can't ground this comparison: James Cook has no computed signal data.",
                "Barkley has signals but Cook has none, so the ranking can't compare them.",
                data_gaps=[{"player_name": "James Cook", "reason": "no_signal_data", "detail": "No computed signals."}],
            ),
        ]
    )

    result = recommend.recommend(
        "Barkley or Cook?", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK,
        client=client, signals_dir=signals_dir,
    )

    assert result["error"] is None
    assert result["player_id"] is None
    assert result["data_gaps"][0]["reason"] == "no_signal_data"
    rejected = [c for c in result["tool_calls"] if c["name"] == "submit_recommendation"]
    assert len(rejected) == 1 and "insufficient_data" in rejected[0]["result"]["error"]


def test_rank_players_reports_ambiguous_and_unknown_names_instead_of_guessing(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, _RB_ROSTER, [_BARKLEY_SIGNAL_ROW, _COOK_SIGNAL_ROW],
        pl.DataFrame([_BARKLEY_ROW, _COOK_ROW, _CHRISTIAN_ROW, _LUKE_ROW]),
    )
    ctx = _ctx(raw_dir, persist_dir, signals_dir)

    tool = recommend.dispatch_tool("rank_players", {"player_names": ["McCaffrey", "Nobody Realname", "Saquon Barkley"]}, ctx)

    assert tool["verdict"] == "insufficient_data"
    reasons = {e["name"]: e["reason"] for e in tool["unranked"]}
    assert reasons == {"McCaffrey": "ambiguous", "Nobody Realname": "unresolved"}
    ambiguous = next(e for e in tool["unranked"] if e["reason"] == "ambiguous")
    assert sorted(c["name"] for c in ambiguous["candidates"]) == ["Christian McCaffrey", "Luke McCaffrey"]


def test_rank_players_needs_two_names():
    ctx = recommend.RecommendContext(raw_dir=None, persist_dir=None, season=_SEASON, as_of_week=_WEEK, player_idx=None)
    assert "error" in recommend.dispatch_tool("rank_players", {"player_names": ["Saquon Barkley"]}, ctx)


def test_rank_players_flags_a_cross_position_comparison(tmp_path, monkeypatch):
    roster = {
        "sleeper_barkley": {"full_name": "Saquon Barkley", "position": "RB", "team": "PHI"},
        "sleeper_luke": {"full_name": "Luke McCaffrey", "position": "WR", "team": "WAS"},
    }
    luke_signal = {**_COOK_SIGNAL_ROW, "player_id": "00-0039355", "player_name": "L.McCaffrey", "team": "WAS"}
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, roster, [_BARKLEY_SIGNAL_ROW, luke_signal], pl.DataFrame([_BARKLEY_ROW, _LUKE_ROW])
    )
    ctx = _ctx(raw_dir, persist_dir, signals_dir)

    tool = recommend.dispatch_tool("rank_players", {"player_names": ["Saquon Barkley", "Luke McCaffrey"]}, ctx)

    assert tool["verdict"] == "clear"
    assert tool["same_position"] is False
    assert "different positions" in tool["note"]


# ---- the real case: Herbert vs. Mahomes on stale prior-season signals ----


def test_herbert_vs_mahomes_verdict_matches_the_feed_under_stale_fallback(tmp_path, monkeypatch):
    """The exact real-usage case, with the real 2025 season-end rows and
    an empty current (2026) season: the Feed's ranking says Herbert
    (opportunity score ~0.46 vs ~0.06 -- Mahomes' 2025 row has no
    computed EPA trend), explicitly stale. rank_players must say the
    same thing, equally explicitly stale."""
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, _QB_ROSTER, [], pl.DataFrame([_HERBERT_ROW, _MAHOMES_ROW]),
        prior_signal_rows=None,
    )
    pl.DataFrame([_HERBERT_2025_SIGNAL_ROW, _MAHOMES_2025_SIGNAL_ROW]).write_parquet(signals_dir / "signals_2025_week19.parquet")
    ctx = _ctx(raw_dir, persist_dir, signals_dir, season=2026, as_of_week=1)

    tool = recommend.dispatch_tool("rank_players", {"player_names": ["Justin Herbert", "Patrick Mahomes"]}, ctx)
    entry = _start_sit_starter(raw_dir, persist_dir, signals_dir, season=2026, as_of_week=1)

    assert tool["verdict"] == "clear"
    assert tool["recommended"]["name"] == entry["recommended_starter"]["name"] == "Justin Herbert"
    assert tool["recommended"]["player_id"] == entry["recommended_starter"]["player_id"] == "00-0036355"
    for e in tool["ranked"]:
        assert e["stale"] is True and e["source_season"] == 2025 and e["source_as_of_week"] == 19
        assert e["signals_summary"].startswith("[STALE -- ")
    assert entry["recommended_starter"]["stale"] is True and entry["recommended_starter"]["source_season"] == 2025
    assert tool["ranked"][1]["signals_summary"] == entry["alternatives_considered"][0]["signals_summary"]


def test_the_original_mahomes_by_reputation_answer_is_bounced_and_herbert_goes_through(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, _QB_ROSTER, [], pl.DataFrame([_HERBERT_ROW, _MAHOMES_ROW])
    )
    pl.DataFrame([_HERBERT_2025_SIGNAL_ROW, _MAHOMES_2025_SIGNAL_ROW]).write_parquet(signals_dir / "signals_2025_week19.parquet")
    client = _FakeClient(
        [
            SimpleNamespace(content=[_tool_use_block("rank_players", {"player_names": ["Justin Herbert", "Patrick Mahomes"]})]),
            _submit(
                "Start Patrick Mahomes.",
                "Current-season signals are stale, so defaulting to Mahomes based on his consistently elite performance history.",
                player_id="00-0033873", id_="bad_submit",
            ),
            _submit(
                "Start Justin Herbert over Patrick Mahomes -- on 2025 season-end numbers, since 2026 has no signals yet.",
                "Herbert's 2025 efficiency trend (+0.19 EPA/play) and 3% red zone share outrank Mahomes' 2% red zone share with no computed EPA trend.",
                player_id="00-0036355",
            ),
        ]
    )

    result = recommend.recommend(
        "Should I start Justin Herbert or Patrick Mahomes?", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir,
        season=2026, as_of_week=1, client=client, signals_dir=signals_dir,
    )

    assert result["error"] is None
    assert result["player_id"] == "00-0036355"
    assert "Herbert" in result["recommendation"]
    rejected = [c for c in result["tool_calls"] if c["name"] == "submit_recommendation"]
    assert len(rejected) == 1 and rejected[0]["input"]["player_id"] == "00-0033873"


# ---- (c) real current-season signals: the normal path is unchanged ----


def test_rank_players_uses_current_season_signals_never_the_stale_fallback_when_they_exist(tmp_path, monkeypatch):
    """A prior-season file with the numbers REVERSED (Cook >> Barkley) is on
    disk too; current-season data exists for both, so the ranking must
    ignore it -- and the Feed's report does the same."""
    reversed_prior = [
        {**_BARKLEY_SIGNAL_ROW, "season": _SEASON - 1, "as_of_week": 19, "red_zone_share": 0.05, "epa_trend": -0.3},
        {**_COOK_SIGNAL_ROW, "season": _SEASON - 1, "as_of_week": 19, "red_zone_share": 0.6, "epa_trend": 0.4},
    ]
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, _RB_ROSTER, [_BARKLEY_SIGNAL_ROW, _COOK_SIGNAL_ROW],
        pl.DataFrame([_BARKLEY_ROW, _COOK_ROW]), prior_signal_rows=reversed_prior,
    )
    ctx = _ctx(raw_dir, persist_dir, signals_dir)

    tool = recommend.dispatch_tool("rank_players", {"player_names": ["Saquon Barkley", "James Cook"]}, ctx)
    entry = _start_sit_starter(raw_dir, persist_dir, signals_dir)

    assert tool["verdict"] == "clear"
    assert tool["recommended"]["name"] == entry["recommended_starter"]["name"] == "Saquon Barkley"
    assert all(e["stale"] is False and e["source_season"] is None for e in tool["ranked"])
    assert not any("STALE" in e["signals_summary"] for e in tool["ranked"])
    assert tool["season"] == _SEASON and tool["as_of_week"] == _WEEK


def test_rank_players_is_as_of_week_filtered(tmp_path, monkeypatch):
    """A later-week file (with Cook far ahead) must not leak into a
    week-5 ranking -- same rule as the report."""
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, _RB_ROSTER, [_BARKLEY_SIGNAL_ROW, _COOK_SIGNAL_ROW], pl.DataFrame([_BARKLEY_ROW, _COOK_ROW])
    )
    future = [
        {**_BARKLEY_SIGNAL_ROW, "as_of_week": _WEEK + 3, "red_zone_share": 0.05},
        {**_COOK_SIGNAL_ROW, "as_of_week": _WEEK + 3, "red_zone_share": 0.6},
    ]
    pl.DataFrame(future).write_parquet(signals_dir / f"signals_{_SEASON}_week{_WEEK + 3}.parquet")
    ctx = _ctx(raw_dir, persist_dir, signals_dir)

    tool = recommend.dispatch_tool("rank_players", {"player_names": ["Saquon Barkley", "James Cook"]}, ctx)

    assert tool["recommended"]["name"] == "Saquon Barkley"
    assert "48%" in tool["recommended"]["signals_summary"]


def test_signal_tables_load_once_per_context(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, _RB_ROSTER, [_BARKLEY_SIGNAL_ROW, _COOK_SIGNAL_ROW], pl.DataFrame([_BARKLEY_ROW, _COOK_ROW])
    )
    ctx = _ctx(raw_dir, persist_dir, signals_dir)
    loads = []
    real_load = ranking.SignalTables.load
    monkeypatch.setattr(ranking.SignalTables, "load", classmethod(lambda cls, *a: loads.append(a) or real_load(*a)))

    recommend.dispatch_tool("rank_players", {"player_names": ["Saquon Barkley", "James Cook"]}, ctx)
    recommend.dispatch_tool("rank_players", {"player_names": ["James Cook", "Saquon Barkley"]}, ctx)

    assert len(loads) == 1


# ---- the prompt rule, and its documented limit ----


def test_system_prompt_requires_rank_players_for_comparisons_and_bars_general_knowledge():
    prompt = recommend._build_system_prompt({"name": "L"}, {"rec": 0.5}, 2026, 1)
    assert "rank_players" in prompt
    assert "'tied'" in prompt and "'insufficient_data'" in prompt
    assert "consistently elite performance history" in prompt  # the real failure, named
    assert "same deterministic ranking the Feed" in prompt
    assert any(t["name"] == "rank_players" for t in recommend.TOOLS)


def test_a_prose_only_contradiction_with_no_player_id_is_not_caught_at_the_code_level(tmp_path, monkeypatch):
    """Documented non-safeguard, same honesty as Phase 3.7's addendum
    test: the guard keys on player_id. A model that names the wrong
    player in prose and sets no player_id passes through -- only the
    system prompt and real-model validation cover that."""
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, _RB_ROSTER, [_BARKLEY_SIGNAL_ROW, _COOK_SIGNAL_ROW], pl.DataFrame([_BARKLEY_ROW, _COOK_ROW])
    )
    client = _FakeClient(
        [
            SimpleNamespace(content=[_tool_use_block("rank_players", {"player_names": ["Saquon Barkley", "James Cook"]})]),
            _submit("Start James Cook.", "Gut feeling."),
        ]
    )

    result = recommend.recommend(
        "Barkley or Cook?", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK,
        client=client, signals_dir=signals_dir,
    )

    assert result["recommendation"] == "Start James Cook."
    assert result["error"] is None
