from types import SimpleNamespace

import polars as pl
import pytest

from evals import run_decision_eval as rde
from src.rag import embed

_QUESTION = {
    "season": 2024,
    "week": 5,
    "position": "RB",
    "player_a": "Christian McCaffrey",
    "player_a_points": 20.0,
    "player_b": "Luke McCaffrey",
    "player_b_points": 8.0,
    "expected_winner": "a",
    "question": "Week 5, 2024: should I start Christian McCaffrey or Luke McCaffrey at RB?",
}

_CHRISTIAN_ROW = {
    "gsis_id": "00-0033280", "display_name": "Christian McCaffrey", "position": "RB",
    "latest_team": "SF", "last_season": 2026,
}
_LUKE_ROW = {
    "gsis_id": "00-0039355", "display_name": "Luke McCaffrey", "position": "WR",
    "latest_team": "WAS", "last_season": 2026,
}


# ---- score_dilemma: pure scoring logic, no network ----


def test_score_dilemma_true_when_player_id_matches_expected():
    outcome = {"player_id": "00-0033280", "recommendation": "Start him."}
    assert rde.score_dilemma(_QUESTION, outcome, expected_player_id="00-0033280") is True


def test_score_dilemma_false_when_player_id_is_the_other_player():
    outcome = {"player_id": "00-0039355", "recommendation": "Start him."}
    assert rde.score_dilemma(_QUESTION, outcome, expected_player_id="00-0033280") is False


def test_score_dilemma_falls_back_to_name_in_text_when_no_player_id():
    outcome = {"player_id": None, "recommendation": "Start Christian McCaffrey this week."}
    assert rde.score_dilemma(_QUESTION, outcome, expected_player_id="00-0033280") is True


def test_score_dilemma_returns_none_when_unscoreable():
    outcome = {"player_id": None, "recommendation": "It's close, hard to say."}
    assert rde.score_dilemma(_QUESTION, outcome, expected_player_id="00-0033280") is None


def test_score_dilemma_returns_none_when_expected_winner_cannot_be_resolved():
    outcome = {"player_id": "00-0033280", "recommendation": "Start him."}
    assert rde.score_dilemma(_QUESTION, outcome, expected_player_id=None) is None


# ---- run(): end-to-end with a fake Claude client, no network ----


def _tool_use_block(name, input_, id_="tool_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_)


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def test_run_scores_a_correctly_recommended_dilemma(tmp_path, monkeypatch):
    raw_dir = tmp_path / "sleeper"
    persist_dir = tmp_path / "chroma"
    questions_path = tmp_path / "decision_questions.jsonl"

    from tests.test_recommend import _LEAGUE_ID, _seed_league

    _seed_league(raw_dir)
    signal_rows = [
        {
            "player_id": "00-0033280", "player_name": "C.McCaffrey", "team": "SF", "season": 2024, "as_of_week": 5,
            "epa_trend": 0.3, "red_zone_share": 0.4, "target_share": 0.15, "opponent": "SEA",
            "run_funnel_rate_vs_avg": 0.05, "implied_total": 24.0, "adot": None, "ryoe_per_att": 1.1, "cpoe": None,
        }
    ]
    embed.embed(embed.build_signal_chunks(signal_rows), persist_dir=persist_dir)

    import json

    questions_path.write_text(json.dumps(_QUESTION) + "\n")

    import src.rag.player_index as player_index_module

    monkeypatch.setattr(player_index_module.nflverse, "fetch_players", lambda: pl.DataFrame([_CHRISTIAN_ROW, _LUKE_ROW]))

    client = _FakeClient(
        [
            SimpleNamespace(
                content=[
                    _tool_use_block(
                        "submit_recommendation",
                        {
                            "recommendation": "Start Christian McCaffrey.",
                            "reasoning": "Better efficiency trend and red zone share.",
                            "player_id": "00-0033280",
                        },
                    )
                ]
            )
        ]
    )

    summary = rde.run(_LEAGUE_ID, questions_path=questions_path, raw_dir=raw_dir, persist_dir=persist_dir, client=client)

    assert summary["total_dilemmas"] == 1
    assert summary["scored_dilemmas"] == 1
    assert summary["correct"] == 1
    assert summary["decision_accuracy"] == 1.0
    assert summary["retrieval_accuracy"] is None


def test_run_appends_each_result_to_the_progress_file_and_skips_them_on_rerun(tmp_path, monkeypatch):
    """One paid API call per dilemma; a crash must cost one dilemma, not the
    run (2026-09-20: ~460 answers lost when credit ran out mid-run)."""
    import json

    from evals import run_decision_eval as rde

    questions = [
        {"season": 2024, "week": 5, "position": "RB", "question": f"Week 5, 2024: should I start A{i} or B{i} at RB?",
         "player_a": f"A{i}", "player_b": f"B{i}", "expected_winner": "a"}
        for i in range(3)
    ]
    qpath = tmp_path / "q.jsonl"
    qpath.write_text("\n".join(json.dumps(q) for q in questions) + "\n")
    progress = tmp_path / "progress.jsonl"
    asked = []

    monkeypatch.setattr(rde.player_index, "build_player_index", lambda season: None)
    monkeypatch.setattr(rde, "_resolve_expected_winner_id", lambda q, idx: "pid-" + q["player_a"])

    crashed = []

    def fake_recommend(question, league_id, **kwargs):
        asked.append(question)
        if len(asked) == 2 and not crashed:  # the account runs dry once, on the second dilemma
            crashed.append(True)
            raise RuntimeError("credit balance too low")
        return {"recommendation": "Start A", "player_id": "pid-" + question.split("start ")[1].split(" ")[0]}

    monkeypatch.setattr(rde.recommend, "recommend", fake_recommend)
    with pytest.raises(RuntimeError):
        rde.run("1", questions_path=qpath, raw_dir=tmp_path, persist_dir=tmp_path, client=object(), progress_path=progress)
    assert len(progress.read_text().splitlines()) == 1  # the one finished dilemma survived the crash

    asked.clear()
    summary = rde.run("1", questions_path=qpath, raw_dir=tmp_path, persist_dir=tmp_path, client=object(), progress_path=progress)
    assert asked == [questions[1]["question"], questions[2]["question"]]  # the finished one was not re-bought
    assert summary["total_dilemmas"] == 3 and summary["correct"] == 3
    assert len(progress.read_text().splitlines()) == 3
