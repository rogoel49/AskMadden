import json

from evals import run_eval
from src.rag import embed
from tests.test_embed import _seed_raw_dir, _write


class FakeCollection:
    """Stands in for a chromadb collection whose .query() always returns
    a fixed set of candidates regardless of query_texts, so we can test
    query_as_of's week-filtering logic in isolation."""

    def __init__(self, candidates):
        self._candidates = candidates

    def query(self, query_texts, n_results):
        ids, docs, metas = zip(*self._candidates) if self._candidates else ([], [], [])
        return {"ids": [list(ids)], "documents": [list(docs)], "metadatas": [list(metas)]}


def test_query_as_of_excludes_future_weeks():
    collection = FakeCollection(
        [
            ("week3", "week 3 fact", {"week": 3}),
            ("week4", "week 4 fact -- shouldn't leak", {"week": 4}),
            ("roster", "roster fact, no week", {"type": "team_roster"}),
        ]
    )
    question = {"question": "what happened", "as_of_week": 3}

    results = run_eval.query_as_of(question, collection, n_results=10)

    ids = {r["id"] for r in results}
    assert ids == {"week3", "roster"}
    assert "week4" not in ids


def test_run_scores_matchup_question_without_future_leakage(tmp_path):
    raw_dir = tmp_path / "sleeper"
    persist_dir = tmp_path / "chroma"
    questions_path = tmp_path / "eval_questions.jsonl"

    _seed_raw_dir(raw_dir)
    # a later week that a week-3 question must not be graded against
    _write(raw_dir, "matchups_week_4.json", [{"roster_id": 1, "matchup_id": 1, "points": 999.9}])

    embed.embed(embed.build_chunks(raw_dir), persist_dir=persist_dir, raw_dir=raw_dir)

    question = {
        "question": "What was the score of week 3 matchup 1?",
        "as_of_week": 3,
        "expected_answer_contains": ["roster 1 scored 110.5", "roster 2 scored 98.2"],
        "type": "matchup",
    }
    questions_path.write_text(json.dumps(question) + "\n")

    summary = run_eval.run(persist_dir=persist_dir, questions_path=questions_path)

    assert summary["total_questions"] == 1
    assert summary["correct"] == 1
    assert summary["retrieval_accuracy"] == 1.0
    assert summary["decision_accuracy"] is None
