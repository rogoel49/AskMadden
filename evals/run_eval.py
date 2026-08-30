"""Backtest harness -- retrieval-accuracy scoring for Phase 1.

For each question in eval_questions.jsonl, retrieves from the ChromaDB
collection built by src/rag/embed.py, filtered to only chunks that would
have been available as of that question's week (no future-week matchup
or transaction leakage into the retrieved context), and checks whether
the retrieved text contains every expected fact.

Decision accuracy -- grading an actual start/sit recommendation against
evals/ground_truth.jsonl -- needs a reasoning agent. That's Phase 3's
recommend.py; this harness reports it as unavailable until then, rather
than faking a score for a feature that doesn't exist yet.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import chromadb

from src.rag.embed import CHROMA_DIR, COLLECTION_NAME

EVAL_QUESTIONS_PATH = Path(__file__).resolve().parent / "eval_questions.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_questions(path: Path = EVAL_QUESTIONS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def query_as_of(question: dict, collection, n_results: int = 3, overfetch: int = 20) -> list[dict]:
    """Query the collection, then drop any chunk tagged with a week later
    than the question's as_of_week -- chunks with no week (rosters, league
    settings) are always in scope."""
    as_of_week = question.get("as_of_week")
    results = collection.query(query_texts=[question["question"]], n_results=overfetch)

    candidates = [
        {"id": id_, "text": text, "metadata": metadata}
        for id_, text, metadata in zip(results["ids"][0], results["documents"][0], results["metadatas"][0])
        if as_of_week is None or "week" not in metadata or metadata["week"] <= as_of_week
    ]
    return candidates[:n_results]


def score_question(question: dict, retrieved: list[dict]) -> bool:
    combined_text = " ".join(r["text"] for r in retrieved)
    return all(fragment in combined_text for fragment in question["expected_answer_contains"])


def run(persist_dir: Path = CHROMA_DIR, questions_path: Path = EVAL_QUESTIONS_PATH) -> dict:
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(COLLECTION_NAME)
    questions = load_questions(questions_path)

    results = []
    for question in questions:
        retrieved = query_as_of(question, collection)
        results.append(
            {
                "question": question["question"],
                "type": question["type"],
                "correct": score_question(question, retrieved),
            }
        )

    total = len(results)
    correct = sum(r["correct"] for r in results)
    return {
        "date": date.today().isoformat(),
        "total_questions": total,
        "correct": correct,
        "retrieval_accuracy": correct / total if total else None,
        "decision_accuracy": None,  # not scoreable until Phase 3's recommend.py exists
        "results": results,
    }


def main() -> None:
    summary = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{summary['date']}_run.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"retrieval accuracy: {summary['correct']}/{summary['total_questions']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
