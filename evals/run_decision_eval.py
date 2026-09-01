"""Backtest harness -- decision-accuracy scoring for Phase 3.

For each dilemma in decision_questions.jsonl, calls
src/reasoning/recommend.recommend() with the "should I start A or B"
question, as-of-week filtered to the dilemma's own week (so the agent's
signals never see beyond what would have been knowable before that
week's games kicked off -- the same as-of-week discipline
matchup_signals.py enforces), and checks whether the player it
recommended is the one who actually scored more that week
(evals/ground_truth.jsonl, real nflverse data).

This is decision accuracy, not retrieval accuracy -- see run_eval.py for
that. CLAUDE.md requires scoring the two separately; this harness never
touches run_eval.py's retrieval_accuracy number and reports its own as
None here for the same reason (nothing here measures it).

Prerequisite: decision_questions.jsonl (see build_decision_questions.py)
and a Chroma collection containing computed signal chunks for each
dilemma's season/week must already exist -- this harness doesn't
compute or embed signals itself, matching run_eval.py's existing
convention of assuming the ingest/signals/embed pipeline has already
been run for the relevant weeks.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import anthropic

from src.rag import player_index
from src.rag.embed import CHROMA_DIR, RAW_DIR
from src.reasoning import recommend

DECISION_QUESTIONS_PATH = Path(__file__).resolve().parent / "decision_questions.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_questions(path: Path = DECISION_QUESTIONS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _resolve_expected_winner_id(question: dict, idx) -> str | None:
    winner_name = question["player_a"] if question["expected_winner"] == "a" else question["player_b"]
    result = player_index.resolve_player(winner_name, idx)
    return result.candidates[0].player_id if result.match_type == "exact" else None


def score_dilemma(question: dict, outcome: dict, expected_player_id: str | None) -> bool | None:
    """True/False if we can tell which real player the agent
    recommended, None if we can't (e.g. the ground-truth winner's name
    doesn't resolve unambiguously, or the agent's answer cites no
    player_id and neither player's name is unambiguously present in the
    recommendation text) -- an unscoreable dilemma must never silently
    count as wrong."""
    if expected_player_id is None:
        return None
    if outcome.get("player_id"):
        return outcome["player_id"] == expected_player_id

    text = (outcome.get("recommendation") or "").lower()
    a_in = question["player_a"].lower() in text
    b_in = question["player_b"].lower() in text
    if a_in and not b_in:
        return question["expected_winner"] == "a"
    if b_in and not a_in:
        return question["expected_winner"] == "b"
    return None


def run(
    questions_path: Path = DECISION_QUESTIONS_PATH,
    raw_dir: Path = RAW_DIR,
    persist_dir: Path = CHROMA_DIR,
    client: "anthropic.Anthropic | None" = None,
) -> dict:
    questions = load_questions(questions_path)
    client = client or anthropic.Anthropic()

    results = []
    for question in questions:
        season, week = question["season"], question["week"]
        idx = player_index.build_player_index(season)
        expected_id = _resolve_expected_winner_id(question, idx)

        outcome = recommend.recommend(
            question["question"],
            raw_dir=raw_dir,
            persist_dir=persist_dir,
            season=season,
            as_of_week=week,
            client=client,
        )
        correct = score_dilemma(question, outcome, expected_id)
        results.append(
            {
                "question": question["question"],
                "expected_winner": question["player_a"]
                if question["expected_winner"] == "a"
                else question["player_b"],
                "recommended": outcome.get("recommendation"),
                "recommended_player_id": outcome.get("player_id"),
                "correct": correct,
            }
        )

    scored = [r for r in results if r["correct"] is not None]
    correct_count = sum(1 for r in scored if r["correct"])
    return {
        "date": date.today().isoformat(),
        "total_dilemmas": len(results),
        "scored_dilemmas": len(scored),
        "correct": correct_count,
        "decision_accuracy": correct_count / len(scored) if scored else None,
        "retrieval_accuracy": None,  # scored separately by run_eval.py, not here
        "results": results,
    }


def main() -> None:
    summary = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{summary['date']}_decision_run.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(
        f"decision accuracy: {summary['correct']}/{summary['scored_dilemmas']} "
        f"(of {summary['total_dilemmas']} total dilemmas)"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
