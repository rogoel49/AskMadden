"""Generate evals/eval_questions.jsonl: systematic retrieval questions
derived directly from this league's own ingested Sleeper data (team
rosters, matchups). Not hand-authored -- see PROJECT_SPEC.md's eval
methodology for why generated questions are preferred over invented ones.

These test retrieval accuracy only: can the RAG pipeline surface facts
we already know are true from our own source data. Decision-accuracy
questions ("which player should you start") need a reasoning agent to
grade against evals/ground_truth.jsonl -- that's Phase 3's recommend.py,
not yet wired into run_eval.py.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.ingest.sleeper import RAW_DIR

EVAL_QUESTIONS_PATH = Path(__file__).resolve().parent / "eval_questions.jsonl"


def _load(raw_dir: Path, filename: str):
    return json.loads((raw_dir / filename).read_text())["data"]


def build(raw_dir: Path = RAW_DIR) -> list[dict]:
    teams = _load(raw_dir, "teams.json")
    players = _load(raw_dir, "players.json")

    questions: list[dict] = []
    for team in teams:
        roster_names = [(players.get(pid) or {}).get("full_name") for pid in (team.get("players") or [])]
        roster_names = [name for name in roster_names if name]
        if not roster_names:
            continue
        questions.append(
            {
                "question": f"Who is on {team.get('team_name')}'s roster?",
                "as_of_week": None,
                "expected_answer_contains": roster_names,
                "type": "roster",
            }
        )

    for path in sorted(raw_dir.glob("matchups_week_*.json")):
        week = int(path.stem.rsplit("_", 1)[-1])
        matchups = json.loads(path.read_text())["data"]
        by_matchup: dict[int, list[dict]] = {}
        for side in matchups:
            by_matchup.setdefault(side.get("matchup_id"), []).append(side)
        for matchup_id, sides in by_matchup.items():
            expected = [f"roster {side.get('roster_id')} scored {side.get('points')}" for side in sides]
            questions.append(
                {
                    "question": f"What was the score of week {week} matchup {matchup_id}?",
                    "as_of_week": week,
                    "expected_answer_contains": expected,
                    "type": "matchup",
                }
            )

    return questions


def main() -> None:
    questions = build()
    with EVAL_QUESTIONS_PATH.open("w") as f:
        for question in questions:
            f.write(json.dumps(question) + "\n")
    print(f"wrote {len(questions)} eval questions -> {EVAL_QUESTIONS_PATH}")


if __name__ == "__main__":
    main()
