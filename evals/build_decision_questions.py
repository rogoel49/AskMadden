"""Generate evals/decision_questions.jsonl: programmatic start/sit
pairwise dilemmas built from evals/ground_truth.jsonl -- this is
PROJECT_SPEC.md's "systematic set" for Phase 3 decision-accuracy
grading, distinct from build_eval_questions.py's Sleeper-sourced
retrieval questions. Never hand-authored: every dilemma and its
expected_winner trace back to a measured nflverse stat line already in
ground_truth.jsonl, per CLAUDE.md's non-negotiable rule.

Pair selection is symmetric in who ends up winning: a pair only needs
both players to have scored at least `min_points` that week (i.e. both
were fantasy-relevant, not one stud vs. one inactive scrub). That
selection criterion never looks at *which* player scored higher, so it
can't bias the eval toward whichever side happened to win -- it only
uses the postgame result to confirm the pairing was a real, plausible
dilemma, not to pick a favorable matchup.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "ground_truth.jsonl"
DECISION_QUESTIONS_PATH = Path(__file__).resolve().parent / "decision_questions.jsonl"
DEFAULT_MIN_POINTS = 8.0
DEFAULT_MAX_PAIRS = 20


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build(
    rows: list[dict] | None = None,
    min_points: float = DEFAULT_MIN_POINTS,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    seed: int = 0,
) -> list[dict]:
    """Same-position, same-week pairs where both players cleared
    min_points, deterministically sampled down to at most max_pairs (a
    live decision-accuracy run costs one Claude API call per dilemma --
    keep the default set small and cheap to run)."""
    if rows is None:
        rows = load_ground_truth()

    groups: dict[tuple[int, int, str], list[dict]] = {}
    for row in rows:
        if row["fantasy_points"] < min_points:
            continue
        key = (row["season"], row["week"], row["position"])
        groups.setdefault(key, []).append(row)

    pairs = []
    for (season, week, position), players in groups.items():
        for a, b in itertools.combinations(sorted(players, key=lambda p: p["player_name"]), 2):
            if a["fantasy_points"] == b["fantasy_points"]:
                continue  # no clear correct answer -- skip genuine ties
            pairs.append((season, week, position, a, b))

    rng = random.Random(seed)
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]

    questions = []
    for season, week, position, a, b in pairs:
        winner = "a" if a["fantasy_points"] > b["fantasy_points"] else "b"
        questions.append(
            {
                "season": season,
                "week": week,
                "position": position,
                "player_a": a["player_name"],
                "player_a_points": a["fantasy_points"],
                "player_b": b["player_name"],
                "player_b_points": b["fantasy_points"],
                "expected_winner": winner,
                "question": f"Week {week}, {season}: should I start {a['player_name']} or {b['player_name']} at {position}?",
            }
        )
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate start/sit decision dilemmas from ground_truth.jsonl")
    parser.add_argument("--min-points", type=float, default=DEFAULT_MIN_POINTS)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    questions = build(min_points=args.min_points, max_pairs=args.max_pairs, seed=args.seed)
    with DECISION_QUESTIONS_PATH.open("w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
    print(f"wrote {len(questions)} decision dilemmas -> {DECISION_QUESTIONS_PATH}")


if __name__ == "__main__":
    main()
