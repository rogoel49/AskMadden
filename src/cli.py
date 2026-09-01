"""CLI Q&A loop: question -> retrieve -> answer.

This is pre-reasoning: it surfaces the chunks retrieve.query() finds most
relevant and prints them as-is. Deciding *when* to call semantic
retrieval vs. a structured lookup (src/rag/lookup.py) vs. future signals
is a Claude tool-use agent's job — that's recommend.py in Phase 3, not
this loop. The one exception carved out here is "my"-flavored position
questions ("who are my quarterbacks"): semantic search has no concept of
identity, so that resolution has to go through lookup.py's
current_roster() (MY_ROSTER_ID) instead -- see src/rag/lookup.py.
"""
from __future__ import annotations

import re

from dotenv import load_dotenv

from src.rag import lookup, retrieve

_POSITION_WORDS = {
    "quarterback": "QB",
    "quarterbacks": "QB",
    "qb": "QB",
    "qbs": "QB",
    "running back": "RB",
    "running backs": "RB",
    "rb": "RB",
    "rbs": "RB",
    "wide receiver": "WR",
    "wide receivers": "WR",
    "wr": "WR",
    "wrs": "WR",
    "tight end": "TE",
    "tight ends": "TE",
    "te": "TE",
    "tes": "TE",
    "kicker": "K",
    "kickers": "K",
    "defense": "DEF",
}


def _detect_my_position(question: str) -> str | None:
    """Return the position (e.g. "QB") if question is a "my <position>"
    question, else None."""
    q = question.lower()
    if not re.search(r"\bmy\b", q):
        return None
    for word, position in _POSITION_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", q):
            return position
    return None


def answer(question: str, n_results: int = 3) -> str:
    position = _detect_my_position(question)
    if position:
        try:
            players = lookup.my_players_by_position(position)
        except RuntimeError as e:
            return str(e)
        if not players:
            return f"No {position} players found on your roster."
        names = [p.get("full_name") or f"player {p['player_id']}" for p in players]
        return f"Your {position}(s): {', '.join(names)}"

    results = retrieve.query(question, n_results=n_results)
    if not results:
        return "No matching data found. Have you run `python -m src.rag.embed` yet?"
    return "\n".join(f"[{r['metadata'].get('type', 'unknown')}] {r['text']}" for r in results)


def main() -> None:
    load_dotenv()
    print("Ask Madden -- type a question (or 'exit' to quit)")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        print(answer(question))
        print()


if __name__ == "__main__":
    main()
