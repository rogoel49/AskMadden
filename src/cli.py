"""CLI Q&A loop: question -> retrieve -> answer.

This is pre-reasoning: it surfaces the chunks retrieve.query() finds most
relevant and prints them as-is. Deciding *when* to call semantic
retrieval vs. a structured lookup (src/rag/lookup.py) vs. future signals
is a Claude tool-use agent's job — that's recommend.py in Phase 3, not
this loop. The one exception carved out here is identity ("my ...")
questions: semantic search has no concept of identity, so ANY question
containing "my" routes to lookup.py's current_roster() (MY_ROSTER_ID)
instead of retrieve.query() -- never the reverse. This is a routing
rule, not a string-matching exercise: getting the specific position
wrong (or not recognizing one at all) still falls back to the user's
full roster, never to retrieve.query(), so a new phrasing can never leak
into semantic search and return some other team's data. This is
deliberately simple string matching, not real intent understanding --
cli.py is a Phase 1/2 dev tool per PROJECT_SPEC.md, not the shipped
interface (that's recommend.py's Claude tool-use agent in Phase 3).
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
    "runningback": "RB",
    "runningbacks": "RB",
    "rb": "RB",
    "rbs": "RB",
    "wide receiver": "WR",
    "wide receivers": "WR",
    "widereceiver": "WR",
    "widereceivers": "WR",
    "wr": "WR",
    "wrs": "WR",
    "tight end": "TE",
    "tight ends": "TE",
    "tightend": "TE",
    "tightends": "TE",
    "te": "TE",
    "tes": "TE",
    "kicker": "K",
    "kickers": "K",
    "defense": "DEF",
}


def _detect_position(question: str) -> str | None:
    """Return the position (e.g. "QB") mentioned in question, if any."""
    q = question.lower()
    for word, position in _POSITION_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", q):
            return position
    return None


def _answer_my_question(question: str) -> str:
    position = _detect_position(question)
    try:
        if position:
            players = lookup.my_players_by_position(position)
        else:
            players = lookup.my_players()
    except RuntimeError as e:
        return str(e)

    label = f"{position}(s)" if position else "roster"
    if not players:
        return f"No {position} players found on your roster." if position else "Your roster has no players listed."
    names = [p.get("full_name") or f"player {p['player_id']}" for p in players]
    return f"Your {label}: {', '.join(names)}"


def answer(question: str, n_results: int = 3) -> str:
    if re.search(r"\bmy\b", question.lower()):
        return _answer_my_question(question)

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
