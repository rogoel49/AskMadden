"""CLI Q&A loop: question -> retrieve -> answer.

This is pre-reasoning: it surfaces the chunks retrieve.query() finds most
relevant and prints them as-is. Deciding *when* to call semantic
retrieval vs. a structured lookup (src/rag/lookup.py) vs. future signals
is a Claude tool-use agent's job — that's recommend.py in Phase 3, not
this loop.
"""
from __future__ import annotations

from src.rag import retrieve


def answer(question: str, n_results: int = 3) -> str:
    results = retrieve.query(question, n_results=n_results)
    if not results:
        return "No matching data found. Have you run `python -m src.rag.embed` yet?"
    return "\n".join(f"[{r['metadata'].get('type', 'unknown')}] {r['text']}" for r in results)


def main() -> None:
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
