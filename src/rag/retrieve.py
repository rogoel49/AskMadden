"""Query the ChromaDB collection built by embed.py.

query() is pure semantic similarity search -- good for open-ended
questions, unreliable for a specific named player (see
src/rag/player_index.py's docstring for why: a shared surname can
outrank the intended player in embedding space). query_player_signal()
is the structured alternative for that case: an exact metadata filter,
never a similarity ranking, so a resolved player_id always returns that
player's own chunk or nothing at all -- never a namesake's.
"""
from __future__ import annotations

from pathlib import Path

import chromadb

from src.rag.embed import CHROMA_DIR, COLLECTION_NAME


def query(question: str, n_results: int = 5, persist_dir: Path = CHROMA_DIR) -> list[dict]:
    """Return the n_results chunks most relevant to question, each as
    {"id", "text", "metadata", "distance"} ordered by relevance."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(query_texts=[question], n_results=n_results)
    return [
        {"id": id_, "text": text, "metadata": metadata, "distance": distance}
        for id_, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def query_player_signal(
    player_id: str,
    season: int | None = None,
    as_of_week: int | None = None,
    persist_dir: Path = CHROMA_DIR,
) -> dict | None:
    """The signal chunk for player_id, resolved by exact metadata match --
    never by embedding similarity, so a shared-surname collision can never
    substitute the wrong player's chunk. If season/as_of_week aren't
    given, returns the most recent (highest season, then week) chunk on
    record for that player. Returns None if no signal chunk has been
    embedded for that player at all (e.g. signals haven't been computed
    yet, or the player has no measurable role)."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(COLLECTION_NAME)

    conditions: list[dict] = [{"type": "player_signal"}, {"player_id": player_id}]
    if season is not None:
        conditions.append({"season": season})
    if as_of_week is not None:
        conditions.append({"week": as_of_week})

    result = collection.get(where={"$and": conditions} if len(conditions) > 1 else conditions[0])
    if not result["ids"]:
        return None

    rows = list(zip(result["ids"], result["documents"], result["metadatas"]))
    rows.sort(key=lambda row: (row[2].get("season", 0), row[2].get("week", 0)))
    id_, text, metadata = rows[-1]
    return {"id": id_, "text": text, "metadata": metadata}
