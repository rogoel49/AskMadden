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
    max_week: int | None = None,
    max_season: int | None = None,
    persist_dir: Path = CHROMA_DIR,
) -> dict | None:
    """The signal chunk for player_id, resolved by exact metadata match --
    never by embedding similarity, so a shared-surname collision can never
    substitute the wrong player's chunk. If season/as_of_week aren't
    given, returns the most recent (highest season, then week) chunk on
    record for that player. Returns None if no signal chunk has been
    embedded for that player at all (e.g. signals haven't been computed
    yet, or the player has no measurable role).

    max_week (mutually exclusive with as_of_week -- pass one or the
    other, not both) bounds the search to week <= max_week within
    `season` instead of pinning an exact week: used by
    query_player_signal_with_fallback() to check "does this player have
    ANY current-season signal as of this point in the season" without
    exact-matching one specific week, while still never leaking a later
    week's data into an as-of-week-filtered lookup (see CLAUDE.md's
    as-of-date-filtering rule).

    max_season (mutually exclusive with season) bounds the search to
    season <= max_season -- used by query_player_signal_with_fallback()
    to find the most recent chunk from strictly BEFORE the requested
    season, so a same-or-later-season chunk (e.g. one that got excluded
    by an as_of_week/max_week bound above) can never be picked up as a
    "prior season" fallback."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(COLLECTION_NAME)

    conditions: list[dict] = [{"type": "player_signal"}, {"player_id": player_id}]
    if season is not None:
        conditions.append({"season": season})
    elif max_season is not None:
        conditions.append({"season": {"$lte": max_season}})
    if as_of_week is not None:
        conditions.append({"week": as_of_week})
    elif max_week is not None:
        conditions.append({"week": {"$lte": max_week}})

    result = collection.get(where={"$and": conditions} if len(conditions) > 1 else conditions[0])
    if not result["ids"]:
        return None

    rows = list(zip(result["ids"], result["documents"], result["metadatas"]))
    rows.sort(key=lambda row: (row[2].get("season", 0), row[2].get("week", 0)))
    id_, text, metadata = rows[-1]
    return {"id": id_, "text": text, "metadata": metadata}


def query_player_signal_with_fallback(
    player_id: str,
    season: int,
    as_of_week: int,
    persist_dir: Path = CHROMA_DIR,
) -> tuple[dict | None, bool]:
    """query_player_signal(), but when the requested season has NO signal
    chunk at all for this player (e.g. a new season's games haven't been
    played yet -- nflverse has nothing to compute from until then, see
    src/reasoning/report.py's module docstring for the same gap in the
    report path), falls back to that player's most recent chunk on record
    from an earlier season, explicitly flagged stale (never silently
    presented as current). Returns (chunk_or_None, is_stale).

    This does NOT fall back within the same season -- if any chunk exists
    for `season` at or before as_of_week (e.g. a signals refresh that's a
    week behind), that's used as-is and is_stale is False. That's a
    `src/scheduler/refresh.py` cadence gap (a different, smaller problem,
    already tracked in TODO.md), not the "no current-season data exists
    yet" gap this fallback exists for -- conflating the two would blend a
    merely-slightly-outdated number with a genuinely-different-season one
    under the same "stale" label. The same-season check is bounded to
    week <= as_of_week (never a later week) so this never leaks future
    data into an as-of-week-filtered lookup.
    """
    exact = query_player_signal(player_id, season=season, as_of_week=as_of_week, persist_dir=persist_dir)
    if exact is not None:
        return exact, False

    same_season = query_player_signal(player_id, season=season, max_week=as_of_week, persist_dir=persist_dir)
    if same_season is not None:
        return same_season, False

    prior = query_player_signal(player_id, max_season=season - 1, persist_dir=persist_dir)
    if prior is None:
        return None, False
    return prior, True
