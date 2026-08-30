"""Chunk the Sleeper JSON pulls into retrievable text and embed them into
a local, persistent ChromaDB collection.

Chunking is intentionally scoped to what's useful for Q&A about this
league: team rosters, league settings, matchups, and transactions. The
full ~11k-player Sleeper player pool is used only to resolve player IDs
to names inside those chunks, not embedded player-by-player — most of it
is irrelevant to a 12-team league and would just be retrieval noise.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "sleeper"
CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma"
COLLECTION_NAME = "askmadden_league"


def _load(raw_dir: Path, filename: str) -> Any:
    payload = json.loads((raw_dir / filename).read_text())
    return payload["data"]


def _player_label(players: dict, player_id: str) -> str:
    player = players.get(player_id) or {}
    name = player.get("full_name") or f"player {player_id}"
    tag = ", ".join(x for x in [player.get("position"), player.get("team")] if x)
    return f"{name} ({tag})" if tag else name


def build_chunks(raw_dir: Path = RAW_DIR) -> list[dict]:
    """Turn the raw Sleeper JSON pulls in raw_dir into text chunks with
    metadata, each shaped as {"id", "text", "metadata"}."""
    league = _load(raw_dir, "league.json")
    teams = _load(raw_dir, "teams.json")
    players = _load(raw_dir, "players.json")

    chunks: list[dict] = []
    settings = league.get("settings", {})
    chunks.append(
        {
            "id": "league:settings",
            "text": (
                f"{league.get('name')} is a {league.get('season')} season Sleeper "
                f"fantasy football league with {settings.get('num_teams')} teams. "
                f"Waiver budget: {settings.get('waiver_budget')}. "
                f"Playoff teams: {settings.get('playoff_teams')}, "
                f"starting week {settings.get('playoff_week_start')}."
            ),
            "metadata": {"type": "league_settings"},
        }
    )

    for team in teams:
        player_names = [_player_label(players, pid) for pid in (team.get("players") or [])]
        starter_names = [_player_label(players, pid) for pid in (team.get("starters") or [])]
        chunks.append(
            {
                "id": f"team:{team.get('roster_id')}",
                "text": (
                    f"{team.get('team_name')} (owner: {team.get('display_name')}, roster ID "
                    f"{team.get('roster_id')}) roster: {', '.join(player_names) or 'no players listed'}. "
                    f"Starters: {', '.join(starter_names) or 'none set'}."
                ),
                "metadata": {"type": "team_roster", "roster_id": team.get("roster_id")},
            }
        )

    for path in sorted(raw_dir.glob("matchups_week_*.json")):
        week = int(path.stem.rsplit("_", 1)[-1])
        matchups = json.loads(path.read_text())["data"]
        by_matchup: dict[int, list[dict]] = {}
        for side in matchups:
            by_matchup.setdefault(side.get("matchup_id"), []).append(side)
        for matchup_id, sides in by_matchup.items():
            desc = "; ".join(f"roster {s.get('roster_id')} scored {s.get('points')}" for s in sides)
            chunks.append(
                {
                    "id": f"matchup:week{week}:{matchup_id}",
                    "text": f"Week {week} matchup {matchup_id}: {desc}.",
                    "metadata": {"type": "matchup", "week": week},
                }
            )

    for path in sorted(raw_dir.glob("transactions_week_*.json")):
        week = int(path.stem.rsplit("_", 1)[-1])
        transactions = json.loads(path.read_text())["data"]
        for i, txn in enumerate(transactions):
            adds = ", ".join(_player_label(players, pid) for pid in (txn.get("adds") or {}))
            drops = ", ".join(_player_label(players, pid) for pid in (txn.get("drops") or {}))
            text = f"Week {week} {txn.get('type')} transaction"
            if adds:
                text += f", added: {adds}"
            if drops:
                text += f", dropped: {drops}"
            chunks.append(
                {
                    "id": f"transaction:week{week}:{i}",
                    "text": text + ".",
                    "metadata": {"type": "transaction", "week": week},
                }
            )

    return chunks


def embed(chunks: list[dict] | None = None, persist_dir: Path = CHROMA_DIR, raw_dir: Path = RAW_DIR):
    """Embed chunks into a persistent local ChromaDB collection, replacing
    any previous contents. The raw JSON is the source of truth, so a full
    rebuild on each run is simpler than incremental syncing."""
    if chunks is None:
        chunks = build_chunks(raw_dir)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    if chunks:
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
    return collection


def main() -> None:
    chunks = build_chunks()
    embed(chunks)
    print(f"embedded {len(chunks)} chunks into {CHROMA_DIR}")


if __name__ == "__main__":
    main()
