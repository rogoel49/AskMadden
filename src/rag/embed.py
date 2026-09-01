"""Chunk the Sleeper JSON pulls into retrievable text and embed them into
a local, persistent ChromaDB collection.

Chunking is intentionally scoped to what's useful for Q&A about this
league: team rosters, league settings, matchups, and transactions. The
full ~11k-player Sleeper player pool is used only to resolve player IDs
to names inside those chunks, not embedded player-by-player — most of it
is irrelevant to a 12-team league and would just be retrieval noise.

Phase 2 adds build_signal_chunks(): one chunk per player per as-of-week,
built from src/signals/matchup_signals.py's league-agnostic signals
table (not from the Sleeper raw pull above). Per TODO.md's locked-in
chunk-granularity rule, signal content is always chunked per player (or
per player-per-week), never bundled into a team-sized blob — that would
dilute retrieval for a question about one player's specific matchup.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
import polars as pl

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "sleeper"
CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma"
SIGNALS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "signals"
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


def _fmt(value: Any, suffix: str = "") -> str:
    return f"{value:.2f}{suffix}" if isinstance(value, (int, float)) else "unknown"


def _signal_sentence(row: dict) -> str:
    """One natural-language sentence per available signal on a player-week
    row -- missing signals (nulls from left joins, e.g. a bye week or a
    position NGS doesn't cover) are simply omitted rather than printed as
    "None", so retrieval never surfaces a fabricated-looking non-fact."""
    parts = []
    if row.get("epa_trend") is not None:
        direction = "up" if row["epa_trend"] > 0 else "down"
        parts.append(f"recent efficiency trending {direction} ({_fmt(row['epa_trend'])} EPA/play change)")
    if row.get("red_zone_share") is not None:
        parts.append(f"red zone role share {_fmt(row['red_zone_share'] * 100, '%')}")
    if row.get("target_share") is not None:
        parts.append(f"target share {_fmt(row['target_share'] * 100, '%')}")
    if row.get("opponent"):
        parts.append(f"faces {row['opponent']} in week {row['as_of_week']}")
    if row.get("run_funnel_rate_vs_avg") is not None:
        lean = "run-funnel" if row["run_funnel_rate_vs_avg"] > 0 else "pass-funnel"
        parts.append(f"opponent defense is {lean} relative to average ({_fmt(row['run_funnel_rate_vs_avg'] * 100, '%')})")
    if row.get("implied_total") is not None:
        parts.append(f"team implied total {_fmt(row['implied_total'])} points")
    if row.get("adot") is not None:
        parts.append(f"aDOT {_fmt(row['adot'])} yards")
    if row.get("ryoe_per_att") is not None:
        parts.append(f"rush yards over expected {_fmt(row['ryoe_per_att'])}/attempt")
    if row.get("cpoe") is not None:
        parts.append(f"completion % over expected {_fmt(row['cpoe'])}")
    return ", ".join(parts) if parts else "no signals available yet this season"


def build_signal_chunks(rows: list[dict]) -> list[dict]:
    """Turn matchup_signals.build_signals_table() rows into one retrievable
    chunk per player per as-of-week."""
    chunks = []
    for row in rows:
        text = (
            f"{row['player_name']} ({row.get('team', 'unknown team')}) entering "
            f"{row['season']} week {row['as_of_week']}: {_signal_sentence(row)}."
        )
        chunks.append(
            {
                "id": f"signal:{row['season']}:week{row['as_of_week']}:{row['player_id']}",
                "text": text,
                "metadata": {
                    "type": "player_signal",
                    "player_id": row["player_id"],
                    "week": row["as_of_week"],
                    "season": row["season"],
                },
            }
        )
    return chunks


def load_signal_chunks(signals_dir: Path = SIGNALS_DIR) -> list[dict]:
    """Load every signals_*.parquet file under signals_dir and build
    chunks from all of them combined -- one call to embed the full history
    of computed signal tables, not just the latest week."""
    chunks: list[dict] = []
    for path in sorted(signals_dir.glob("signals_*.parquet")):
        chunks.extend(build_signal_chunks(pl.read_parquet(path).to_dicts()))
    return chunks


def embed(
    chunks: list[dict] | None = None,
    persist_dir: Path = CHROMA_DIR,
    raw_dir: Path = RAW_DIR,
    signals_dir: Path | None = None,
):
    """Embed chunks into a persistent local ChromaDB collection, replacing
    any previous contents. The raw JSON is the source of truth, so a full
    rebuild on each run is simpler than incremental syncing.

    signals_dir: when chunks isn't given explicitly, also embed every
    computed signals table found there (see load_signal_chunks). Pass
    signals_dir=None (or leave it) to skip signals entirely -- e.g. before
    Phase 2 ingest has been run.
    """
    if chunks is None:
        chunks = build_chunks(raw_dir)
        if signals_dir is not None:
            chunks = chunks + load_signal_chunks(signals_dir)

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
    signal_chunks = load_signal_chunks() if SIGNALS_DIR.exists() else []
    embed(chunks + signal_chunks)
    print(f"embedded {len(chunks)} league chunks + {len(signal_chunks)} signal chunks into {CHROMA_DIR}")


if __name__ == "__main__":
    main()
