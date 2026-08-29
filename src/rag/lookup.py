"""Structured (exact) lookups over the raw Sleeper pull.

retrieve.query() does semantic similarity search over embedded text
chunks — good for open-ended natural-language questions, but unreliable
for exact filters, joins, or counts (e.g. "which team has the most
players from NFL team X" isn't a "most similar chunk" question, it's an
aggregation). These functions answer that class of question directly
against teams.json/players.json instead of going through embeddings.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.rag.embed import RAW_DIR


def _load(raw_dir: Path, filename: str) -> Any:
    payload = json.loads((raw_dir / filename).read_text())
    return payload["data"]


def _load_teams_and_players(raw_dir: Path) -> tuple[list[dict], dict]:
    return _load(raw_dir, "teams.json"), _load(raw_dir, "players.json")


def find_team_by_player(player_name: str, raw_dir: Path = RAW_DIR) -> dict | None:
    """Return the team whose roster contains a player matching player_name
    (case-insensitive substring match on full_name), or None."""
    teams, players = _load_teams_and_players(raw_dir)
    needle = player_name.lower()
    for team in teams:
        for pid in team.get("players") or []:
            full_name = (players.get(pid) or {}).get("full_name", "")
            if needle in full_name.lower():
                return team
    return None


def players_by_position_for_owner(owner_display_name: str, position: str, raw_dir: Path = RAW_DIR) -> list[dict]:
    """Return the rostered players at `position` (e.g. "QB") for the team
    owned by owner_display_name (case-insensitive exact match)."""
    teams, players = _load_teams_and_players(raw_dir)
    needle = owner_display_name.lower()
    matches = []
    for team in teams:
        if (team.get("display_name") or "").lower() != needle:
            continue
        for pid in team.get("players") or []:
            player = players.get(pid) or {}
            if (player.get("position") or "").upper() == position.upper():
                matches.append({"player_id": pid, **player})
    return matches


def teams_by_nfl_team_count(nfl_team: str, raw_dir: Path = RAW_DIR) -> list[tuple[dict, int]]:
    """Return (team, count) pairs — how many of each fantasy team's
    rostered players currently play for the given real NFL team (e.g.
    "SF") — sorted descending, excluding teams with zero such players."""
    teams, players = _load_teams_and_players(raw_dir)
    needle = nfl_team.upper()
    counts = []
    for team in teams:
        count = sum(
            1
            for pid in (team.get("players") or [])
            if (players.get(pid) or {}).get("team", "").upper() == needle
        )
        if count:
            counts.append((team, count))
    return sorted(counts, key=lambda pair: pair[1], reverse=True)
