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
import os
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


def _players_at_position(team: dict, players: dict, position: str) -> list[dict]:
    return [
        {"player_id": pid, **(players.get(pid) or {})}
        for pid in (team.get("players") or [])
        if (players.get(pid) or {}).get("position", "").upper() == position.upper()
    ]


def players_by_position_for_owner(owner_display_name: str, position: str, raw_dir: Path = RAW_DIR) -> list[dict]:
    """Return the rostered players at `position` (e.g. "QB") for the team
    owned by owner_display_name (case-insensitive exact match)."""
    teams, players = _load_teams_and_players(raw_dir)
    needle = owner_display_name.lower()
    for team in teams:
        if (team.get("display_name") or "").lower() == needle:
            return _players_at_position(team, players, position)
    return []


def current_roster(raw_dir: Path = RAW_DIR) -> dict:
    """Resolve the team configured as "mine" via the MY_ROSTER_ID env var.

    Single source of truth for "my"-flavored questions (my roster, my
    quarterbacks, etc.) -- callers read this instead of each hardcoding
    or re-parsing which roster_id belongs to the current user.
    """
    my_roster_id = os.environ.get("MY_ROSTER_ID")
    if not my_roster_id:
        raise RuntimeError(
            "MY_ROSTER_ID is not set. Copy .env.example to .env and set it to "
            "your roster_id from data/raw/sleeper/teams.json."
        )
    teams, _ = _load_teams_and_players(raw_dir)
    for team in teams:
        if str(team.get("roster_id")) == str(my_roster_id):
            return team
    raise RuntimeError(f"No team found with roster_id={my_roster_id!r} in teams.json.")


def my_players_by_position(position: str, raw_dir: Path = RAW_DIR) -> list[dict]:
    """Rostered players at `position` for the team configured as "mine"
    (see current_roster())."""
    team = current_roster(raw_dir)
    _, players = _load_teams_and_players(raw_dir)
    return _players_at_position(team, players, position)


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
