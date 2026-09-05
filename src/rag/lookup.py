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


def _team_by_owner(owner_display_name: str, teams: list[dict]) -> dict | None:
    needle = owner_display_name.lower()
    for team in teams:
        if (team.get("display_name") or "").lower() == needle:
            return team
    return None


def _team_by_roster_id(roster_id: Any, teams: list[dict]) -> dict | None:
    for team in teams:
        if str(team.get("roster_id")) == str(roster_id):
            return team
    return None


def players_by_position_for_owner(owner_display_name: str, position: str, raw_dir: Path = RAW_DIR) -> list[dict]:
    """Return the rostered players at `position` (e.g. "QB") for the team
    owned by owner_display_name (case-insensitive exact match)."""
    teams, players = _load_teams_and_players(raw_dir)
    team = _team_by_owner(owner_display_name, teams)
    return _players_at_position(team, players, position) if team else []


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


def my_players(raw_dir: Path = RAW_DIR) -> list[dict]:
    """All rostered players for the team configured as "mine" (see
    current_roster()), unfiltered by position."""
    team = current_roster(raw_dir)
    _, players = _load_teams_and_players(raw_dir)
    return [{"player_id": pid, **(players.get(pid) or {})} for pid in (team.get("players") or [])]


def team_record(roster_id: Any, raw_dir: Path = RAW_DIR) -> dict:
    """This team's win/loss/tie record, read directly from Sleeper's own
    roster settings (already present in teams.json's "settings" field --
    Sleeper's API computes wins/losses/ties itself from completed
    matchups, so there's no need to recompute it here from matchup
    history)."""
    teams, _ = _load_teams_and_players(raw_dir)
    team = _team_by_roster_id(roster_id, teams)
    if team is None:
        raise RuntimeError(f"No team found with roster_id={roster_id!r} in teams.json.")
    settings = team.get("settings") or {}
    return {
        "roster_id": team.get("roster_id"),
        "team_name": team.get("team_name"),
        "wins": settings.get("wins", 0),
        "losses": settings.get("losses", 0),
        "ties": settings.get("ties", 0),
    }


def team_record_for_owner(owner_display_name: str, raw_dir: Path = RAW_DIR) -> dict | None:
    """team_record() for the team owned by owner_display_name
    (case-insensitive exact match), or None if no team matches."""
    teams, _ = _load_teams_and_players(raw_dir)
    team = _team_by_owner(owner_display_name, teams)
    return team_record(team["roster_id"], raw_dir) if team else None


def my_team_record(raw_dir: Path = RAW_DIR) -> dict:
    """team_record() for the team configured as "mine" (see current_roster())."""
    team = current_roster(raw_dir)
    return team_record(team["roster_id"], raw_dir)


def current_matchup(roster_id: Any, week: int, raw_dir: Path = RAW_DIR) -> dict | None:
    """This roster's opponent for `week`, from that week's locally-
    ingested matchups_week_{week}.json (see src/ingest/sleeper.py --
    `--week N` fetches a specific week). Returns None if that week
    hasn't been ingested locally yet, or if this roster isn't in any
    matchup that week (e.g. a bye)."""
    path = raw_dir / f"matchups_week_{week}.json"
    if not path.exists():
        return None
    sides = json.loads(path.read_text())["data"]
    by_matchup: dict[Any, list[dict]] = {}
    for side in sides:
        by_matchup.setdefault(side.get("matchup_id"), []).append(side)

    teams, _ = _load_teams_and_players(raw_dir)
    for pair in by_matchup.values():
        if not any(str(s.get("roster_id")) == str(roster_id) for s in pair):
            continue
        others = [s for s in pair if str(s.get("roster_id")) != str(roster_id)]
        if not others:
            return {
                "week": week,
                "opponent_roster_id": None,
                "note": "No opponent found in this matchup -- possible bye week.",
            }
        opponent_team = _team_by_roster_id(others[0].get("roster_id"), teams)
        return {
            "week": week,
            "opponent_roster_id": others[0].get("roster_id"),
            "opponent_team_name": opponent_team.get("team_name") if opponent_team else None,
            "opponent_display_name": opponent_team.get("display_name") if opponent_team else None,
        }
    return None


def current_matchup_for_owner(owner_display_name: str, week: int, raw_dir: Path = RAW_DIR) -> dict | None:
    """current_matchup() for the team owned by owner_display_name
    (case-insensitive exact match), or None if no team matches."""
    teams, _ = _load_teams_and_players(raw_dir)
    team = _team_by_owner(owner_display_name, teams)
    return current_matchup(team["roster_id"], week, raw_dir) if team else None


def my_current_matchup(week: int, raw_dir: Path = RAW_DIR) -> dict | None:
    """current_matchup() for the team configured as "mine" (see current_roster())."""
    team = current_roster(raw_dir)
    return current_matchup(team["roster_id"], week, raw_dir)


def all_rostered_players(raw_dir: Path = RAW_DIR) -> list[dict]:
    """Every player (deduped by Sleeper player_id) rostered by ANY team in
    this league, across the whole league -- not just "mine". Used by
    src/reasoning/report.py's waiver_pickups report as the set to subtract
    from the full NFL player pool to find who's actually unrostered."""
    teams, players = _load_teams_and_players(raw_dir)
    seen: dict[str, dict] = {}
    for team in teams:
        for pid in team.get("players") or []:
            if pid not in seen:
                seen[pid] = {"player_id": pid, **(players.get(pid) or {})}
    return list(seen.values())


def all_team_rosters(raw_dir: Path = RAW_DIR) -> list[dict]:
    """Every team's full roster, across the whole league -- the
    league-wide, per-team-grouped counterpart to my_players(). Reads the
    same already-ingested teams.json/players.json all_rostered_players()
    already proves is accessible league-wide; this returns it grouped
    per team (with a per-position count) instead of flattened into one
    league-wide set, since "which teams are deep/shallow at position X"
    needs per-team grouping, not a flat list.

    Used by src/reasoning/recommend.py's get_league_rosters tool for
    roster-COMPOSITION questions only (what a team has) -- this has
    nothing to do with trade value, and callers must not treat it as
    one; see recommend.py's docstring for that boundary.
    """
    teams, players = _load_teams_and_players(raw_dir)
    rosters = []
    for team in teams:
        roster_players = [
            {
                "player_id": pid,
                "name": (players.get(pid) or {}).get("full_name"),
                "position": (players.get(pid) or {}).get("position"),
                "team": (players.get(pid) or {}).get("team"),
            }
            for pid in (team.get("players") or [])
        ]
        counts_by_position: dict[str, int] = {}
        for p in roster_players:
            position = p.get("position") or "UNKNOWN"
            counts_by_position[position] = counts_by_position.get(position, 0) + 1
        rosters.append(
            {
                "roster_id": team.get("roster_id"),
                "team_name": team.get("team_name"),
                "owner_display_name": team.get("display_name"),
                "players": roster_players,
                "counts_by_position": counts_by_position,
            }
        )
    return rosters


def team_roster_for_owner(owner_display_name: str, raw_dir: Path = RAW_DIR) -> dict | None:
    """all_team_rosters()'s entry for the team owned by owner_display_name
    (case-insensitive exact match, same matching rule as
    team_record_for_owner/current_matchup_for_owner), or None if no team
    matches."""
    needle = owner_display_name.lower()
    for roster in all_team_rosters(raw_dir):
        if (roster.get("owner_display_name") or "").lower() == needle:
            return roster
    return None


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
