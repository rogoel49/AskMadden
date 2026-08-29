"""Sleeper ingest: pulls league, rosters, users, matchups, and the player
pool for the Ask Madden league and writes them to data/raw/sleeper/ as JSON.

Sleeper's API is public and unauthenticated. See https://docs.sleeper.com/.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

API_BASE = "https://api.sleeper.app/v1"
DEFAULT_LEAGUE_ID = "1389341490030862336"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "sleeper"

# Sleeper asks that the full player pool (~5MB) be fetched at most once a day.
PLAYERS_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60


def _get(path: str) -> Any:
    resp = requests.get(f"{API_BASE}/{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_nfl_state() -> dict:
    return _get("state/nfl")


def fetch_league(league_id: str) -> dict:
    return _get(f"league/{league_id}")


def fetch_rosters(league_id: str) -> list[dict]:
    return _get(f"league/{league_id}/rosters")


def fetch_users(league_id: str) -> list[dict]:
    return _get(f"league/{league_id}/users")


def fetch_matchups(league_id: str, week: int) -> list[dict]:
    return _get(f"league/{league_id}/matchups/{week}")


def fetch_transactions(league_id: str, week: int) -> list[dict]:
    """Trades, waiver claims, and free-agent moves processed in a given
    week. Each item's "type" field is "trade", "waiver", or "free_agent"."""
    return _get(f"league/{league_id}/transactions/{week}")


def build_teams(rosters: list[dict], users: list[dict]) -> list[dict]:
    """Join rosters with users by owner_id into one record per team.

    Sleeper's league-level endpoints already return every roster and every
    user for the whole league in a single call each — there's no separate
    per-team or per-user ID needed to look up a specific team's roster or
    owner. This just merges the two responses client-side for convenience.
    """
    users_by_id = {user["user_id"]: user for user in users}
    teams = []
    for roster in rosters:
        owner = users_by_id.get(roster.get("owner_id"), {})
        teams.append(
            {
                "roster_id": roster.get("roster_id"),
                "owner_id": roster.get("owner_id"),
                "display_name": owner.get("display_name"),
                "team_name": (owner.get("metadata") or {}).get("team_name") or owner.get("display_name"),
                "players": roster.get("players"),
                "starters": roster.get("starters"),
                "settings": roster.get("settings"),
            }
        )
    return teams


def fetch_players(cache_path: Path, force_refresh: bool = False) -> dict:
    """Fetch the full NFL player pool, using a local cache since Sleeper
    asks clients not to hit this endpoint more than once per day. The
    cache stores the raw API response, not the wrapped output format."""
    if not force_refresh and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < PLAYERS_CACHE_MAX_AGE_SECONDS:
            return json.loads(cache_path.read_text())
    players = _get("players/nfl")
    _save_json(players, cache_path)
    return players


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def run(league_id: str = DEFAULT_LEAGUE_ID, week: int | None = None, out_dir: Path = RAW_DIR) -> dict[str, Path]:
    """Fetch league/rosters/users/matchups/players and write each to
    out_dir. Returns a dict of logical name -> file path written."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    written: dict[str, Path] = {}

    state = fetch_nfl_state()
    if week is None:
        week = int(state.get("display_week") or state.get("week") or 1)

    league = fetch_league(league_id)
    rosters = fetch_rosters(league_id)
    users = fetch_users(league_id)
    matchups = fetch_matchups(league_id, week)
    transactions = fetch_transactions(league_id, week)
    teams = build_teams(rosters, users)

    players_cache_path = out_dir / "players_cache.json"
    players = fetch_players(players_cache_path)

    payloads = {
        "nfl_state.json": state,
        "league.json": league,
        "rosters.json": rosters,
        "users.json": users,
        "teams.json": teams,
        f"matchups_week_{week}.json": matchups,
        f"transactions_week_{week}.json": transactions,
        "players.json": players,
    }
    for filename, data in payloads.items():
        path = out_dir / filename
        _save_json({"fetched_at": fetched_at, "source": "sleeper", "data": data}, path)
        written[filename] = path

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull Sleeper league data to data/raw/sleeper/")
    parser.add_argument("--league-id", default=os.environ.get("SLEEPER_LEAGUE_ID", DEFAULT_LEAGUE_ID))
    parser.add_argument("--week", type=int, default=None, help="defaults to the current NFL week")
    args = parser.parse_args()

    written = run(league_id=args.league_id, week=args.week)
    for name, path in written.items():
        print(f"wrote {name} -> {path}")


if __name__ == "__main__":
    main()
