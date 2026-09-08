"""Phase 5.2 "auth": Sleeper username -> user_id -> that user's leagues.

No password, no OAuth, per PROJECT_SPEC.md's Phase 5: Sleeper's API is
public and read-only, so knowing a username is all it takes to list
someone's leagues and find their roster in each. This module is the
only place the API layer talks to Sleeper for *user* data; per-league
data (rosters, matchups, the player pool) still goes through
src/ingest/sleeper.py's existing run(), unchanged.

Endpoints used, all documented at https://docs.sleeper.com/ :
  GET /v1/user/<username>                         -> user object
      {"username", "user_id", "display_name", "avatar", ...}
      (Sleeper returns a JSON null body for an unknown username)
  GET /v1/user/<user_id>/leagues/nfl/<season>     -> list of league objects
      each shaped like GET /v1/league/<league_id>: {"league_id", "name",
      "season", "status", "sport", "total_rosters", "settings",
      "scoring_settings", "roster_positions", ...}
  GET /v1/league/<league_id>/rosters              -> list of rosters
      each {"roster_id", "owner_id", "co_owners", "players", ...}
      (reused via src/ingest/sleeper.py's fetch_rosters)
  GET /v1/state/nfl                               -> {"season", "week", ...}
      (reused via fetch_nfl_state, to default `season`)

**Validation status -- read before trusting this in production.**
Sleeper's API is blocked in the sandbox every phase of this project has
been built in (a proxy policy denial, confirmed again in Phase 5.2), so
this module was written against Sleeper's documented shapes above and
tested only against mocked responses of those shapes (tests/
test_api_auth.py). The response fields it reads are the documented,
stable ones (user_id, league_id, name, season, roster_id, owner_id,
co_owners), not anything inferred -- but the end-to-end username ->
leagues -> roster_id flow still needs one real run on a machine with
Sleeper access before it's considered validated. See TODO.md's Phase
5.2 entry for the exact command.
"""
from __future__ import annotations

from typing import Any

import requests

from src.ingest import sleeper


class UnknownSleeperUser(LookupError):
    """No Sleeper account matches the given username."""


def resolve_user(username: str) -> dict:
    """GET /v1/user/<username> -> {"user_id", "username", "display_name"}.
    Raises UnknownSleeperUser if Sleeper has no such account."""
    username = (username or "").strip()
    if not username:
        raise UnknownSleeperUser("a Sleeper username is required")
    try:
        data = sleeper._get(f"user/{username}")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            raise UnknownSleeperUser(f"No Sleeper user named {username!r}.") from e
        raise
    if not data or not data.get("user_id"):
        # Sleeper answers an unknown username with a JSON null body.
        raise UnknownSleeperUser(f"No Sleeper user named {username!r}.")
    return {
        "user_id": str(data["user_id"]),
        "username": data.get("username") or username,
        "display_name": data.get("display_name") or data.get("username") or username,
    }


def list_leagues(user_id: str, season: str | int | None = None) -> list[dict]:
    """GET /v1/user/<user_id>/leagues/nfl/<season> -> one summary dict per
    league. `season` defaults to Sleeper's current NFL season (GET
    /v1/state/nfl), the same source src/ingest/sleeper.py already uses
    for the current week."""
    if season is None:
        season = sleeper.fetch_nfl_state().get("season")
    leagues = sleeper._get(f"user/{user_id}/leagues/nfl/{season}") or []
    return [_league_summary(league) for league in leagues if league.get("league_id")]


def _league_summary(league: dict) -> dict:
    return {
        "league_id": str(league["league_id"]),
        "name": league.get("name"),
        "season": str(league.get("season")) if league.get("season") is not None else None,
        "status": league.get("status"),
        "total_rosters": league.get("total_rosters"),
        # Passed through verbatim -- the same real per-league settings
        # Phase 5.1's load_league() reads from the ingested league.json.
        "scoring_settings": dict(league.get("scoring_settings") or {}),
    }


def roster_id_for_user(league_id: str, user_id: str, rosters: list[dict] | None = None) -> int | str | None:
    """Which roster in league_id belongs to user_id -- the roster whose
    owner_id is the user, or that lists the user as a co-owner. None if
    the user has no roster there (e.g. a commissioner-only account).
    `rosters` may be supplied to avoid a second fetch."""
    if rosters is None:
        rosters = sleeper.fetch_rosters(league_id)
    for roster in rosters:
        owners: list[Any] = [roster.get("owner_id")] + list(roster.get("co_owners") or [])
        if any(str(owner) == str(user_id) for owner in owners if owner is not None):
            return roster.get("roster_id")
    return None


def resolve_user_leagues(username: str, season: str | int | None = None) -> dict:
    """The whole login flow in one call: username -> user -> leagues, each
    with the user's own roster_id in it. Returns {"user": {...},
    "leagues": [{..._league_summary fields..., "roster_id"}, ...]}."""
    user = resolve_user(username)
    leagues = list_leagues(user["user_id"], season=season)
    for league in leagues:
        league["roster_id"] = roster_id_for_user(league["league_id"], user["user_id"])
    return {"user": user, "leagues": leagues}
