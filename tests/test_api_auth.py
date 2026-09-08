"""src/api/auth.py against MOCKED Sleeper responses shaped per Sleeper's
documented API (https://docs.sleeper.com/). Sleeper is blocked in the
sandbox this was built in, so nothing here proves the live API answers
this way -- it proves the module handles the documented shapes (and the
documented "null for unknown user" case) correctly. The live flow needs
one real run on a machine with Sleeper access -- see TODO.md's Phase 5.2
entry.
"""
from __future__ import annotations

import pytest
import requests

from src.api import auth
from src.ingest import sleeper

# Documented shapes.
USER = {"username": "rogoel49", "user_id": "112233", "display_name": "Rohan", "avatar": "abc"}
STATE = {"season": "2026", "week": 1, "display_week": 1}
LEAGUE_A = {
    "league_id": "1389341490030862336", "name": "Victorious Secret 3.0", "season": "2026", "status": "in_season",
    "sport": "nfl", "total_rosters": 12, "scoring_settings": {"rec": 0.5, "pass_td": 4},
    "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN"], "settings": {"num_teams": 12},
}
LEAGUE_B = {"league_id": "555", "name": "Friends PPR", "season": "2026", "status": "pre_draft", "total_rosters": 10,
            "scoring_settings": {"rec": 1.0}}
ROSTERS_A = [
    {"roster_id": 1, "owner_id": "999", "co_owners": None, "players": []},
    {"roster_id": 8, "owner_id": "112233", "co_owners": None, "players": []},
]
ROSTERS_B = [
    {"roster_id": 3, "owner_id": "777", "co_owners": ["112233"], "players": []},  # user is a co-owner here
]


def _fake_get(responses: dict):
    def _get(path: str):
        if path not in responses:
            raise AssertionError(f"unexpected Sleeper call: {path}")
        value = responses[path]
        if isinstance(value, Exception):
            raise value
        return value

    return _get


def _http_404() -> requests.HTTPError:
    resp = requests.Response()
    resp.status_code = 404
    return requests.HTTPError(response=resp)


@pytest.fixture
def sleeper_api(monkeypatch):
    responses = {
        "user/rogoel49": USER,
        "user/ghost": None,  # Sleeper answers an unknown username with a JSON null body
        "user/gone": _http_404(),
        "state/nfl": STATE,
        "user/112233/leagues/nfl/2026": [LEAGUE_A, LEAGUE_B],
        "user/112233/leagues/nfl/2025": [LEAGUE_A],
        "league/1389341490030862336/rosters": ROSTERS_A,
        "league/555/rosters": ROSTERS_B,
    }
    monkeypatch.setattr(sleeper, "_get", _fake_get(responses))
    return responses


def test_resolve_user_returns_the_documented_identity_fields(sleeper_api):
    assert auth.resolve_user("rogoel49") == {"user_id": "112233", "username": "rogoel49", "display_name": "Rohan"}


def test_resolve_user_strips_whitespace(sleeper_api):
    assert auth.resolve_user("  rogoel49 ")["user_id"] == "112233"


@pytest.mark.parametrize("username", ["ghost", "gone"])
def test_resolve_user_unknown_raises_whether_sleeper_answers_null_or_404(sleeper_api, username):
    with pytest.raises(auth.UnknownSleeperUser):
        auth.resolve_user(username)


@pytest.mark.parametrize("username", ["", "   ", None])
def test_resolve_user_requires_a_username(sleeper_api, username):
    with pytest.raises(auth.UnknownSleeperUser):
        auth.resolve_user(username)


def test_list_leagues_defaults_season_to_sleepers_current_nfl_state(sleeper_api):
    leagues = auth.list_leagues("112233")
    assert [l["league_id"] for l in leagues] == ["1389341490030862336", "555"]
    assert leagues[0]["name"] == "Victorious Secret 3.0"
    assert leagues[0]["scoring_settings"] == {"rec": 0.5, "pass_td": 4}  # verbatim, no format label
    assert leagues[1]["scoring_settings"] == {"rec": 1.0}


def test_list_leagues_honors_an_explicit_season(sleeper_api):
    assert [l["league_id"] for l in auth.list_leagues("112233", season=2025)] == ["1389341490030862336"]


def test_roster_id_for_user_matches_owner_id(sleeper_api):
    assert auth.roster_id_for_user("1389341490030862336", "112233") == 8


def test_roster_id_for_user_matches_co_owner(sleeper_api):
    assert auth.roster_id_for_user("555", "112233") == 3


def test_roster_id_for_user_none_when_user_has_no_roster(sleeper_api):
    assert auth.roster_id_for_user("1389341490030862336", "nobody") is None


def test_resolve_user_leagues_is_the_whole_login_flow(sleeper_api):
    resolved = auth.resolve_user_leagues("rogoel49")
    assert resolved["user"]["user_id"] == "112233"
    by_id = {l["league_id"]: l for l in resolved["leagues"]}
    assert by_id["1389341490030862336"]["roster_id"] == 8
    assert by_id["555"]["roster_id"] == 3
