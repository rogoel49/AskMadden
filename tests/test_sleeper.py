"""Unit tests for src/ingest/sleeper.py.

These tests never touch the network. `fake_get` stands in for
`sleeper._get` (the one function that calls `requests.get`), keyed by the
URL path Sleeper would receive, so `run()` and friends exercise their real
logic (defaulting the week, joining rosters+users, writing/wrapping JSON,
caching) against known inputs instead of live API responses. The FAKE_*
dicts are trimmed fixtures — only the fields this code actually reads —
not full copies of Sleeper's real payloads, which carry many more fields
(see a real `league/{id}` response for comparison). A live end-to-end
check against the real API lives in .github/workflows/ci.yml, which runs
on GitHub's runners where outbound network isn't restricted.
"""
import json
from unittest.mock import patch

from src.ingest import sleeper


FAKE_STATE = {"week": 3, "display_week": 3, "season": "2026"}
FAKE_LEAGUE = {"league_id": "123", "name": "Victorious Secret 3.0"}
FAKE_ROSTERS = [{"roster_id": 1, "owner_id": "u1", "players": ["4046"], "starters": ["4046"], "settings": {"wins": 2}}]
FAKE_USERS = [{"user_id": "u1", "display_name": "rogoel49", "metadata": {"team_name": "Victorious Secret"}}]
FAKE_MATCHUPS = [{"roster_id": 1, "matchup_id": 1, "points": 0}]
FAKE_TRANSACTIONS = [{"type": "waiver", "roster_ids": [1], "adds": {"4046": 1}}]
FAKE_PLAYERS = {"4046": {"full_name": "Patrick Mahomes"}}


def fake_get(path: str):
    return {
        "state/nfl": FAKE_STATE,
        "league/123/rosters": FAKE_ROSTERS,
        "league/123/users": FAKE_USERS,
        "league/123/matchups/3": FAKE_MATCHUPS,
        "league/123/transactions/3": FAKE_TRANSACTIONS,
        "players/nfl": FAKE_PLAYERS,
    }.get(path) or FAKE_LEAGUE


def test_run_writes_expected_files(tmp_path):
    out_dir = tmp_path / "sleeper"
    with patch.object(sleeper, "_get", side_effect=fake_get):
        written = sleeper.run(league_id="123", out_dir=out_dir)

    assert set(written) == {
        "nfl_state.json",
        "league.json",
        "rosters.json",
        "users.json",
        "teams.json",
        "matchups_week_3.json",
        "transactions_week_3.json",
        "players.json",
    }
    league_payload = json.loads((out_dir / "league.json").read_text())
    assert league_payload["source"] == "sleeper"
    assert league_payload["data"] == FAKE_LEAGUE

    matchups_payload = json.loads((out_dir / "matchups_week_3.json").read_text())
    assert matchups_payload["data"] == FAKE_MATCHUPS

    transactions_payload = json.loads((out_dir / "transactions_week_3.json").read_text())
    assert transactions_payload["data"] == FAKE_TRANSACTIONS

    teams_payload = json.loads((out_dir / "teams.json").read_text())
    assert teams_payload["data"] == sleeper.build_teams(FAKE_ROSTERS, FAKE_USERS)


def test_build_teams_joins_roster_and_owner_by_id():
    teams = sleeper.build_teams(FAKE_ROSTERS, FAKE_USERS)

    assert teams == [
        {
            "roster_id": 1,
            "owner_id": "u1",
            "display_name": "rogoel49",
            "team_name": "Victorious Secret",
            "players": ["4046"],
            "starters": ["4046"],
            "settings": {"wins": 2},
        }
    ]


def test_build_teams_falls_back_to_display_name_without_team_name():
    users = [{"user_id": "u1", "display_name": "rogoel49"}]

    teams = sleeper.build_teams(FAKE_ROSTERS, users)

    assert teams[0]["team_name"] == "rogoel49"


def test_fetch_players_uses_cache_on_second_call(tmp_path):
    cache_path = tmp_path / "players_cache.json"
    with patch.object(sleeper, "_get", return_value=FAKE_PLAYERS) as mock_get:
        first = sleeper.fetch_players(cache_path)
        second = sleeper.fetch_players(cache_path)

    assert first == FAKE_PLAYERS
    assert second == FAKE_PLAYERS
    mock_get.assert_called_once_with("players/nfl")
