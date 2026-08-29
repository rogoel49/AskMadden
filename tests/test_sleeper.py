import json
from unittest.mock import patch

from src.ingest import sleeper


FAKE_STATE = {"week": 3, "display_week": 3, "season": "2026"}
FAKE_LEAGUE = {"league_id": "123", "name": "Victorious Secret 3.0"}
FAKE_ROSTERS = [{"roster_id": 1, "owner_id": "u1"}]
FAKE_USERS = [{"user_id": "u1", "display_name": "rogoel49"}]
FAKE_MATCHUPS = [{"roster_id": 1, "matchup_id": 1, "points": 0}]
FAKE_PLAYERS = {"4046": {"full_name": "Patrick Mahomes"}}


def fake_get(path: str):
    return {
        "state/nfl": FAKE_STATE,
        "league/123/rosters": FAKE_ROSTERS,
        "league/123/users": FAKE_USERS,
        "league/123/matchups/3": FAKE_MATCHUPS,
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
        "matchups_week_3.json",
        "players.json",
    }
    league_payload = json.loads((out_dir / "league.json").read_text())
    assert league_payload["source"] == "sleeper"
    assert league_payload["data"] == FAKE_LEAGUE

    matchups_payload = json.loads((out_dir / "matchups_week_3.json").read_text())
    assert matchups_payload["data"] == FAKE_MATCHUPS


def test_fetch_players_uses_cache_on_second_call(tmp_path):
    cache_path = tmp_path / "players_cache.json"
    with patch.object(sleeper, "_get", return_value=FAKE_PLAYERS) as mock_get:
        first = sleeper.fetch_players(cache_path)
        second = sleeper.fetch_players(cache_path)

    assert first == FAKE_PLAYERS
    assert second == FAKE_PLAYERS
    mock_get.assert_called_once_with("players/nfl")
