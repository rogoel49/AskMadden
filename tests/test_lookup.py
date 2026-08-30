from src.rag import lookup
from tests.test_embed import _seed_raw_dir


def test_find_team_by_player(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)

    team = lookup.find_team_by_player("Patrick Mahomes", raw_dir)

    assert team is not None
    assert team["roster_id"] == 1
    assert team["display_name"] == "rogoel49"


def test_find_team_by_player_no_match(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)

    assert lookup.find_team_by_player("Tom Brady", raw_dir) is None


def test_players_by_position_for_owner(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)

    qbs = lookup.players_by_position_for_owner("rogoel49", "QB", raw_dir)

    assert len(qbs) == 1
    assert qbs[0]["full_name"] == "Patrick Mahomes"


def test_teams_by_nfl_team_count_ranks_descending(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)

    ranked = lookup.teams_by_nfl_team_count("SF", raw_dir)

    # team 2's whole roster (McCaffrey, Purdy, Kittle) is SF; team 1 has none
    assert [team["roster_id"] for team, _count in ranked] == [2]
    assert ranked[0][1] == 3
