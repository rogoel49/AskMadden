import pytest

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


def test_current_roster_resolves_from_env(tmp_path, monkeypatch):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)
    monkeypatch.setenv("MY_ROSTER_ID", "2")

    team = lookup.current_roster(raw_dir)

    assert team["roster_id"] == 2
    assert team["display_name"] == "otheruser"


def test_current_roster_missing_env_raises(tmp_path, monkeypatch):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)
    monkeypatch.delenv("MY_ROSTER_ID", raising=False)

    with pytest.raises(RuntimeError, match="MY_ROSTER_ID is not set"):
        lookup.current_roster(raw_dir)


def test_current_roster_unknown_id_raises(tmp_path, monkeypatch):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)
    monkeypatch.setenv("MY_ROSTER_ID", "999")

    with pytest.raises(RuntimeError, match="No team found"):
        lookup.current_roster(raw_dir)


def test_my_players_by_position_uses_current_roster(tmp_path, monkeypatch):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)
    monkeypatch.setenv("MY_ROSTER_ID", "1")

    qbs = lookup.my_players_by_position("QB", raw_dir)

    assert len(qbs) == 1
    assert qbs[0]["full_name"] == "Patrick Mahomes"


def test_my_players_returns_full_roster_unfiltered(tmp_path, monkeypatch):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)
    monkeypatch.setenv("MY_ROSTER_ID", "1")

    players = lookup.my_players(raw_dir)

    assert {p["full_name"] for p in players} == {"Patrick Mahomes", "Cooper Kupp"}


def test_team_record_reads_wins_losses_ties_from_sleeper_roster_settings(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)

    record = lookup.team_record(1, raw_dir)

    # _seed_raw_dir's team 1 has settings={"wins": 2} -- losses/ties absent,
    # so they should default to 0 rather than KeyError.
    assert record == {"roster_id": 1, "team_name": "Victorious Secret", "wins": 2, "losses": 0, "ties": 0}


def test_team_record_unknown_roster_id_raises(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)

    with pytest.raises(RuntimeError, match="No team found"):
        lookup.team_record(999, raw_dir)


def test_team_record_for_owner_resolves_by_display_name(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)

    record = lookup.team_record_for_owner("otheruser", raw_dir)

    assert record["roster_id"] == 2
    assert record["wins"] == 1


def test_team_record_for_owner_unknown_owner_returns_none(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)

    assert lookup.team_record_for_owner("nobody", raw_dir) is None


def test_my_team_record_uses_current_roster(tmp_path, monkeypatch):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)
    monkeypatch.setenv("MY_ROSTER_ID", "1")

    record = lookup.my_team_record(raw_dir)

    assert record["wins"] == 2


def test_current_matchup_finds_the_opponent(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)

    matchup = lookup.current_matchup(1, week=3, raw_dir=raw_dir)

    assert matchup["opponent_roster_id"] == 2
    assert matchup["opponent_team_name"] == "Team Two"


def test_current_matchup_returns_none_when_week_not_ingested(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)

    # week 99 was never fetched (_seed_raw_dir only writes week 3) -- this
    # must come back None, never crash or fabricate an opponent.
    assert lookup.current_matchup(1, week=99, raw_dir=raw_dir) is None


def test_current_matchup_for_owner_resolves_by_display_name(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)

    matchup = lookup.current_matchup_for_owner("rogoel49", week=3, raw_dir=raw_dir)

    assert matchup["opponent_roster_id"] == 2


def test_my_current_matchup_uses_current_roster(tmp_path, monkeypatch):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)
    monkeypatch.setenv("MY_ROSTER_ID", "1")

    matchup = lookup.my_current_matchup(week=3, raw_dir=raw_dir)

    assert matchup["opponent_roster_id"] == 2


# ---- Phase 3.8: league-wide roster composition ----


def test_all_team_rosters_returns_every_team_grouped_by_position(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)

    rosters = lookup.all_team_rosters(raw_dir)

    assert {r["roster_id"] for r in rosters} == {1, 2}
    team1 = next(r for r in rosters if r["roster_id"] == 1)
    assert team1["owner_display_name"] == "rogoel49"
    assert {p["name"] for p in team1["players"]} == {"Patrick Mahomes", "Cooper Kupp"}
    assert team1["counts_by_position"] == {"QB": 1, "WR": 1}

    team2 = next(r for r in rosters if r["roster_id"] == 2)
    assert team2["counts_by_position"] == {"RB": 1, "QB": 1, "TE": 1}


def test_team_roster_for_owner_resolves_by_display_name(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)

    roster = lookup.team_roster_for_owner("otheruser", raw_dir)

    assert roster["roster_id"] == 2
    assert roster["counts_by_position"]["TE"] == 1


def test_team_roster_for_owner_unknown_owner_returns_none(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)

    assert lookup.team_roster_for_owner("nobody", raw_dir) is None
