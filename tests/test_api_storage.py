from __future__ import annotations

import pytest

from src.api.storage import QueryCapExceeded, Storage

USER = {"username": "Rogoel49", "user_id": "112233", "display_name": "Rohan"}
LEAGUES = [
    {"league_id": "1389341490030862336", "name": "Victorious Secret 3.0", "season": "2026", "roster_id": 8},
    {"league_id": "555", "name": "Friends PPR", "season": "2026", "roster_id": None},
]


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path / "test.sqlite3")


def test_schema_is_created_on_open_and_reopen_is_safe(tmp_path):
    path = tmp_path / "db.sqlite3"
    Storage(path)
    Storage(path)  # CREATE TABLE IF NOT EXISTS -- no migration framework, no error
    assert path.exists()


def test_save_and_read_user_leagues(storage):
    storage.save_user_leagues(USER, LEAGUES)
    assert storage.user("rogoel49")["user_id"] == "112233"
    assert storage.user("ROGOEL49")["display_name"] == "Rohan"  # usernames normalize case-insensitively
    leagues = storage.leagues("rogoel49")
    assert {l["league_id"] for l in leagues} == {"1389341490030862336", "555"}
    assert storage.league("rogoel49", "1389341490030862336")["roster_id"] == "8"
    assert storage.league("rogoel49", "555")["roster_id"] is None


def test_save_replaces_the_league_list_rather_than_accumulating(storage):
    storage.save_user_leagues(USER, LEAGUES)
    storage.save_user_leagues(USER, LEAGUES[:1])
    assert [l["league_id"] for l in storage.leagues("rogoel49")] == ["1389341490030862336"]


def test_unknown_user_reads_as_none_and_empty(storage):
    assert storage.user("nobody") is None
    assert storage.leagues("nobody") == []
    assert storage.league("nobody", "1") is None


def test_create_session_requires_a_logged_in_user(storage):
    with pytest.raises(KeyError):
        storage.create_session("nobody")


def test_session_lifecycle_and_active_league(storage):
    storage.save_user_leagues(USER, LEAGUES)
    session = storage.create_session("rogoel49")
    assert session["league_id"] is None and session["roster_id"] is None

    switched = storage.set_active_league(session["session_id"], "1389341490030862336")
    assert switched["league_id"] == "1389341490030862336"
    assert switched["roster_id"] == "8"  # defaults to the stored roster for that user in that league

    explicit = storage.set_active_league(session["session_id"], "555", roster_id=3)
    assert (explicit["league_id"], explicit["roster_id"]) == ("555", "3")
    assert storage.session(session["session_id"])["league_id"] == "555"


def test_create_session_with_league_set_immediately(storage):
    storage.save_user_leagues(USER, LEAGUES)
    session = storage.create_session("rogoel49", league_id="1389341490030862336", roster_id=8)
    assert (session["league_id"], session["roster_id"]) == ("1389341490030862336", "8")


def test_session_cannot_point_at_a_league_the_user_is_not_in(storage):
    storage.save_user_leagues(USER, LEAGUES)
    session = storage.create_session("rogoel49")
    with pytest.raises(KeyError):
        storage.set_active_league(session["session_id"], "not-my-league")


def test_unknown_session(storage):
    assert storage.session("nope") is None
    with pytest.raises(KeyError):
        storage.set_active_league("nope", "555")


def test_daily_query_cap_counts_and_then_refuses(storage):
    assert storage.queries_today("rogoel49", day="2026-09-08") == 0
    assert storage.record_query("rogoel49", cap=2, day="2026-09-08") == 1
    assert storage.record_query("rogoel49", cap=2, day="2026-09-08") == 2
    with pytest.raises(QueryCapExceeded) as excinfo:
        storage.record_query("rogoel49", cap=2, day="2026-09-08")
    assert excinfo.value.used == 2 and excinfo.value.cap == 2
    assert storage.queries_today("rogoel49", day="2026-09-08") == 2  # the refused query was not counted


def test_daily_query_cap_is_per_user_and_per_day(storage):
    storage.record_query("rogoel49", cap=1, day="2026-09-08")
    assert storage.record_query("friend", cap=1, day="2026-09-08") == 1  # someone else's cap is separate
    assert storage.record_query("rogoel49", cap=1, day="2026-09-09") == 1  # a new day resets


def test_query_cap_uses_utc_today_by_default(storage):
    assert storage.record_query("rogoel49", cap=5) == 1
    assert storage.queries_today("rogoel49") == 1
