"""src/scheduler/refresh.py: the automated ingest -> signals -> embed cycle.

What runs for real here: the week-resolution logic (the as-of-date rule's
whole point), the cycle's orchestration and error handling, the status
file, src.rag.embed's real Chroma write, and real repeated runs for the
idempotency the scheduler depends on. What's faked, and only at those
boundaries: nflverse's pbp/NGS/schedules pulls and Sleeper's HTTP (both
network; Sleeper is blocked in this sandbox at all).

The as-of-date rule itself is enforced inside
matchup_signals._history (tests/test_matchup_signals.py covers that);
what this file pins is the scheduler never ASKING for a week whose
history isn't finished.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import polars as pl
import pytest

from src.api import leagues
from src.rag import embed
from src.rag.embed import COLLECTION_NAME
from src.scheduler import refresh
from tests.test_matchup_signals import _sample_pbp

REPO_ROOT = Path(__file__).resolve().parents[1]

NOW = datetime(2025, 9, 30, 12, 0, tzinfo=timezone.utc)
# Every fixture game is scheduled well inside COMPLETION_GRACE_DAYS (here:
# comfortably in the future), so completeness can only come from `result` --
# which is what the tests below are about, and keeps them independent of the
# date the suite happens to run on. The grace clause gets its own test with
# a deliberately old gameday.
UPCOMING = "2099-09-28"


def _sched(rows: list[dict]) -> pl.DataFrame:
    """A schedules frame shaped like nflverse's, with only the columns the
    week logic reads."""
    return pl.DataFrame(rows, schema={"season": pl.Int64, "game_type": pl.Utf8, "week": pl.Int64,
                                      "gameday": pl.Utf8, "result": pl.Int64})


def _game(week: int, result: int | None, gameday: str = UPCOMING, game_type: str = "REG") -> dict:
    return {"season": 2025, "game_type": game_type, "week": week, "gameday": gameday, "result": result}


# ---- which week is safe to compute (the as-of-date rule) ----


def test_a_week_counts_as_complete_only_when_every_game_has_a_final_result():
    schedules = _sched([_game(1, 3), _game(1, -7), _game(2, 10), _game(2, None)])
    assert refresh.completed_weeks(schedules, 2025, NOW) == {1}


def test_the_target_week_is_one_past_the_last_completed_week():
    schedules = _sched([_game(1, 3), _game(2, 10), _game(3, None)])
    assert refresh.target_as_of_week(schedules, 2025, NOW) == 3


def test_the_target_week_never_jumps_over_a_week_still_being_played():
    """The leak this guards against: week 4 is mid-play but week 5 somehow
    already carries results, so max(completed) + 1 would be 6 -- and a
    week-6 table's trailing history would silently include a half-played
    week 4. Walking up from week 1 stops at 4 instead."""
    schedules = _sched([_game(1, 3), _game(2, 3), _game(3, 3), _game(4, None), _game(5, 3)])
    assert refresh.completed_weeks(schedules, 2025, NOW) == {1, 2, 3, 5}
    assert refresh.target_as_of_week(schedules, 2025, NOW) == 4


def test_the_target_week_is_one_before_any_game_of_the_season_is_final():
    """Honest rather than clever: week 1 means the table is all-nulls and
    Phase 3.6's prior-season fallback carries the answer."""
    schedules = _sched([_game(1, None), _game(2, None)])
    assert refresh.target_as_of_week(schedules, 2025, NOW) == 1


def test_a_game_that_never_gets_played_stops_pinning_the_target_week_forever():
    """Without the grace clause a single postponed game would freeze the
    target week for the rest of the season and every later refresh would
    silently recompute the same stale week."""
    stuck = _sched([_game(1, 3), _game(1, None, gameday="2025-09-07")])
    assert refresh.target_as_of_week(stuck, 2025, NOW) == 2
    # ...but not while that game is still plausibly upcoming.
    recent = _sched([_game(1, 3), _game(1, None, gameday=UPCOMING)])
    assert refresh.target_as_of_week(recent, 2025, NOW) == 1


def test_postseason_games_are_not_part_of_the_regular_season_week_count():
    schedules = _sched([_game(1, 3), _game(1, 7), _game(1, None, game_type="POST")])
    assert refresh.completed_weeks(schedules, 2025, NOW) == {1}


def test_resolve_season_and_week_uses_nflverse_for_both_and_honours_overrides(monkeypatch):
    schedules = _sched([_game(1, 3), _game(2, None)])
    monkeypatch.setattr(refresh.nflverse, "current_season", lambda: 2025)
    monkeypatch.setattr(refresh.nflverse, "fetch_schedules", lambda season: schedules)

    assert refresh.resolve_season_and_week()[:2] == (2025, 2)
    assert refresh.resolve_season_and_week(as_of_week=1)[:2] == (2025, 1)
    assert refresh.resolve_season_and_week()[3] == 1  # one completed week


# ---- configuration ----


@pytest.mark.parametrize(
    "value,expected",
    [(None, refresh.DEFAULT_INTERVAL_SECONDS), ("", refresh.DEFAULT_INTERVAL_SECONDS),
     ("900", 900), ("1", refresh.MIN_INTERVAL_SECONDS), ("not-a-number", refresh.DEFAULT_INTERVAL_SECONDS)],
)
def test_interval_seconds_reads_the_env_var_and_refuses_a_busy_loop(monkeypatch, value, expected):
    monkeypatch.delenv("ASKMADDEN_REFRESH_INTERVAL_SECONDS", raising=False)
    if value is not None:
        monkeypatch.setenv("ASKMADDEN_REFRESH_INTERVAL_SECONDS", value)
    assert refresh.interval_seconds() == expected


@pytest.mark.parametrize("value,expected", [(None, True), ("", True), ("1", True), ("0", False), ("false", False), ("OFF", False)])
def test_refresh_is_enabled_unless_explicitly_turned_off(monkeypatch, value, expected):
    monkeypatch.delenv("ASKMADDEN_REFRESH_ENABLED", raising=False)
    if value is not None:
        monkeypatch.setenv("ASKMADDEN_REFRESH_ENABLED", value)
    assert refresh.refresh_enabled() is expected


def test_the_background_thread_does_not_start_when_refresh_is_disabled(monkeypatch):
    monkeypatch.setenv("ASKMADDEN_REFRESH_ENABLED", "0")
    assert refresh.start_background_refresh() is None


# ---- a whole cycle ----


# tests/test_matchup_signals.py's real fixture (season 2024, regular-season
# weeks 1-2 plus week-3 plays that must never leak in) so the signals step
# runs its real computation rather than against a shape invented here.
SEASON = 2024
PBP = _sample_pbp()
SCHEDULES = pl.DataFrame(
    [
        {"season": SEASON, "game_type": "REG", "week": week, "gameday": UPCOMING,
         "home_team": home, "away_team": away, "result": (3 if week < 3 else None),
         "spread_line": 3.0, "total_line": 45.0}
        for week in (1, 2, 3)
        for home, away in (("A", "C"), ("B", "D"), ("E", "Z"))
    ]
)


@pytest.fixture
def cycle(tmp_path, monkeypatch):
    """A cycle whose only fakes are the two network boundaries: nflverse's
    pulls and Sleeper's HTTP. Signals computation, the parquet write, the
    Chroma embed and the status file all run for real."""
    signals_dir = tmp_path / "signals"
    status_path = tmp_path / "refresh_status.json"
    monkeypatch.setattr(refresh.nflverse, "current_season", lambda: SEASON)
    monkeypatch.setattr(refresh.nflverse, "fetch_schedules", lambda season: SCHEDULES)
    monkeypatch.setattr(refresh.nflverse, "fetch_pbp", lambda season: PBP)
    monkeypatch.setattr(refresh.ngs, "fetch_ngs", lambda season, stat_type: None)

    monkeypatch.setattr(leagues, "LEAGUES_DIR", tmp_path / "leagues")
    monkeypatch.delenv("SLEEPER_LEAGUE_ID", raising=False)

    sleeper_calls: list[dict] = []

    def fake_sleeper_run(league_id, week=None, out_dir=None):
        sleeper_calls.append({"league_id": league_id, "week": week, "out_dir": out_dir})
        _seed_league_files(out_dir, league_id)
        return {name: out_dir / name for name in sorted(p.name for p in out_dir.glob("*.json"))}

    monkeypatch.setattr(refresh.sleeper, "run", fake_sleeper_run)
    return {"signals_dir": signals_dir, "status_path": status_path, "sleeper_calls": sleeper_calls,
            "tmp_path": tmp_path}


def _seed_league_files(raw_dir: Path, league_id: str, week: int = 3) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    def write(name, data):
        (raw_dir / name).write_text(json.dumps({"fetched_at": "t", "source": "sleeper", "data": data}))

    write("league.json", {"league_id": league_id, "name": f"League {league_id}", "season": "2025",
                          "settings": {"num_teams": 2, "waiver_budget": 100}, "scoring_settings": {"rec": 0.5}})
    write("players.json", {"p1": {"full_name": "Saquon Barkley", "position": "RB", "team": "PHI"}})
    write("teams.json", [{"roster_id": 1, "owner_id": "u1", "display_name": "rogoel49", "team_name": "VS",
                          "players": ["p1"], "starters": ["p1"], "settings": {"wins": 1, "losses": 0, "ties": 0}}])
    write("nfl_state.json", {"season": "2025", "display_week": week, "week": week})
    write(f"matchups_week_{week}.json", [{"roster_id": 1, "matchup_id": 1, "points": 101.0}])


def _ingest_one_league(cycle, league_id: str = "555") -> tuple[Path, Path]:
    """Put a league on disk the way ensure_league_data would, so
    ingested_league_ids() finds it without any Sleeper call."""
    raw_dir, persist_dir = leagues.league_dirs(league_id)
    _seed_league_files(raw_dir, league_id)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, persist_dir


def test_a_cycle_computes_the_shared_signals_table_then_refreshes_every_league(cycle):
    raw_dir, persist_dir = _ingest_one_league(cycle)

    record = refresh.run_cycle(signals_dir=cycle["signals_dir"], status_path=cycle["status_path"])

    assert record["outcome"] == "ok" and record["error"] is None
    # target week is 3: weeks 1-2 are final, week 3 isn't.
    assert (record["season"], record["as_of_week"], record["completed_weeks"]) == (SEASON, 3, 2)
    assert [t["as_of_week"] for t in record["signals"]["tables"]] == [3]
    assert (cycle["signals_dir"] / f"signals_{SEASON}_week3.parquet").exists()

    assert [entry["league_id"] for entry in record["leagues"]] == ["555"]
    assert record["leagues"][0]["sleeper"] == "ok" and record["leagues"][0]["embed"] == "ok"
    assert record["leagues"][0]["chunks"] > 0

    collection = chromadb.PersistentClient(path=str(persist_dir)).get_collection(COLLECTION_NAME)
    ids = collection.get()["ids"]
    assert "league:settings" in ids
    assert any(i.startswith(f"signal:{SEASON}:week3:") for i in ids), "the new signals table must reach the RAG corpus"


def test_the_signals_table_is_computed_once_per_cycle_not_once_per_league(cycle, monkeypatch):
    """CLAUDE.md's key principle: the signals table is league-agnostic, so
    N leagues must not mean N identical nflverse pulls."""
    for league_id in ("555", "666", "777"):
        _ingest_one_league(cycle, league_id)
    pulls = []
    monkeypatch.setattr(refresh.nflverse, "fetch_pbp", lambda season: pulls.append(season) or PBP)

    record = refresh.run_cycle(signals_dir=cycle["signals_dir"], status_path=cycle["status_path"])

    assert len(record["leagues"]) == 3
    assert pulls == [SEASON]


def test_backfill_computes_every_earlier_week_too(cycle):
    _ingest_one_league(cycle)
    record = refresh.run_cycle(signals_dir=cycle["signals_dir"], status_path=cycle["status_path"], backfill=True)
    assert [t["as_of_week"] for t in record["signals"]["tables"]] == [1, 2, 3]
    assert sorted(p.name for p in cycle["signals_dir"].glob("*.parquet")) == [
        f"signals_{SEASON}_week{week}.parquet" for week in (1, 2, 3)
    ]


def test_the_league_pull_leaves_the_week_to_sleeper_not_to_the_signals_as_of_week(cycle):
    """A league's live matchup week is Sleeper's own question; the signals
    as-of-week is nflverse's. Passing the latter into sleeper.run() would
    fetch the wrong matchup whenever the two disagree."""
    _ingest_one_league(cycle)
    refresh.run_cycle(signals_dir=cycle["signals_dir"], status_path=cycle["status_path"])
    assert cycle["sleeper_calls"][0]["week"] is None


def test_a_league_that_is_registered_but_never_opened_is_not_refreshed(cycle):
    """ingested_league_ids() reads disk, not the SQLite leagues table: a
    league nobody ever selected has no local data to keep current."""
    record = refresh.run_cycle(signals_dir=cycle["signals_dir"], status_path=cycle["status_path"])
    assert record["leagues"] == []
    assert record["outcome"] == "ok", "no leagues yet is not a failure"
    assert cycle["sleeper_calls"] == []


def test_one_leagues_failure_does_not_stop_the_others(cycle, monkeypatch):
    _ingest_one_league(cycle, "555")
    _ingest_one_league(cycle, "666")
    real_run = refresh.sleeper.run

    def flaky(league_id, week=None, out_dir=None):
        if league_id == "555":
            raise RuntimeError("Sleeper says no")
        return real_run(league_id, week=week, out_dir=out_dir)

    monkeypatch.setattr(refresh.sleeper, "run", flaky)

    record = refresh.run_cycle(signals_dir=cycle["signals_dir"], status_path=cycle["status_path"])

    assert record["outcome"] == "partial"
    by_id = {entry["league_id"]: entry for entry in record["leagues"]}
    assert by_id["555"]["sleeper"] == "error" and "Sleeper says no" in by_id["555"]["error"]
    assert by_id["555"]["embed"] == "skipped", "a failed pull must not be embedded as if it had succeeded"
    assert by_id["666"]["sleeper"] == "ok" and by_id["666"]["embed"] == "ok"


def test_a_cycle_that_cannot_reach_nflverse_records_the_failure_instead_of_raising(cycle, monkeypatch):
    """The loop has to survive a transient outage, so run_cycle never
    raises -- and it must not go on to refresh leagues against a signals
    table it failed to compute."""
    _ingest_one_league(cycle)
    monkeypatch.setattr(refresh.nflverse, "fetch_pbp", lambda season: (_ for _ in ()).throw(OSError("nflverse down")))

    record = refresh.run_cycle(signals_dir=cycle["signals_dir"], status_path=cycle["status_path"])

    assert record["outcome"] == "error"
    assert "nflverse down" in record["error"]
    assert record["leagues"] == []
    assert cycle["sleeper_calls"] == []


# ---- idempotency: a scheduler calls all of this repeatedly ----


def _signal_rows(path: Path) -> list[dict]:
    """A signals table's content, order-independent. Deliberately not a
    bytes comparison of the parquet: polars' group_by/join do not promise
    a stable row order, so two runs over identical input can write
    different byte streams holding the same rows. Idempotency here means
    the same content, which is all anything downstream reads (every
    consumer keys by player_id)."""
    return sorted(pl.read_parquet(path).to_dicts(), key=lambda row: str(row["player_id"]))


def test_running_a_cycle_twice_changes_nothing(cycle):
    """The property the whole scheduler rests on. Every step overwrites
    rather than appends, so a second identical cycle must leave the
    signals table's content unchanged, the Sleeper JSON unchanged, and the
    Chroma collection the same size with the same ids -- no duplicates,
    no growth."""
    raw_dir, persist_dir = _ingest_one_league(cycle)

    first = refresh.run_cycle(signals_dir=cycle["signals_dir"], status_path=cycle["status_path"])
    signals_path = cycle["signals_dir"] / f"signals_{SEASON}_week3.parquet"
    signals_rows = _signal_rows(signals_path)
    raw_bytes = {p.name: p.read_bytes() for p in sorted(raw_dir.glob("*.json"))}
    collection = chromadb.PersistentClient(path=str(persist_dir)).get_collection(COLLECTION_NAME)
    ids_before = sorted(collection.get()["ids"])

    second = refresh.run_cycle(signals_dir=cycle["signals_dir"], status_path=cycle["status_path"])

    assert second["outcome"] == first["outcome"] == "ok"
    assert _signal_rows(signals_path) == signals_rows
    assert {p.name: p.read_bytes() for p in sorted(raw_dir.glob("*.json"))} == raw_bytes
    after = chromadb.PersistentClient(path=str(persist_dir)).get_collection(COLLECTION_NAME)
    assert sorted(after.get()["ids"]) == ids_before
    assert after.count() == first["leagues"][0]["chunks"] == second["leagues"][0]["chunks"]


def test_repeated_embeds_of_the_same_data_do_not_accumulate_chunks(tmp_path):
    """embed.embed()'s delete-all-then-add contract, asserted directly --
    the scheduler calls it on every cycle forever."""
    raw_dir, persist_dir = tmp_path / "sleeper", tmp_path / "chroma"
    _seed_league_files(raw_dir, "555")
    counts = [embed.embed(persist_dir=persist_dir, raw_dir=raw_dir, signals_dir=None).count() for _ in range(3)]
    assert counts[0] > 0 and len(set(counts)) == 1


# ---- status visibility ----


def test_the_status_file_records_the_run_and_counts_consecutive_failures(cycle, monkeypatch):
    _ingest_one_league(cycle)
    status_path = cycle["status_path"]

    refresh.run_cycle(signals_dir=cycle["signals_dir"], status_path=status_path)
    status = json.loads(status_path.read_text())
    assert status["last_run"]["outcome"] == "ok"
    assert status["consecutive_failures"] == 0
    assert status["interval_seconds"] == refresh.interval_seconds()
    assert status["next_run_after"] > status["last_run"]["finished_at"]

    monkeypatch.setattr(refresh.nflverse, "fetch_pbp", lambda season: (_ for _ in ()).throw(OSError("down")))
    for expected in (1, 2, 3):
        refresh.run_cycle(signals_dir=cycle["signals_dir"], status_path=status_path)
        assert json.loads(status_path.read_text())["consecutive_failures"] == expected

    monkeypatch.setattr(refresh.nflverse, "fetch_pbp", lambda season: PBP)
    refresh.run_cycle(signals_dir=cycle["signals_dir"], status_path=status_path)
    assert json.loads(status_path.read_text())["consecutive_failures"] == 0


def test_the_status_file_is_replaced_atomically_and_leaves_no_temp_file(cycle):
    _ingest_one_league(cycle)
    refresh.run_cycle(signals_dir=cycle["signals_dir"], status_path=cycle["status_path"])
    assert list(cycle["tmp_path"].glob("refresh_status.json*")) == [cycle["status_path"]]


def test_reading_a_missing_or_corrupt_status_file_is_not_an_error(tmp_path):
    assert refresh.read_status(tmp_path / "nope.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert refresh.read_status(broken) is None


def test_the_cli_reports_success_and_failure_through_its_exit_code(cycle, monkeypatch, capsys):
    _ingest_one_league(cycle)
    argv = ["--once", "--status-path", str(cycle["status_path"])]
    monkeypatch.setattr(refresh, "SIGNALS_DIR", cycle["signals_dir"])
    monkeypatch.setattr(refresh.matchup_signals, "PROCESSED_DIR", cycle["signals_dir"])

    assert refresh.main(argv) == 0
    assert refresh.main(["--status", "--status-path", str(cycle["status_path"])]) == 0
    assert "outcome:    ok" in capsys.readouterr().out

    monkeypatch.setattr(refresh.sleeper, "run", lambda **kw: (_ for _ in ()).throw(RuntimeError("nope")))
    assert refresh.main(argv) == 1, "a cron job has to be able to tell that a cycle failed"
    assert refresh.main(["--status", "--status-path", str(cycle["status_path"])]) == 1


def test_the_cli_always_writes_a_status_file_even_without_status_path(cycle, monkeypatch):
    """run_cycle reads an explicit status_path=None as "don't write one",
    so --once must not pass argparse's default straight through -- a cron
    job that never wrote a status file would be invisible."""
    _ingest_one_league(cycle)
    monkeypatch.setattr(refresh, "SIGNALS_DIR", cycle["signals_dir"])
    monkeypatch.setattr(refresh, "STATUS_PATH", cycle["status_path"])

    assert refresh.main(["--once"]) == 0
    assert refresh.read_status(cycle["status_path"])["last_run"]["outcome"] == "ok"


def test_run_cycle_writes_no_status_file_when_told_not_to(cycle):
    _ingest_one_league(cycle)
    refresh.run_cycle(signals_dir=cycle["signals_dir"], status_path=None)
    assert not cycle["status_path"].exists()


def test_status_exits_nonzero_when_no_cycle_has_ever_run(tmp_path, capsys):
    assert refresh.main(["--status", "--status-path", str(tmp_path / "none.json")]) == 1
    assert "no refresh has run yet" in capsys.readouterr().out


# ---- which leagues get refreshed ----


def test_ingested_league_ids_finds_the_flat_league_and_every_per_league_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(leagues, "LEAGUES_DIR", tmp_path / "leagues")
    monkeypatch.setattr(leagues, "RAW_DIR", tmp_path / "flat" / "sleeper")
    monkeypatch.setattr(leagues, "CHROMA_DIR", tmp_path / "flat" / "chroma")
    monkeypatch.setenv("SLEEPER_LEAGUE_ID", "own-league")

    assert leagues.ingested_league_ids() == []

    _seed_league_files(tmp_path / "flat" / "sleeper", "own-league")
    (tmp_path / "flat" / "chroma").mkdir(parents=True)
    for league_id in ("666", "555"):
        _seed_league_files(tmp_path / "leagues" / league_id / "sleeper", league_id)
        (tmp_path / "leagues" / league_id / "chroma").mkdir(parents=True)
    # a directory that exists but was never actually ingested
    (tmp_path / "leagues" / "999" / "sleeper").mkdir(parents=True)

    assert leagues.ingested_league_ids() == ["own-league", "555", "666"]


# ---- the API server noticing a refreshed index (the Phase 5.7 freshness fix) ----


def test_warming_the_same_unchanged_index_twice_reuses_the_cached_client(tmp_path):
    raw_dir, persist_dir = tmp_path / "sleeper", tmp_path / "chroma"
    _seed_league_files(raw_dir, "555")
    embed.embed(persist_dir=persist_dir, raw_dir=raw_dir, signals_dir=None)

    leagues.warm_chroma(persist_dir)
    first = chromadb.PersistentClient(path=str(persist_dir))._system
    leagues.warm_chroma(persist_dir)
    assert chromadb.PersistentClient(path=str(persist_dir))._system is first


def test_warming_a_brand_new_directory_twice_does_not_rebuild_the_client(tmp_path):
    """Opening a PersistentClient creates (and so timestamps) the index's
    SQLite file, so a stamp captured BEFORE the open is stale immediately
    and every later request would rebuild the client. warm_chroma records
    the stamp after opening; this pins that."""
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()

    leagues.warm_chroma(persist_dir)
    first = chromadb.PersistentClient(path=str(persist_dir))._system
    leagues.warm_chroma(persist_dir)
    leagues.warm_chroma(persist_dir)

    assert chromadb.PersistentClient(path=str(persist_dir))._system is first


def test_a_rewritten_index_makes_the_next_warm_up_build_a_fresh_client(tmp_path, monkeypatch):
    """The mechanism behind the fix: chromadb answers collection.query()
    from a per-process in-memory vector index, so a server that warmed a
    path before a refresh keeps serving the old index. warm_chroma() (which
    ensure_league_data calls on every request) compares the on-disk stamp
    and forgets the cached System when it changed."""
    raw_dir, persist_dir = tmp_path / "sleeper", tmp_path / "chroma"
    _seed_league_files(raw_dir, "555")
    embed.embed(persist_dir=persist_dir, raw_dir=raw_dir, signals_dir=None)

    leagues.warm_chroma(persist_dir)
    before = chromadb.PersistentClient(path=str(persist_dir))._system

    # what a refresh does to this league's index
    monkeypatch.setattr(leagues, "_index_stamp", lambda _dir: "changed")
    leagues.warm_chroma(persist_dir)

    assert chromadb.PersistentClient(path=str(persist_dir))._system is not before


def test_invalidate_chroma_forces_the_next_warm_up_to_rebuild(tmp_path):
    raw_dir, persist_dir = tmp_path / "sleeper", tmp_path / "chroma"
    _seed_league_files(raw_dir, "555")
    embed.embed(persist_dir=persist_dir, raw_dir=raw_dir, signals_dir=None)

    leagues.warm_chroma(persist_dir)
    before = chromadb.PersistentClient(path=str(persist_dir))._system
    leagues.invalidate_chroma(persist_dir)
    leagues.warm_chroma(persist_dir)

    assert chromadb.PersistentClient(path=str(persist_dir))._system is not before


def test_invalidating_one_league_leaves_another_warm_client_alone(tmp_path):
    """Deliberately per-path: chromadb's public clear_system_cache() drops
    every path, which would let another league's request thread re-create
    its System outside warm_chroma's lock -- the PR #24 start race."""
    systems = {}
    for league_id in ("555", "666"):
        raw_dir = tmp_path / league_id / "sleeper"
        persist_dir = tmp_path / league_id / "chroma"
        _seed_league_files(raw_dir, league_id)
        embed.embed(persist_dir=persist_dir, raw_dir=raw_dir, signals_dir=None)
        leagues.warm_chroma(persist_dir)
        systems[league_id] = (persist_dir, chromadb.PersistentClient(path=str(persist_dir))._system)

    leagues.invalidate_chroma(systems["555"][0])

    assert chromadb.PersistentClient(path=str(systems["666"][0]))._system is systems["666"][1]


SEMANTIC_PROBE = """
import json, sys
from pathlib import Path
from src.api import leagues
from src.rag import embed, retrieve

raw_dir, persist_dir, week = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
leagues.warm_chroma(persist_dir)                      # what /api/sessions does, once
before = [h["id"] for h in retrieve.query("week 8 matchup result", n_results=5, persist_dir=persist_dir)]

# what refresh_league() does: rewrite the league data, re-embed, nudge the server
for old in raw_dir.glob("matchups_week_*.json"):
    old.unlink()
(raw_dir / "matchups_week_8.json").write_text(json.dumps(
    {"fetched_at": "t", "source": "sleeper", "data": [{"roster_id": 1, "matchup_id": 1, "points": 108.0}]}))
embed.embed(persist_dir=persist_dir, raw_dir=raw_dir, signals_dir=None)
leagues.invalidate_chroma(persist_dir)

leagues.warm_chroma(persist_dir)                      # the next request's warm-up
hits = retrieve.query("week 8 matchup result", n_results=5, persist_dir=persist_dir)
print(json.dumps({"before": before, "after": [h["id"] for h in hits],
                  "null_documents": [h["id"] for h in hits if h["text"] is None]}))
"""


def test_a_warmed_process_sees_a_refreshed_index_through_retrieve_query(tmp_path):
    """End to end, in one long-lived process the way the server is: warm
    the index, refresh it, and the semantic path must return the NEW chunk
    and no phantom (text=None) hit for the removed one. Runs in a
    subprocess because this test process has already warmed other Chroma
    paths, and a fresh interpreter is what makes "was it warmed first"
    unambiguous."""
    raw_dir, persist_dir = tmp_path / "sleeper", tmp_path / "chroma"
    _seed_league_files(raw_dir, "555", week=3)
    embed.embed(persist_dir=persist_dir, raw_dir=raw_dir, signals_dir=None)

    proc = subprocess.run(
        [sys.executable, "-c", SEMANTIC_PROBE, str(raw_dir), str(persist_dir), "8"],
        capture_output=True, text=True, cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    assert "matchup:week3:1" in result["before"]
    assert "matchup:week8:1" in result["after"], "the refreshed chunk must be reachable without a restart"
    assert "matchup:week3:1" not in result["after"]
    assert result["null_documents"] == [], (
        "a phantom hit from the stale vector index is what made search_league_info crash"
    )
