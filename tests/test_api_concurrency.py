"""Regression test for the Phase 5.3 Feed bug: concurrent report requests
on a fresh server process 500'd for the report types that read Chroma.

Root cause (see src/api/leagues.py's warm_chroma note): chromadb caches a
new per-path System before it has finished starting, with no lock, so
two request threads opening `PersistentClient` on the same league
directory at the same moment can hand the second one a half-initialized
client. It only bites when the *process* has never opened that path
before -- which is exactly the deployed case (league ingested earlier,
server started later) and exactly what a single-request test never
exercises.

To reproduce that deterministically here:
  1. The league's Chroma index is built in a SUBPROCESS, so this test
     process is genuinely cold for that path (the shared `api` fixture in
     test_api_main.py ingests in-process, which is why it never saw this).
  2. chromadb's RustBindingsAPI.start() is wrapped with a short sleep to
     widen the race window from "sometimes" to "always" (without the fix,
     the second thread reliably lands inside it).
  3. The three Feed reports are requested concurrently through
     TestClient (sync endpoints run in a threadpool, real threads).
Without warm_chroma() this fails with two 500s; with it, all three are
200 and the same Chroma path was opened exactly once during warm-up.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from src.api import auth, leagues, main
from src.api.storage import Storage
from tests.test_league import _ALL_INDEX_ROWS, _ALL_SIGNALS, _SEASON, _WEEK, VS30_ID

REPO = Path(__file__).resolve().parents[1]

_SEED_IN_SUBPROCESS = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, {repo!r})
import polars as pl
from src.rag import embed
from tests.test_api_main import _seed_two_team_league
from tests.test_league import _ALL_SIGNALS
raw_dir, persist_dir = Path({raw!r}), Path({chroma!r})
_seed_two_team_league(raw_dir, {league_id!r}, "Victorious Secret 3.0")
embed.embed(embed.build_chunks(raw_dir) + embed.build_signal_chunks(_ALL_SIGNALS), persist_dir=persist_dir)
print("seeded")
"""


@pytest.fixture
def cold_league(tmp_path, monkeypatch):
    """A league whose data + Chroma index exist on disk but have never
    been opened by THIS process."""
    monkeypatch.setattr(leagues, "LEAGUES_DIR", tmp_path / "leagues")
    monkeypatch.delenv("SLEEPER_LEAGUE_ID", raising=False)
    raw_dir, persist_dir = leagues.league_dirs(VS30_ID)
    out = subprocess.run(
        [sys.executable, "-c", _SEED_IN_SUBPROCESS.format(repo=str(REPO), raw=str(raw_dir), chroma=str(persist_dir), league_id=VS30_ID)],
        capture_output=True, text=True, timeout=300, cwd=str(REPO),
    )
    assert out.returncode == 0 and "seeded" in out.stdout, out.stderr[-2000:]
    assert leagues.is_ingested(VS30_ID)
    # Forget any warm-up from earlier tests in this process, so this test is a real cold start.
    monkeypatch.setattr(leagues, "_chroma_warmed", set())

    signals_dir = tmp_path / "signals"
    signals_dir.mkdir()
    pl.DataFrame(_ALL_SIGNALS).write_parquet(signals_dir / f"signals_{_SEASON}_week{_WEEK}.parquet")
    monkeypatch.setattr(main, "SIGNALS_DIR", signals_dir)
    import src.rag.player_index as player_index_module

    monkeypatch.setattr(player_index_module.nflverse, "fetch_players", lambda: pl.DataFrame(_ALL_INDEX_ROWS))
    monkeypatch.setattr(leagues, "ingest_league", lambda *a, **kw: pytest.fail("league is on disk; ingest must not run"))
    monkeypatch.setattr(
        auth,
        "resolve_user_leagues",
        lambda username, season=None: {
            "user": {"user_id": "112233", "username": "rogoel49", "display_name": "Rohan"},
            "leagues": [{"league_id": VS30_ID, "name": "Victorious Secret 3.0", "season": str(_SEASON), "roster_id": 1}],
        },
    )
    storage = Storage(tmp_path / "api.sqlite3")
    main.app.dependency_overrides[main.get_storage] = lambda: storage
    client = TestClient(main.app)
    client.post("/api/leagues", json={"username": "rogoel49"})
    # NB: creating the session goes through ensure_league_data() and therefore warm_chroma().
    # To test the race at the REPORT layer we forget the warm-up again right after, so the
    # three concurrent report requests are the first Chroma openers -- the worst case.
    session_id = client.post("/api/sessions", json={"username": "rogoel49", "league_id": VS30_ID}).json()["session_id"]
    _forget_chroma(persist_dir)
    monkeypatch.setattr(leagues, "_chroma_warmed", set())
    yield client, session_id, persist_dir
    main.app.dependency_overrides.clear()


def _forget_chroma(persist_dir: Path) -> None:
    """Make chromadb treat persist_dir as never-opened in this process
    (what a fresh server process sees). Internal-API poke, test-only."""
    from chromadb.api.shared_system_client import SharedSystemClient

    for key in [k for k in SharedSystemClient._identifier_to_system if str(persist_dir) in k]:
        SharedSystemClient._identifier_to_system.pop(key).stop()


def _widen_the_race(monkeypatch, seconds: float = 0.4):
    """Make chromadb's per-path start() slow, so a second thread reliably
    arrives while the first is still starting -- the exact window the
    production race lives in."""
    from chromadb.api.rust import RustBindingsAPI

    real_start = RustBindingsAPI.start
    calls = []

    def slow_start(self):
        calls.append(threading.get_ident())
        time.sleep(seconds)
        return real_start(self)

    monkeypatch.setattr(RustBindingsAPI, "start", slow_start)
    return calls


def _concurrent_reports(client, session_id):
    results: dict[str, tuple[int, str]] = {}

    def fetch(kind):
        resp = client.get(f"/api/reports/{kind}", params={"session_id": session_id, "season": _SEASON, "as_of_week": _WEEK})
        results[kind] = (resp.status_code, resp.text[:300])

    threads = [threading.Thread(target=fetch, args=(k,)) for k in ("start_sit", "drop", "waiver_pickups")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_concurrent_feed_reports_all_succeed_on_a_cold_process(cold_league, monkeypatch):
    client, session_id, persist_dir = cold_league
    starts = _widen_the_race(monkeypatch)

    results = _concurrent_reports(client, session_id)

    assert {k: v[0] for k, v in results.items()} == {"start_sit": 200, "drop": 200, "waiver_pickups": 200}, results
    assert json.loads(client.get(f"/api/reports/drop", params={"session_id": session_id, "season": _SEASON, "as_of_week": _WEEK}).text)["entries"]
    # The shared System for this path was started exactly once, by whichever thread won the warm-up lock.
    assert len(starts) == 1, f"chromadb start() ran {len(starts)} times -- the warm-up did not serialize the first open"


def test_warm_chroma_is_idempotent_and_serialized(tmp_path, monkeypatch):
    """Direct contract of the fix: N threads warming the same path start
    chromadb once, and later calls are no-ops."""
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    monkeypatch.setattr(leagues, "_chroma_warmed", set())
    starts = _widen_the_race(monkeypatch, seconds=0.2)

    threads = [threading.Thread(target=leagues.warm_chroma, args=(persist_dir,)) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    leagues.warm_chroma(persist_dir)

    assert len(starts) == 1
    assert str(persist_dir) in leagues._chroma_warmed
