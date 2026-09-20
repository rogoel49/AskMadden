"""The request-path signals resync (src/api/leagues.py) must stand down
while the same collection -- or the whole scheduler cycle -- is already
being rebuilt, instead of running a second, concurrent rebuild on the
user's clock. Found 2026-09-20: a league select that took over a minute
and a chat answer missing two QBs' signals, both during the 6-hourly
cycle's re-embed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.api import leagues
from src.rag import embed
from src.scheduler import refresh


@pytest.fixture
def stale_league(tmp_path, monkeypatch):
    """A league whose stamp says the signals table has changed since its
    collection was built -- the one state in which the resync would
    normally rebuild."""
    signals_dir = tmp_path / "signals"
    signals_dir.mkdir()
    (signals_dir / "signals_2026_week2.parquet").write_bytes(b"new")
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    leagues._write_stamp(persist_dir, "signals_2026_week2.parquet:old")
    calls = []
    monkeypatch.setattr(embed, "embed", lambda *a, **k: calls.append(1))
    return {"raw_dir": tmp_path / "sleeper", "persist_dir": persist_dir, "signals_dir": signals_dir, "calls": calls}


def _resync(league):
    return leagues._resync_signals_if_changed("1", league["raw_dir"], league["persist_dir"], league["signals_dir"])


def test_a_stale_league_is_rebuilt_when_nothing_else_is_rebuilding_it(stale_league):
    assert _resync(stale_league) is True
    assert stale_league["calls"] == [1]
    assert leagues._read_stamp(stale_league["persist_dir"]) == leagues.signals_fingerprint(stale_league["signals_dir"])


def test_resync_stands_down_while_that_collection_is_being_rebuilt(stale_league):
    with embed.rebuild_lock(stale_league["persist_dir"]):  # e.g. the scheduler thread, mid-embed of this league
        assert _resync(stale_league) is False
    assert stale_league["calls"] == []
    # and the stamp is left for the rebuilder to write -- the next request re-checks
    assert leagues._read_stamp(stale_league["persist_dir"]) == "signals_2026_week2.parquet:old"


def test_resync_stands_down_while_a_scheduler_cycle_is_running(stale_league, monkeypatch):
    refresh._cycle_running.set()
    try:
        assert _resync(stale_league) is False
    finally:
        refresh._cycle_running.clear()
    assert stale_league["calls"] == []
    assert _resync(stale_league) is True  # once the cycle is over, a still-stale league is rebuilt as before


def test_run_cycle_marks_itself_in_progress_only_while_running(monkeypatch):
    seen = []

    def fake_run(*args):
        seen.append(refresh.cycle_in_progress())
        return {"outcome": "ok"}

    monkeypatch.setattr(refresh, "_run_cycle", fake_run)
    assert refresh.cycle_in_progress() is False
    assert refresh.run_cycle(status_path=None) == {"outcome": "ok"}
    assert seen == [True]
    assert refresh.cycle_in_progress() is False


def test_run_cycle_clears_the_flag_even_if_the_cycle_raises(monkeypatch):
    def boom(*args):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(refresh, "_run_cycle", boom)
    with pytest.raises(RuntimeError):
        refresh.run_cycle(status_path=None)
    assert refresh.cycle_in_progress() is False
