"""Regression tests for tests/conftest.py's .env isolation.

Background (see conftest.py's docstring): after PR #18, a test that
`monkeypatch.delenv()`s a variable was silently un-isolated on any
machine with a real `.env`, because the function under test calls
`load_dotenv()`, which rediscovers the file and restores the variable.

Three tests, deliberately split so each proves one thing:
1. The MECHANISM, with isolation switched off: a no-arg `load_dotenv()`
   really does discover a `.env` by walking up from the *calling
   module's* directory and really does restore a deleted variable. No
   repo-root `.env` is written (that could clobber a developer's real
   one); instead a tiny caller module is imported from tmp_path so the
   genuine discovery walk starts there.
2. The SYMPTOM, with isolation switched off: the exact failure seen on
   Rohan's machine -- generate_report() after delenv fails with "No
   team found with roster_id='8'" rather than the MY_ROSTER_ID error.
3. The FIX: with the autouse isolation active (the normal state of every
   test), the same `.env` in cwd, the same caller module, and the same
   delenv leave the variable absent, and generate_report() raises the
   MY_ROSTER_ID error it should. This test fails if conftest.py's
   fixture is removed.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import dotenv.main
import polars as pl
import pytest

from src.reasoning import report
from tests.conftest import REAL_FIND_DOTENV
from tests.test_report import _CHRISTIAN_ROW, _CHRISTIAN_SIGNAL_ROW, _LEAGUE_ID, _SEASON, _WEEK, _setup

_ROHANS_ROSTER_ID = "8"  # the real value from the reported failure


def _write_dotenv(directory: Path) -> Path:
    """A .env shaped like a developer's real one -- MY_ROSTER_ID set to a
    roster that does NOT exist in the test fixture's teams.json."""
    path = directory / ".env"
    path.write_text(f"MY_ROSTER_ID={_ROHANS_ROSTER_ID}\nSLEEPER_LEAGUE_ID={_LEAGUE_ID}\n")
    return path


def _import_caller_module(directory: Path):
    """A module living in `directory` that calls load_dotenv() with no
    arguments -- the same call shape recommend()/generate_report() use.
    Because find_dotenv() starts its upward search from the directory of
    the module that called load_dotenv(), this is how a test exercises
    the genuine discovery walk against a temp directory instead of
    against the repo root."""
    src = directory / "dotenv_caller.py"
    src.write_text("from dotenv import load_dotenv\n\n\ndef load():\n    return load_dotenv()\n")
    spec = importlib.util.spec_from_file_location("dotenv_caller_for_test", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seeded_report_kwargs(tmp_path, monkeypatch) -> dict:
    roster = {"sleeper_cmc": {"full_name": "Christian McCaffrey", "position": "RB", "team": "SF"}}
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, roster, [_CHRISTIAN_SIGNAL_ROW], pl.DataFrame([_CHRISTIAN_ROW])
    )
    return dict(raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir)


# ---- 1. the mechanism, isolation deliberately OFF ----


def test_mechanism_no_arg_load_dotenv_discovers_a_dotenv_and_restores_a_deleted_var(tmp_path, monkeypatch):
    monkeypatch.setattr(dotenv.main, "find_dotenv", REAL_FIND_DOTENV)  # genuine discovery, just for this test
    _write_dotenv(tmp_path)
    caller = _import_caller_module(tmp_path)
    monkeypatch.delenv("MY_ROSTER_ID", raising=False)
    assert "MY_ROSTER_ID" not in os.environ

    loaded = caller.load()

    assert loaded is True  # a file was found and at least one variable set...
    assert os.environ["MY_ROSTER_ID"] == _ROHANS_ROSTER_ID  # ...and the "deleted" variable is back


# ---- 2. the symptom, isolation deliberately OFF ----


def test_symptom_generate_report_after_delenv_picks_up_the_dotenv_roster_id(tmp_path, monkeypatch):
    """Rohan's exact failure. The only simulated piece is *where* the
    file is found: find_dotenv() is pointed at a temp .env rather than
    the developer's real repo-root one (which this test must not
    overwrite). Everything after discovery -- load_dotenv() inside the
    real generate_report(), current_roster() reading the restored
    variable -- is the production code path."""
    kwargs = _seeded_report_kwargs(tmp_path, monkeypatch)
    dotenv_path = _write_dotenv(tmp_path)
    monkeypatch.setattr(dotenv.main, "find_dotenv", lambda *a, **kw: str(dotenv_path))
    monkeypatch.delenv("MY_ROSTER_ID")

    with pytest.raises(RuntimeError) as excinfo:
        report.generate_report("start_sit", _LEAGUE_ID, **kwargs)

    assert f"No team found with roster_id={_ROHANS_ROSTER_ID!r}" in str(excinfo.value)
    assert "MY_ROSTER_ID is not set" not in str(excinfo.value)


# ---- 3. the fix, isolation ON (the normal state of every test) ----


def test_dotenv_in_cwd_does_not_leak_into_a_test_expecting_the_var_absent(tmp_path, monkeypatch):
    kwargs = _seeded_report_kwargs(tmp_path, monkeypatch)
    _write_dotenv(tmp_path)
    monkeypatch.chdir(tmp_path)  # the .env is in cwd AND in the calling module's directory below
    caller = _import_caller_module(tmp_path)
    monkeypatch.delenv("MY_ROSTER_ID")

    # The real load_dotenv() still runs -- it just finds nothing.
    assert caller.load() is False
    assert "MY_ROSTER_ID" not in os.environ

    with pytest.raises(RuntimeError, match="MY_ROSTER_ID is not set"):
        report.generate_report("start_sit", _LEAGUE_ID, **kwargs)

    assert "MY_ROSTER_ID" not in os.environ  # still absent after the production code's own load_dotenv()


def test_isolation_does_not_stub_out_load_dotenv_itself():
    """The fixture patches file *discovery* only; load_dotenv() is the
    genuine python-dotenv function, so tests that assert it's called
    (tests/test_recommend.py, tests/test_report.py) still exercise the
    real thing and an explicit dotenv_path still works."""
    assert dotenv.main.load_dotenv is dotenv.load_dotenv
    assert dotenv.main.find_dotenv() == ""
