"""Shared test setup.

**Isolating tests from a developer's real .env file.** PR #18 moved
`load_dotenv()` inside recommend()/generate_report() (correct for
production: a direct import gets the same behavior as the CLI). The
side effect on tests: any test that `monkeypatch.delenv()`s a variable
to simulate it being unset was no longer actually isolated -- the
function under test called `load_dotenv()`, which walked up from
src/reasoning/ to the repo root, found the developer's real `.env`,
and quietly put the deleted variable straight back. Invisible in CI
and in a sandbox (no `.env` there), real on a developer machine:
`test_start_sit_requires_my_roster_id` failed on Rohan's machine with
"No team found with roster_id='8'" (his real MY_ROSTER_ID) instead of
the expected "MY_ROSTER_ID is not set" error. Reproduced directly in
the Phase 5.1 follow-up session by dropping a `.env` with
MY_ROSTER_ID=8 into the repo root: identical failure.

The fix lives here, not in the production code: python-dotenv's
`load_dotenv()` (called with no path, as every module in this project
does) locates the file via `dotenv.main.find_dotenv()`, looked up by
name at call time. The autouse fixture below replaces that lookup with
one that finds nothing, for every test, so `load_dotenv()` still runs
for real (tests asserting it's *called* keep working unchanged) but
can never read a file off the developer's disk. Any test that needs
the genuine discovery behavior (tests/test_dotenv_isolation.py's
mechanism tests) restores REAL_FIND_DOTENV for itself.

Why patch discovery rather than each module's `load_dotenv` reference:
every module that does `from dotenv import load_dotenv` holds its own
reference, so a per-module patch list would go stale the moment a new
module loads .env (Phase 5.2's API layer, for one). Discovery is the
single choke point they all share. (python-dotenv >= 1.1 also honors a
PYTHON_DOTENV_DISABLED env var; not relied on here so the isolation
doesn't depend on which dotenv version a machine has installed.)
"""
from __future__ import annotations

import dotenv.main
import pytest

# The genuine file-discovery function, captured before any test patches
# it -- for tests that deliberately need the real behavior.
REAL_FIND_DOTENV = dotenv.main.find_dotenv


def _find_nothing(*args, **kwargs) -> str:
    """find_dotenv()'s "no .env file found" result -- load_dotenv() then
    reads an empty stream and sets nothing, exactly as on a machine with
    no .env file at all."""
    return ""


@pytest.fixture(autouse=True)
def isolate_tests_from_real_dotenv_file(monkeypatch):
    """Every test runs as if no .env file exists anywhere on disk, so a
    `monkeypatch.delenv()` stays deleted regardless of what the developer
    running the suite has in their real .env. See module docstring."""
    monkeypatch.setattr(dotenv.main, "find_dotenv", _find_nothing)
