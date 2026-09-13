"""Does a RUNNING API server notice a signals refresh done out of band?

The operational scenario this pins down: the league is already ingested,
the server is up and has already answered a request (so its Chroma client
is warmed by src/api/leagues.py's warm_chroma), and then in another
terminal `python -m src.ingest.nflverse && python -m src.signals.matchup_signals`
writes a NEW week's signals parquet. Nothing restarts. What do the report
and chat paths serve after that?

The investigation found three separate answers, one per layer, and only
the third was actually broken -- so all three are pinned here:

  1. warm_chroma()'s cached client is NOT a staleness source. A warmed
     client sees rows a SEPARATE PROCESS wrote to the same path on its
     very next query (test_warmed_chroma_client_sees_another_process_rewrite).
     This was the original suspicion and it is wrong; the test exists so a
     future reader doesn't re-suspect it.
  2. Reports read the parquet fresh on every request
     (ranking.load_signals_table re-globs), so refreshed numbers are
     served with no restart -- but only within the as-of-week bound, and
     the default bound comes from Sleeper's nfl_state.json, which a
     signals-only refresh does not advance. So the DEFAULT report keeps
     showing the old week (correctly -- the newer rows are "future" data
     by the as-of-date rule) while an explicit as_of_week shows the new
     numbers immediately.
  3. Chat reads those numbers out of Chroma, and Chroma is only written by
     embed(). ensure_league_data() skipped ingest entirely for an
     already-ingested league, so embed() never re-ran and the refreshed
     week was never embedded -- permanently, across restarts too. That is
     the real gap; _resync_signals_if_changed() closes it.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from src.api import leagues, main
from src.rag import retrieve
from tests.test_api_main import _login_and_session, api  # noqa: F401  (fixture)
from tests.test_league import _ALL_SIGNALS, _SEASON, _WEEK, VS30_ID

REPO = Path(__file__).resolve().parents[1]
NEW_WEEK = _WEEK + 1
BARKLEY = _ALL_SIGNALS[0]["player_id"]

# Deliberately extreme so a stale read is unmistakable in an assertion.
REFRESHED_TARGET_SHARE = 0.999
REFRESHED_RED_ZONE_SHARE = 0.99


def _refresh_signals(signals_dir: Path) -> None:
    """Exactly what a real `matchup_signals` re-run leaves behind: a new
    week's parquet next to the existing ones. It does NOT touch the
    league's raw Sleeper pull (so nfl_state.json keeps its old
    display_week) and it does NOT touch Chroma."""
    rows = [dict(row) for row in _ALL_SIGNALS]
    for row in rows:
        row["as_of_week"] = NEW_WEEK
    rows[0]["target_share"] = REFRESHED_TARGET_SHARE
    rows[0]["red_zone_share"] = REFRESHED_RED_ZONE_SHARE
    pl.DataFrame(rows).write_parquet(signals_dir / f"signals_{_SEASON}_week{NEW_WEEK}.parquet")


@pytest.fixture
def running_server(api, monkeypatch):  # noqa: F811
    """A server that has already ingested the league and answered a
    request -- i.e. warm, exactly like the real operational case."""
    # The api fixture points main at a temp signals dir; leagues.py reads
    # its own module global, so point that at the same place.
    signals_dir = main.SIGNALS_DIR
    monkeypatch.setattr(leagues, "SIGNALS_DIR", signals_dir)
    session_id = _login_and_session(api)
    first = api["client"].get(f"/api/reports/drop?session_id={session_id}")
    assert first.status_code == 200, first.text
    return {**api, "session_id": session_id, "signals_dir": signals_dir, "before": first.json()}


def _barkley_entry(payload: dict) -> dict:
    matches = [e for e in payload["entries"] if "Barkley" in e["name"]]
    assert matches, f"Barkley missing from {[e['name'] for e in payload['entries']]}"
    return matches[0]


# ---------------------------------------------------------------- finding 1

_REEMBED_IN_SUBPROCESS = r"""
import sys
sys.path.insert(0, {repo!r})
from pathlib import Path
from src.rag import embed
from tests.test_league import _ALL_SIGNALS
rows = [dict(r) for r in _ALL_SIGNALS]
rows[0]["target_share"] = {target_share}
embed.embed(embed.build_signal_chunks(rows), persist_dir=Path({persist_dir!r}))
"""


def _embed_in_subprocess(persist_dir: Path, target_share: float) -> None:
    code = _REEMBED_IN_SUBPROCESS.format(
        repo=str(REPO), persist_dir=str(persist_dir), target_share=target_share
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]


def test_warmed_chroma_client_sees_another_process_rewrite(tmp_path):
    """warm_chroma()'s cached client is not a staleness source.

    This is the negative result that redirected the whole investigation:
    the suspicion was that PR #24's warmed per-league client would keep
    serving the embeddings it first saw. It does not -- retrieve.py opens a
    PersistentClient per query against a SQLite-backed store, so a rewrite
    from a different process is visible on the next read."""
    persist_dir = tmp_path / "chroma"
    _embed_in_subprocess(persist_dir, 0.111)

    leagues.warm_chroma(persist_dir)  # warm exactly as a live server would
    before = retrieve.query_player_signal(BARKLEY, season=_SEASON, as_of_week=_WEEK, persist_dir=persist_dir)
    assert before is not None and "11.10%" in before["text"], before

    _embed_in_subprocess(persist_dir, 0.999)  # a refresh, while this process stays up

    after = retrieve.query_player_signal(BARKLEY, season=_SEASON, as_of_week=_WEEK, persist_dir=persist_dir)
    assert after is not None and "99.90%" in after["text"], (
        "a warmed client failed to see another process's rewrite -- if this ever fails, "
        "the warm_chroma note in src/api/leagues.py needs revisiting"
    )


# ---------------------------------------------------------------- finding 2


def test_report_serves_refreshed_signals_without_restart(running_server):
    """The parquet layer has no cache to invalidate: ask for the refreshed
    week explicitly and the new numbers come straight back, same process."""
    _refresh_signals(running_server["signals_dir"])

    resp = running_server["client"].get(
        f"/api/reports/drop?session_id={running_server['session_id']}&as_of_week={NEW_WEEK}"
    )
    assert resp.status_code == 200, resp.text
    summary = _barkley_entry(resp.json())["signals_summary"]
    assert "100%" in summary, f"refreshed target share not served: {summary!r}"
    assert summary != _barkley_entry(running_server["before"])["signals_summary"]


def test_default_report_stays_pinned_to_the_leagues_own_state_week(running_server):
    """...but the DEFAULT report still shows the old week, because
    as_of_week is inferred from Sleeper's nfl_state.json and a
    signals-only refresh doesn't advance it. This is the as-of-date rule
    holding (the new rows are future data relative to the league's state),
    not a cache -- pinned so the behaviour is a documented decision rather
    than a surprise, and so that changing the inference is a deliberate act."""
    _refresh_signals(running_server["signals_dir"])

    resp = running_server["client"].get(f"/api/reports/drop?session_id={running_server['session_id']}")
    assert resp.status_code == 200, resp.text
    assert _barkley_entry(resp.json())["signals_summary"] == _barkley_entry(running_server["before"])["signals_summary"]

    # ...and the reason is on disk, not in memory:
    config = leagues.ensure_league_data(VS30_ID)
    state = json.loads((config.raw_dir / "nfl_state.json").read_text())["data"]
    assert int(state["display_week"]) == _WEEK


# ---------------------------------------------------------------- finding 3


def test_chat_signals_are_reembedded_after_a_refresh(running_server):
    """The real gap: Chroma is only written by embed(), and an
    already-ingested league never re-ran it. ensure_league_data() now
    re-embeds when the signals table's fingerprint changes, so the
    refreshed week reaches the chat path (get_player_signals ->
    retrieve.query_player_signal) with no restart."""
    config = leagues.ensure_league_data(VS30_ID)
    assert retrieve.query_player_signal(
        BARKLEY, season=_SEASON, as_of_week=NEW_WEEK, persist_dir=config.persist_dir
    ) is None, "the refreshed week should not be embedded before the refresh happens"

    _refresh_signals(running_server["signals_dir"])

    # Any league-scoped request goes through ensure_league_data(), which is
    # where the resync happens -- so a plain report request is enough.
    assert running_server["client"].get(
        f"/api/reports/drop?session_id={running_server['session_id']}"
    ).status_code == 200

    chunk = retrieve.query_player_signal(
        BARKLEY, season=_SEASON, as_of_week=NEW_WEEK, persist_dir=config.persist_dir
    )
    assert chunk is not None, "refreshed signals never reached Chroma -- chat would serve stale numbers"
    assert "99.90%" in chunk["text"], chunk["text"]


def test_without_the_resync_chat_would_stay_stale(running_server, monkeypatch):
    """The same scenario with the resync neutralized, proving the test
    above is actually load-bearing: this is what the server did before,
    and what a restart would NOT have fixed (is_ingested() stays true)."""
    monkeypatch.setattr(leagues, "_resync_signals_if_changed", lambda *a, **k: False)
    _refresh_signals(running_server["signals_dir"])

    assert running_server["client"].get(
        f"/api/reports/drop?session_id={running_server['session_id']}"
    ).status_code == 200

    config = leagues.ensure_league_data(VS30_ID)
    assert retrieve.query_player_signal(
        BARKLEY, season=_SEASON, as_of_week=NEW_WEEK, persist_dir=config.persist_dir
    ) is None


def test_resync_runs_once_per_refresh_not_once_per_request(running_server, monkeypatch):
    """The resync is fingerprint-gated, so a warm server doesn't re-embed
    on every request -- that would make every Feed load rebuild the
    collection."""
    calls: list = []
    real_embed = leagues.embed.embed

    def counting_embed(*args, **kwargs):
        calls.append(kwargs.get("persist_dir"))
        return real_embed(*args, **kwargs)

    monkeypatch.setattr(leagues.embed, "embed", counting_embed)
    client, session_id = running_server["client"], running_server["session_id"]

    for _ in range(3):
        assert client.get(f"/api/reports/drop?session_id={session_id}").status_code == 200
    assert calls == [], "re-embedded without any signals change"

    _refresh_signals(running_server["signals_dir"])
    for _ in range(3):
        assert client.get(f"/api/reports/drop?session_id={session_id}").status_code == 200
    assert len(calls) == 1, f"expected exactly one re-embed for one refresh, got {len(calls)}"


def test_a_league_with_no_stamp_adopts_it_instead_of_rebuilding(tmp_path, monkeypatch):
    """A league ingested before the fingerprint existed must NOT trigger a
    surprise re-embed on the first request after upgrading -- it adopts the
    current fingerprint and only resyncs on the NEXT real change.

    This is the deliberate trade recorded in _resync_signals_if_changed's
    docstring: treating "unknown" as "stale" would re-embed every existing
    league on upgrade, and would turn a partially-written raw_dir (which
    is_ingested() accepts -- it only looks for league.json) into a failed
    request. The cost, a one-time `refresh: true` for leagues refreshed
    before this landed, is documented rather than silently absorbed."""
    signals_dir = tmp_path / "signals"
    signals_dir.mkdir()
    pl.DataFrame(_ALL_SIGNALS).write_parquet(signals_dir / f"signals_{_SEASON}_week{_WEEK}.parquet")
    persist_dir, raw_dir = tmp_path / "chroma", tmp_path / "sleeper"
    persist_dir.mkdir()

    def explode(*args, **kwargs):
        raise AssertionError("must not re-embed a league that has no stamp yet")

    monkeypatch.setattr(leagues.embed, "embed", explode)

    assert leagues._resync_signals_if_changed("555", raw_dir, persist_dir, signals_dir) is False
    assert leagues._read_stamp(persist_dir) == leagues.signals_fingerprint(signals_dir)

    # ...and now that it has a stamp, a real change IS picked up.
    monkeypatch.undo()
    embedded: list = []
    monkeypatch.setattr(leagues.embed, "embed", lambda *a, **kw: embedded.append(kw.get("persist_dir")))
    _refresh_signals(signals_dir)
    assert leagues._resync_signals_if_changed("555", raw_dir, persist_dir, signals_dir) is True
    assert embedded == [persist_dir]
