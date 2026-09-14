"""Does a RUNNING API server notice a signals refresh done out of band?

The operational scenario this pins down: the league is already ingested,
the server is up and has already answered a request (so its Chroma client
is warmed by src/api/leagues.py's warm_chroma), and then in another
terminal `python -m src.ingest.nflverse && python -m src.signals.matchup_signals`
writes a NEW week's signals parquet. Nothing restarts. What do the report
and chat paths serve after that?

The investigation found three separate answers, one per layer, and only
the third was actually broken -- so all three are pinned here:

  1. chromadb's two read paths behave differently in a warmed process.
     collection.get() (metadata lookups -- query_player_signal, so
     get_player_signals) reads SQLite and sees a separate process's rewrite
     on the very next call. collection.query() (semantic search --
     retrieve.query, so search_league_info) answers from a per-process
     in-memory vector index, and left alone returns removed ids with
     documents/metadatas of None. That second path is a real defect, and
     the one PR #28 fixed with an (mtime, size) stamp on warm_chroma().
     Since #28 landed, the test below asserts the freshness rather than
     the limitation, and pins the boundary (bypassing warm_chroma is
     still stale) so the reason the fix sits where it does stays visible.
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

Findings 1 and 3 are the two halves of one guarantee -- #28's READ side
(a re-embed, by whatever means, is noticed by a warm server) and this
unit's TRIGGER side (a re-embed happens at all for an already-ingested
league). The last two tests in this file exercise them together, in both
directions: a refresh this process resyncs itself, and a refresh the
scheduler already re-embedded.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from src.api import leagues, main
from src.rag import embed, retrieve
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


_MATCHUP_CHUNK_IN_SUBPROCESS = r"""
import sys
sys.path.insert(0, {repo!r})
from pathlib import Path
from src.rag import embed
embed.embed(
    [{{"id": "c{week}", "text": "Week {week} matchup 1: Team A vs Team B",
      "metadata": {{"type": "matchup", "week": {week}}}}}],
    persist_dir=Path({persist_dir!r}),
)
"""


def _embed_matchup_chunk_in_subprocess(persist_dir: Path, week: int) -> None:
    """Replace the collection with a single matchup chunk for `week`, from a
    genuinely separate process (embed() clears then re-adds, so this both
    removes the old id and adds a new one)."""
    code = _MATCHUP_CHUNK_IN_SUBPROCESS.format(repo=str(REPO), persist_dir=str(persist_dir), week=week)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]


def _embed_in_subprocess(persist_dir: Path, target_share: float) -> None:
    code = _REEMBED_IN_SUBPROCESS.format(
        repo=str(REPO), persist_dir=str(persist_dir), target_share=target_share
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]


def test_warmed_client_metadata_lookups_see_another_process_rewrite(tmp_path):
    """chromadb's METADATA path stays fresh in a warmed process.

    query_player_signal() -- and therefore the get_player_signals chat tool
    and the prior-season fallback -- goes through collection.get(), which
    reads SQLite directly. A warmed per-league client sees another
    process's rewrite on the next call, so this path needed no fix.

    The semantic path does NOT share that property; see
    test_warmed_client_semantic_queries_go_stale_after_another_process_rewrite
    immediately below. Keeping both pinned is the point: the original
    suspicion about warm_chroma() was half right, and conflating the two
    paths is what made it look wholly wrong."""
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


def test_warmed_client_semantic_queries_are_fresh_after_another_process_rewrite(tmp_path):
    """chromadb's SEMANTIC path stays fresh in a warmed process -- now.

    This test used to assert the opposite, as a known limitation: this
    unit's own PR pinned collection.query() going stale after an
    out-of-process re-embed (returning removed ids with documents and
    metadatas of None, which a caller doing r["metadata"].get("type")
    turns into an AttributeError), and its docstring said to delete it
    once PR #28's warm_chroma() stamp landed. #28 has landed, so the
    assertion is inverted rather than deleted -- a test proving the
    freshness we now depend on is worth more than the absence of a test
    documenting a limitation we no longer have.

    The re-warm below is not test scaffolding: warm_chroma() is where
    #28's (mtime, size) check lives, and ensure_league_data() calls it on
    every league-scoped request, so this is exactly what the second
    request after a refresh does. The boundary is asserted too --
    querying without going through warm_chroma() is still stale, because
    chromadb's per-process vector index has no way to know -- which is
    why nothing should reach retrieve.query() around
    ensure_league_data()."""
    persist_dir = tmp_path / "chroma"
    _embed_matchup_chunk_in_subprocess(persist_dir, 3)

    leagues.warm_chroma(persist_dir)
    before = retrieve.query("what is my matchup", n_results=3, persist_dir=persist_dir)
    assert [r["text"] for r in before] == ["Week 3 matchup 1: Team A vs Team B"], before

    _embed_matchup_chunk_in_subprocess(persist_dir, 8)  # deletes c3, adds c8

    # The boundary: no re-warm, so chromadb still answers from the index
    # it built at warm time. This is what #28 exists to prevent reaching
    # a request, not something any production path does.
    stale = retrieve.query("what is my matchup", n_results=3, persist_dir=persist_dir)
    assert [r["text"] for r in stale] == [None], (
        "the per-process vector index refreshed on its own -- if chromadb ever starts "
        "doing that, warm_chroma()'s stamp check can be simplified"
    )

    # ...and the real path: every request re-warms first, which notices
    # the index's changed (mtime, size) and rebuilds the client.
    leagues.warm_chroma(persist_dir)
    fresh = retrieve.query("what is my matchup", n_results=3, persist_dir=persist_dir)
    assert [r["text"] for r in fresh] == ["Week 8 matchup 1: Team A vs Team B"], (
        "a re-warmed client still served the pre-refresh index -- PR #28's warm_chroma "
        "stamp check has regressed"
    )
    assert fresh[0]["metadata"] is not None


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


# ------------------------------------------------- both fixes, together
# Neither PR tested this interaction: #27 only ever proved that a resync
# gets triggered, #28 only ever proved that a triggered re-embed gets
# noticed. The end-to-end path that actually matters is the two in
# sequence, through a warm server, for the SEMANTIC read path (the one
# that goes stale) rather than the metadata one.


def test_a_refresh_reaches_semantic_search_on_a_warm_server(running_server):
    """Trigger side + read side, in one pass, through the real endpoint.

    A warm server, a signals refresh on disk, and then one ordinary
    request: ensure_league_data() notices the changed fingerprint and
    re-embeds (#27), and warm_chroma() -- called at the end of that same
    ensure_league_data() -- re-reads the rewritten index so the semantic
    path serves the new chunks (#28). Asserted on retrieve.query(), not
    query_player_signal(), because the metadata path was never the one
    at risk."""
    config = leagues.ensure_league_data(VS30_ID)

    def barkley_hits():
        hits = retrieve.query("Saquon Barkley usage", n_results=10, persist_dir=config.persist_dir)
        return [h["text"] for h in hits if h["text"] and "S.Barkley" in h["text"]]

    before = barkley_hits()
    assert before, "no Barkley signal chunk to begin with -- fixture problem, not a finding"
    assert not any("99.90%" in t for t in before), before

    _refresh_signals(running_server["signals_dir"])
    assert running_server["client"].get(
        f"/api/reports/drop?session_id={running_server['session_id']}"
    ).status_code == 200

    after = barkley_hits()
    assert any("99.90%" in t for t in after), (
        f"semantic search still serves pre-refresh chunks: {after}"
    )
    # The specific failure mode #28 fixes: removed ids coming back as
    # None and blowing up search_league_info's r["metadata"].get(...).
    assert all(
        h["metadata"] is not None
        for h in retrieve.query("Saquon Barkley usage", n_results=10, persist_dir=config.persist_dir)
    ), "phantom None hits survived the refresh"


def test_a_scheduler_cycle_does_not_leave_a_redundant_re_embed_behind(running_server, monkeypatch):
    """The other half of the combination: when the SCHEDULER does the
    re-embed, the next request must not redo it.

    src/scheduler/refresh.py re-embeds every league itself each cycle.
    Before refresh.py recorded the fingerprint, that left the stamp
    describing the previous signals table, so the first request after
    every cycle hit a mismatch and rebuilt an already-current collection
    -- a full re-embed charged to whichever user's request landed first.
    Reproduced, then fixed by _note_reembedded(); this pins it.

    The semantic path is still asserted fresh afterwards, so "no second
    re-embed" can never be satisfied by simply not refreshing."""
    from src.scheduler import refresh

    signals_dir = running_server["signals_dir"]
    raw_dir, persist_dir = leagues.league_dirs(VS30_ID)
    _refresh_signals(signals_dir)

    calls: list = []
    real_embed = leagues.embed.embed

    def counting_embed(*args, **kwargs):
        calls.append(kwargs.get("persist_dir"))
        return real_embed(*args, **kwargs)

    monkeypatch.setattr(leagues.embed, "embed", counting_embed)
    monkeypatch.setattr(refresh.embed, "embed", counting_embed)
    monkeypatch.setattr(refresh.sleeper, "run", lambda **kwargs: [])  # blocked in this sandbox

    result = refresh.refresh_league(VS30_ID, raw_dir, persist_dir, signals_dir=signals_dir)
    assert result["embed"] == "ok", result
    assert len(calls) == 1, calls

    for _ in range(3):
        assert running_server["client"].get(
            f"/api/reports/drop?session_id={running_server['session_id']}"
        ).status_code == 200
    assert len(calls) == 1, (
        f"the scheduler already re-embedded this league, but {len(calls) - 1} more "
        "re-embed(s) were charged to the next request(s)"
    )

    hits = retrieve.query("Saquon Barkley usage", n_results=10, persist_dir=persist_dir)
    assert any(h["text"] and "99.90%" in h["text"] for h in hits), (
        f"no redundant re-embed, but also no fresh data: {[h['text'] for h in hits]}"
    )


_CRON_CYCLE_IN_SUBPROCESS = r"""
import sys
sys.path.insert(0, {repo!r})
from pathlib import Path
from src.api import leagues
from src.rag import embed
persist_dir, raw_dir = Path({persist_dir!r}), Path({raw_dir!r})
signals_dir = Path({signals_dir!r})
# Exactly what src/scheduler/refresh.py's refresh_league() does after its
# Sleeper pull, minus the Sleeper pull (blocked in this sandbox).
embed.embed(persist_dir=persist_dir, raw_dir=raw_dir, signals_dir=signals_dir)
leagues.invalidate_chroma(persist_dir)
leagues.record_signals_fingerprint(persist_dir, signals_dir)
"""


def _run_cron_cycle_in_subprocess(persist_dir: Path, raw_dir: Path, signals_dir: Path) -> None:
    code = _CRON_CYCLE_IN_SUBPROCESS.format(
        repo=str(REPO), persist_dir=str(persist_dir), raw_dir=str(raw_dir), signals_dir=str(signals_dir)
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]


def test_an_out_of_process_cycle_reaches_semantic_search_without_a_restart(running_server, monkeypatch):
    """The case that needs BOTH fixes, and that neither PR covered.

    Phase 5.6 runs the scheduler as its own cron/worker process, so the
    re-embed happens somewhere the API server's memory knows nothing
    about. That breaks the convenient property the in-process tests lean
    on (an in-process embed() goes through the same cached System, so
    that process's vector index is correct for free). Here:

      - #27's trigger side must NOT fire -- the cycle already embedded
        and recorded the fingerprint, so a re-embed on the next request
        would be pure waste (asserted: zero embeds in this process), and
      - #28's read side is then the ONLY thing that can make the warm
        server serve the new chunks (asserted: fresh semantic hits).

    Neutering warm_chroma()'s stamp check fails this test; neutering the
    fingerprint recording makes the first assertion fail. That is the
    whole point of merging the two PRs rather than either alone."""
    signals_dir = running_server["signals_dir"]
    config = leagues.ensure_league_data(VS30_ID)
    persist_dir, raw_dir = config.persist_dir, leagues.league_dirs(VS30_ID)[0]

    def barkley_hits():
        hits = retrieve.query("Saquon Barkley usage", n_results=10, persist_dir=persist_dir)
        return [h["text"] for h in hits if h["text"] and "S.Barkley" in h["text"]]

    assert barkley_hits() and not any("99.90%" in t for t in barkley_hits())

    _refresh_signals(signals_dir)
    _run_cron_cycle_in_subprocess(persist_dir, raw_dir, signals_dir)

    embeds: list = []
    real_embed = leagues.embed.embed
    monkeypatch.setattr(
        leagues.embed, "embed",
        lambda *a, **k: (embeds.append(k.get("persist_dir")), real_embed(*a, **k))[1],
    )

    assert running_server["client"].get(
        f"/api/reports/drop?session_id={running_server['session_id']}"
    ).status_code == 200
    assert embeds == [], "re-embedded work the out-of-process cycle had already done"

    after = barkley_hits()
    assert any("99.90%" in t for t in after), (
        f"a warm server never saw the out-of-process re-embed: {after}"
    )
