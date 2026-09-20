"""Phase 5.2: where each league's ingested data lives, and making sure
it exists before the API answers for that league.

Phase 5.1 deliberately left the on-disk convention alone: one league in
the flat data/raw/sleeper/ + data/chroma/ that `python -m
src.ingest.sleeper` / `python -m src.rag.embed` write, verified by
league_id at query time. A multi-league server needs one directory per
league. This module is that mapping -- and nothing else in the project
has to know about it, because recommend()/generate_report() already
take raw_dir/persist_dir explicitly.

Layout:
  - The league named by SLEEPER_LEAGUE_ID (the developer's own, already
    ingested by the CLI convention) keeps using the flat dirs, so the
    CLI and the API server see the same data for it.
  - Every other league lives under data/raw/leagues/<league_id>/
    {sleeper,chroma}/ -- under data/raw/ so the existing .gitignore rule
    (raw pulls are reproducible, never committed) already covers it.

The signals table (data/processed/signals/) is shared by every league:
it is league-agnostic by construction (CLAUDE.md's key principle), so
it is computed once and embedded into each league's Chroma collection
alongside that league's own roster/matchup chunks -- the same thing
src/rag/embed.py's main() already does for the single flat league.
(Re-embedding the shared signal chunks per league is redundant work at
friend-group scale; splitting the collection is a src/rag/ change and
was flagged, not done, in Phase 5.1.)

ensure_league_data() runs the existing ingest + embed for a league the
first time it's needed (network: Sleeper + nflverse's player list),
then just verifies it via Phase 5.1's load_league(). It also re-embeds
that league when the shared signals table has changed on disk since the
league's Chroma collection was last built -- see the signals-resync note
below for why that is needed and what it does NOT fix. Sleeper is blocked
in the sandbox this was built in, so the ingest path here is exercised
in tests only with the ingest/embed functions mocked -- see TODO.md's
Phase 5.2 entry for the live re-run.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import chromadb
from chromadb.api.shared_system_client import SharedSystemClient

from src.ingest import sleeper
from src.rag import embed
from src.scheduler import refresh  # cycle_in_progress(); refresh imports this module lazily, so no cycle
from src.rag.embed import CHROMA_DIR, RAW_DIR, SIGNALS_DIR
from src.reasoning.league import LeagueConfig, load_league

LEAGUES_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "leagues"

# ---- Chroma client warm-up (Phase 5.3 bugfix) ----
# src/rag/retrieve.py opens `chromadb.PersistentClient(path)` on every
# query. chromadb caches one shared System per path, but it registers the
# new System in its cache BEFORE `start()` finishes and takes no lock
# (SharedSystemClient._create_system_if_not_exists). So the first two
# requests to touch a league's Chroma index in a fresh server process can
# race: thread A is still starting the Rust bindings while thread B finds
# the cached-but-unstarted System and dies with
# "AttributeError: 'RustBindingsAPI' object has no attribute 'bindings'"
# -> "Could not connect to tenant default_tenant" -> HTTP 500. Observed
# for real: the Feed fires start_sit + drop + waiver_pickups together;
# start_sit and drop both read Chroma (per-player signal lookups) and
# 500'd, waiver_pickups reads only parquet and succeeded. A single curl
# never races. Once a path's System has started, later concurrent
# PersistentClient calls reuse it safely -- so the fix is to create the
# first client for each league path exactly once, under a lock, before
# any request-thread reaches retrieve.py. (The deeper fix, one shared
# client per path inside src/rag/retrieve.py, is a src/rag/ change and
# was out of scope for the bugfix pass that added this.)
#
# ---- ...and re-warming it when a refresh rewrites the index (Phase 5.7) ----
# That same process-wide System cache is also a staleness trap, which
# Phase 5.7's investigation reproduced through the real production
# functions (src/api/leagues.warm_chroma + src/rag/retrieve.query):
# once a process has warmed a path, chromadb keeps that path's in-memory
# vector index, and `collection.query()` is answered from it. So after
# `src/scheduler/refresh.py` re-embeds a league, a long-running server:
#   - never surfaces a chunk ADDED by the refresh (the new week's signal
#     chunks are invisible to semantic search), and
#   - returns phantom hits for chunks the refresh REMOVED -- same id,
#     document None, which is what retrieve.query() then hands the model.
# `collection.get()` (the metadata-filtered path: query_player_signal,
# so get_player_signals) reads SQLite directly and stays fresh, as do
# the parquet ranking tables and the raw Sleeper JSON. Only the semantic
# path goes stale -- measured, not assumed; see TODO.md's Phase 5.7 entry.
#
# The fix: remember the on-disk index's stamp when we warm a path and
# re-check it on every request (warm_chroma is already called per request
# by ensure_league_data). A changed stamp means someone re-embedded, so we
# forget chromadb's cached System for THAT path and build a new one, which
# reads the index off disk. Deliberately per-path rather than chromadb's
# public clear_system_cache(): that drops every path's System, and another
# league's request thread could then re-create its System outside this
# lock -- exactly the unsynchronized-start race the note above fixes.
# Nothing is stopped, only forgotten, so a query already in flight on the
# old System keeps working (verified under four concurrent query threads).
# This works whether the refresh runs inside this process (dev_server's
# background thread) or in a separate one (Phase 5.6's cron/worker).
_chroma_warm_lock = threading.Lock()
_chroma_warmed: dict[str, tuple[int, int] | None] = {}
_NEVER_WARMED = object()


def _index_stamp(persist_dir: Path) -> tuple[int, int] | None:
    """A cheap fingerprint of the on-disk Chroma index: the (mtime, size)
    of its SQLite file, which every add/delete touches. None when the
    index doesn't exist yet (an un-ingested league).

    Measured, because it decides where warm_chroma reads this: merely
    OPENING a PersistentClient bumps the SQLite mtime (creating the file
    on a fresh directory, touching it on an existing one), while reads
    afterwards do not. So the stamp to remember is the one taken AFTER
    the open -- a stamp taken before it is stale the moment we warm, and
    would make every single request rebuild the client.

    Known, accepted window: if a refresh's final commit lands inside our
    client construction, the stamp we record already covers a write our
    new System may have opened just before, and nothing would re-warm
    until the next refresh. Milliseconds wide, and invalidate_chroma()
    closes it outright for a refresh running in this process. Closing it
    for an out-of-process refresh too would mean a generation marker
    written by src/rag/embed.py itself -- a src/rag/ change, flagged in
    TODO.md rather than done here.
    """
    try:
        stat = (persist_dir / "chroma.sqlite3").stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _forget_cached_system(key: str) -> None:
    """Drop chromadb's cached System for this one path so the next
    PersistentClient(path=key) builds a fresh one. Uses chromadb
    internals on purpose (see the note above for why the public
    clear_system_cache() is the wrong tool here); falls back to it if a
    future chromadb drops these attributes, trading the race back for
    correctness rather than silently serving stale results."""
    try:
        SharedSystemClient._identifier_to_system.pop(key, None)
        SharedSystemClient._identifier_to_refcount.pop(key, None)
    except AttributeError:  # pragma: no cover - depends on the installed chromadb
        SharedSystemClient.clear_system_cache()


def warm_chroma(persist_dir: Path) -> None:
    """Open (and thereby fully start) the shared Chroma client for
    persist_dir, serialized across threads -- once per process, and again
    whenever the index on disk has been rewritten since. Idempotent and
    cheap after the first call (one stat() per call)."""
    key = str(persist_dir)
    if _chroma_warmed.get(key, _NEVER_WARMED) == _index_stamp(persist_dir):
        return
    with _chroma_warm_lock:
        if _chroma_warmed.get(key, _NEVER_WARMED) == _index_stamp(persist_dir):
            return
        if key in _chroma_warmed:
            _forget_cached_system(key)
        chromadb.PersistentClient(path=key)
        _chroma_warmed[key] = _index_stamp(persist_dir)  # after the open -- see _index_stamp


def invalidate_chroma(persist_dir: Path) -> None:
    """Force the next query against persist_dir to read the index from
    disk. warm_chroma()'s stamp check already catches an out-of-process
    refresh on the next request; src/scheduler/refresh.py calls this
    directly after re-embedding in-process so the very next query is
    fresh without depending on filesystem timestamp granularity."""
    key = str(persist_dir)
    with _chroma_warm_lock:
        if key in _chroma_warmed:
            _forget_cached_system(key)
            del _chroma_warmed[key]


# ---- Signals resync (investigated and added after a manual-refresh report) ----
# The signals table (data/processed/signals/*.parquet) is shared by every
# league and is refreshed OUT OF BAND, by re-running `python -m
# src.ingest.nflverse` + `python -m src.signals.matchup_signals` while the
# API server keeps running. Two different layers read it, and only one of
# them noticed the refresh:
#
#   - Reports read the parquet directly (ranking.load_signals_table
#     re-globs signals_dir on EVERY call), so a new week's file is picked
#     up with no restart and no cache to invalidate. What can still hide it
#     is the as-of-week bound: generate_report() infers as_of_week from
#     Sleeper's own nfl_state.json (recommend._infer_season_and_week), and
#     a signals-only refresh doesn't advance that file -- so the newer
#     week's rows are correctly filtered out as "future" until the Sleeper
#     ingest is re-run too. That is the as-of-date rule working, not a bug;
#     it is documented in README/TODO rather than "fixed" here, because
#     inferring the current week from the signals directory instead of the
#     league's own state would be a different (and weaker) definition.
#
#   - Chat reads the SAME numbers out of Chroma (get_player_signals ->
#     retrieve.query_player_signal), and Chroma is only ever written by
#     embed(). ensure_league_data() used to skip ingest entirely whenever
#     is_ingested() was true, so for an already-ingested league embed()
#     never ran again and the refreshed weeks were never embedded at all.
#     Restarting the server did NOT fix that -- is_ingested() is still
#     true on the next boot -- so the stale chunks were permanent until
#     someone re-ran the embed by hand or passed refresh=True.
#
# This is the TRIGGER half of freshness, and it is not the same thing as
# warm_chroma()'s stamp check above -- the two are complementary, and
# conflating them is what made the original investigation look wrong in
# both directions. chromadb has two read paths and they behave differently
# in a warmed process (both pinned in tests/test_api_signals_refresh.py):
#   - collection.get() -- metadata lookups, which is what
#     query_player_signal()/get_player_signals use -- reads SQLite directly
#     and DOES see another process's rewrite on the next call.
#   - collection.query() -- semantic search, which is what retrieve.query()/
#     search_league_info use -- answers from a per-process in-memory vector
#     index. Left alone it returns removed ids with documents/metadatas of
#     None, which callers doing r["metadata"].get(...) turn into an
#     AttributeError. That is what warm_chroma()'s (mtime, size) stamp
#     fixes, per request.
# So: warm_chroma() is the READ side -- once embed() has re-run by ANY
# means, a warm server notices. _resync_signals_if_changed() is the
# TRIGGER side -- it is what makes embed() re-run at all for an
# already-ingested league when no scheduler cycle happens to be forcing
# it (a manual refresh, a fast one between cycles, or ASKMADDEN_REFRESH_
# ENABLED=0). Neither substitutes for the other: without the read side a
# re-embed is invisible to a warm process, and without the trigger side
# there is no re-embed to notice.
#
# The gap this half closes is purely "nothing re-ran embed", handled by
# fingerprinting the signals files the collection was built from.
# (Leagues ingested before that fingerprint existed adopt it on first
# sight rather than rebuilding -- see the function's docstring for why,
# and for the one-time migration that leaves behind. Anything else that
# re-embeds a league -- src/scheduler/refresh.py -- must record the
# fingerprint too, via record_signals_fingerprint(), or the next request
# would rebuild a collection that is already current.)
_SIGNALS_STAMP = ".askmadden_signals_fingerprint"
_signals_resync_lock = threading.Lock()


def signals_fingerprint(signals_dir: Path) -> str:
    """Identity of the signals table as it currently sits on disk: every
    parquet's name, size and mtime. Changes whenever matchup_signals.py
    adds a week or rewrites one; empty string when there are no signals
    at all (which is a valid state -- see ingest_league)."""
    if not signals_dir.exists():
        return ""
    parts = []
    for path in sorted(signals_dir.glob("signals_*.parquet")):
        try:
            stat = path.stat()
        except OSError:  # file vanished mid-scan (a refresh writing right now)
            continue
        parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(parts)


def _read_stamp(persist_dir: Path) -> str | None:
    try:
        return (persist_dir / _SIGNALS_STAMP).read_text()
    except OSError:
        return None


def _write_stamp(persist_dir: Path, fingerprint: str) -> None:
    try:
        persist_dir.mkdir(parents=True, exist_ok=True)
        (persist_dir / _SIGNALS_STAMP).write_text(fingerprint)
    except OSError:
        pass  # a stamp we can't persist just means we re-check next time


def record_signals_fingerprint(persist_dir: Path, signals_dir: Path | None = None) -> None:
    """Record that persist_dir's collection was just rebuilt from the
    signals table as it currently stands.

    For embedders OUTSIDE this module -- today that means
    src/scheduler/refresh.py, whose every cycle re-embeds each league
    itself. Without this, that cycle leaves the fingerprint stamp
    describing the PREVIOUS signals table, so the next request through
    ensure_league_data() sees a mismatch and re-embeds a collection that
    is already current: a full rebuild per cycle, paid for by whichever
    user's request happens to arrive first. Reproduced before it was
    fixed; pinned by
    test_a_scheduler_cycle_does_not_leave_a_redundant_re_embed_behind.

    Deliberately separate from invalidate_chroma() rather than folded
    into it: that one is about THIS process's in-memory client (a no-op
    in a cron run), this one is about durable on-disk state every future
    process reads. A caller that re-embeds wants both, and refresh.py
    calls both."""
    _write_stamp(persist_dir, signals_fingerprint(SIGNALS_DIR if signals_dir is None else signals_dir))


def _resync_signals_if_changed(league_id: str, raw_dir: Path, persist_dir: Path, signals_dir: Path) -> bool:
    """Re-embed league_id when signals_dir has changed since its Chroma
    collection was built. Returns True if a re-embed actually happened.

    Local only -- this re-runs embed() against the ALREADY-ingested
    raw_dir, so it never re-hits Sleeper; it is the embed half of
    ingest_league(), not the network half.

    Serialized on _signals_resync_lock, and called from
    ensure_league_data() -- which every league-scoped request goes through
    before it touches Chroma -- so two requests never both rebuild. It
    also stands down when the scheduler is mid-cycle or this collection is
    already being rebuilt (see the inline note below): embed() is
    upsert-based, so a collection under rebuild is a usable, never-empty
    collection, and the rebuilder will stamp it when done.

    A league with NO stamp at all -- one ingested before this check
    existed -- adopts the current fingerprint WITHOUT re-embedding. The
    alternative (treat "unknown" as "stale" and rebuild) would make the
    first request after upgrading silently re-embed every existing
    league, and would turn a partially-written raw_dir -- which
    is_ingested() happily accepts, it only looks for league.json -- into
    a failed request where today it is merely incomplete. The cost is
    that a league whose signals were refreshed BEFORE this code landed
    stays stale until the next refresh; that one-time migration is
    documented in README/TODO as a single `refresh: true` session, not
    left for someone to discover."""
    fingerprint = signals_fingerprint(signals_dir)
    stamp = _read_stamp(persist_dir)
    if stamp == fingerprint:
        return False
    if stamp is None:
        _write_stamp(persist_dir, fingerprint)
        return False
    # Someone else is already rebuilding this league -- the scheduler's
    # cycle (which re-embeds and stamps every ingested league itself), or
    # this collection specifically -- so a rebuild here would be the same
    # work a second time, paid for by this request (20-40s), while the two
    # interleave their writes into one collection. Serve what is on disk
    # instead: embed() is upsert-based, so it is never empty mid-rebuild,
    # and the finisher writes the stamp. Real case, 2026-09-20: a league
    # select that took over a minute, and a chat answer with no signal
    # data for two QBs, both while the 6-hourly cycle was mid-embed.
    if embed.rebuild_in_progress(persist_dir) or refresh.cycle_in_progress():
        return False
    with _signals_resync_lock:
        if _read_stamp(persist_dir) == fingerprint:  # another request just did it
            return False
        if embed.rebuild_in_progress(persist_dir) or refresh.cycle_in_progress():
            return False
        embed.embed(
            persist_dir=persist_dir,
            raw_dir=raw_dir,
            signals_dir=signals_dir if signals_dir.exists() else None,
        )
        _write_stamp(persist_dir, fingerprint)
        return True


def league_dirs(league_id: str) -> tuple[Path, Path]:
    """(raw_dir, persist_dir) for league_id -- the flat CLI dirs for the
    developer's own SLEEPER_LEAGUE_ID league, a per-league directory for
    everyone else's."""
    league_id = str(league_id)
    if league_id == (os.environ.get("SLEEPER_LEAGUE_ID") or "").strip():
        return RAW_DIR, CHROMA_DIR
    return LEAGUES_DIR / league_id / "sleeper", LEAGUES_DIR / league_id / "chroma"


def is_ingested(league_id: str) -> bool:
    raw_dir, persist_dir = league_dirs(league_id)
    return (raw_dir / "league.json").exists() and persist_dir.exists()


def ingested_league_ids() -> list[str]:
    """Every league whose data is actually on disk, newest-registered
    last: the per-league directories plus (when it's ingested) the flat
    SLEEPER_LEAGUE_ID league. This is what src/scheduler/refresh.py
    iterates over.

    "On disk" rather than "in the SQLite leagues table" on purpose. That
    table holds every league Sleeper listed for everyone who has logged
    in, including ones nobody ever opened; refreshing those would hit
    Sleeper for leagues with no local data to keep current. A league is
    ingested the first time a session selects it (ensure_league_data),
    and from then on it's here. It also means the scheduler needs no
    database access at all, which matters for Phase 5.6's cron/worker
    shape.
    """
    found: list[str] = []
    own = (os.environ.get("SLEEPER_LEAGUE_ID") or "").strip()
    if own and is_ingested(own):
        found.append(own)
    if LEAGUES_DIR.exists():
        for child in sorted(LEAGUES_DIR.iterdir()):
            if child.is_dir() and child.name not in found and is_ingested(child.name):
                found.append(child.name)
    return found


def ingest_league(league_id: str, week: int | None = None) -> tuple[Path, Path]:
    """Pull league_id from Sleeper and (re)build its Chroma collection,
    using the exact same functions the CLI convention uses
    (src/ingest/sleeper.py's run(), src/rag/embed.py's embed()) pointed
    at this league's directories. Network."""
    raw_dir, persist_dir = league_dirs(league_id)
    sleeper.run(league_id=str(league_id), week=week, out_dir=raw_dir)
    embed.embed(persist_dir=persist_dir, raw_dir=raw_dir, signals_dir=SIGNALS_DIR if SIGNALS_DIR.exists() else None)
    return raw_dir, persist_dir


def ensure_league_data(league_id: str, refresh: bool = False) -> LeagueConfig:
    """The verified LeagueConfig for league_id, ingesting it first if it
    isn't on disk yet (or if refresh=True). Raises LeagueMismatchError
    via load_league() if the directory somehow holds a different league.

    For a league that IS already ingested, re-embeds it when the shared
    signals table has changed on disk since its collection was built, so
    an out-of-band `matchup_signals` refresh reaches the chat path without
    a restart (which would not have helped anyway) -- see the
    signals-resync note above. SIGNALS_DIR is read through the module
    global at call time, not bound as a default, so tests and any future
    per-deployment override can point it elsewhere."""
    raw_dir, persist_dir = league_dirs(league_id)
    signals_dir = SIGNALS_DIR
    if refresh or not is_ingested(league_id):
        ingest_league(league_id)
        # Stamp here rather than inside ingest_league() so a fresh ingest
        # is never immediately followed by a redundant re-embed.
        _write_stamp(persist_dir, signals_fingerprint(signals_dir))
    else:
        _resync_signals_if_changed(league_id, raw_dir, persist_dir, signals_dir)
    config = load_league(str(league_id), raw_dir=raw_dir, persist_dir=persist_dir)
    warm_chroma(config.persist_dir)  # see the note above -- must happen before any request thread queries Chroma
    return config
