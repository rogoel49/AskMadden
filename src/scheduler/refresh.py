"""Phase 5.7: the automated ingest -> signals -> embed refresh.

PROJECT_SPEC.md has listed this module since Phase 1 ("runs ingest ->
signals -> embed on a schedule (shared across all leagues)") and it has
been a flagged gap since Phase 2: every report and every recommend()
call was only as current as whenever someone last ran
`matchup_signals.py` + `embed.py` + `sleeper.py` by hand. On a live
server that is not viable, so this module runs the same pipeline on a
cadence, reusing the existing ingest/signals/embed functions rather than
reimplementing any of them.

    python -m src.scheduler.refresh --once           # one cycle, then exit (what a cron job runs)
    python -m src.scheduler.refresh                  # the loop, in the foreground
    python -m src.scheduler.refresh --status         # what the last cycle did
    python -m src.scheduler.refresh --once --backfill   # first-run: every week up to now

One cycle, in this order:
  (a) the shared, league-agnostic signals table for the current
      season/as-of-week -- computed ONCE per cycle, not per league
      (CLAUDE.md's key architectural principle: the signals table is
      NFL-wide, only roster/matchup/scoring are league-specific);
  (b) each ingested league's Sleeper pull (roster, matchups,
      transactions) via src.ingest.sleeper.run();
  (c) each league's Chroma collection re-embedded via src.rag.embed.embed()
      so the new signal chunks and the new roster chunks are retrievable.
Then the running API server is told its Chroma index changed (see
src/api/leagues.py's invalidate_chroma) and a status record is written.

**The as-of-date rule (CLAUDE.md, non-negotiable) decides the cadence's
target week, and nflverse -- not Sleeper -- is the authority for it.**
A signals table for as_of_week = N may only see plays from weeks
strictly before N (matchup_signals._history enforces that), so the
target week is `last fully-completed week + 1`: completeness is read off
nflverse's own schedules (`result` is null until a game is final), not
off Sleeper's display_week, which advances on its own clock and would
let a cycle compute week N while some of week N-1 was still being
played. Exposure the scheduler deliberately accepts: the upcoming-week
opponent and Vegas implied total for week N, both of which are published
before kickoff and are not results. See target_as_of_week().

**Cadence: every 6 hours by default**, from what nflverse's own build
schedule actually is (checked against the live repos, Sept 2026 -- see
TODO.md's Phase 5.7 entry for the raw evidence):
  - play-by-play rebuilds at 09:00 UTC daily plus five game-window runs
    a week (Fri 05:30 post-TNF, Sun 22:00, Mon 00:05, Mon 05:30
    post-SNF, Tue 05:30 post-MNF) -- at most two on any calendar day;
  - Next Gen Stats rebuilds once a day, 07:00 UTC;
  - schedules (the Vegas lines and game results this reads) rebuild
    every 5 minutes.
So polling more often than the slowest input changes is wasted work: six
hours picks up every pbp build within a few hours and still hits NGS's
once-a-day refresh. nflverse also folds in the NFL's stat corrections
between Monday and Wednesday, so a cycle recomputes the current target
week's table every time rather than skipping an already-written file --
the corrected numbers are the point.

Override with ASKMADDEN_REFRESH_INTERVAL_SECONDS (see
interval_seconds()); ASKMADDEN_REFRESH_ENABLED=0 turns the background
thread off entirely.

**Where it runs.** web/dev_server.py starts start_background_refresh()
in a daemon thread, which is the right shape for local dev and for one
long-running uvicorn process. Phase 5.6 (hosting) will need its own
equivalent and should NOT assume this thread: on a platform that scales
to more than one web process, or that sleeps idle ones, every replica
would run its own cycle, and a sleeping one would run none. `python -m
src.scheduler.refresh --once` is the entry point for whatever 5.6
actually picks (a cron job, a separate worker process, a scheduled cloud
function) -- it is idempotent, needs no database, and exits non-zero on
failure. Deliberately not over-built for a hosting setup that doesn't
exist yet: no distributed lock, no queue, no retry backoff beyond the
next cycle.

**Idempotency** (a scheduler calls these repeatedly, not once) was
checked before building on it, and all three steps fully overwrite
rather than append: sleeper.run() rewrites each JSON file in place
(one per week for matchups/transactions, so weeks accumulate as separate
files and none is ever double-written); save_signals_table() overwrites
`signals_{season}_week{N}.parquet`; embed.embed() deletes every existing
id in the collection and re-adds, by design ("the raw JSON is the source
of truth, so a full rebuild on each run is simpler than incremental
syncing"). tests/test_refresh.py pins this with real repeated runs.

**Not covered: src/ingest/realtime.py's "tighter cadence" tier.** The
spec pairs this module with a fast-refresh tier for injuries/weather.
Skipped on purpose, and not because the data is missing -- nflverse's
injury feed is live again (182 rows for 2026 as of 2026-09-13, despite
nflreadr's docs still saying the source died after 2024). It's that
nothing downstream consumes realtime.py yet: no signal in
matchup_signals.py reads it and no chunk in embed.py carries it, so
refreshing it would write data no report or answer can reach. Wiring
injuries into the signals table is signal-computation work, not
scheduling work. Tracked in TODO.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl

from src.ingest import nflverse, ngs, sleeper
from src.rag import embed
from src.signals import matchup_signals, player_stats

LOGGER = logging.getLogger("askmadden.refresh")

# Set for the duration of run_cycle() in this process. src/api/leagues.py's
# request-path resync checks it and stands down while a cycle is running,
# because the cycle re-embeds and stamps every ingested league itself -- a
# request rebuilding one in parallel is the same work twice, interleaved
# into the same collection, on the user's clock.
_cycle_running = threading.Event()
_cycle_started_at: str | None = None


def cycle_in_progress() -> bool:
    return _cycle_running.is_set()

SIGNALS_DIR = matchup_signals.PROCESSED_DIR
STATUS_PATH = SIGNALS_DIR.parent / "refresh_status.json"
# Every public function below resolves signals_dir/status_path from these
# module constants at CALL time rather than defaulting to them in its own
# signature: a signature default is bound once at import, so reassigning
# SIGNALS_DIR (tests do; a future caller might) would silently not apply
# and a cycle would write into the real data directory anyway. Found the
# hard way -- the CLI path did exactly that and a test leaked a fixture
# signals table into data/processed/signals/.
_UNSET = object()  # "argument omitted", as distinct from an explicit None

DEFAULT_INTERVAL_SECONDS = 6 * 60 * 60
MIN_INTERVAL_SECONDS = 60

# A week whose games all have a final result is complete. A week can also
# be declared complete once every one of its games is this far past its
# scheduled date without a result -- otherwise a single postponed or
# relocated game would pin the target week for the rest of the season and
# silently stop every later refresh from advancing.
COMPLETION_GRACE_DAYS = 7


def _now() -> datetime:
    return datetime.now(timezone.utc)


def interval_seconds() -> int:
    """Seconds between cycles: ASKMADDEN_REFRESH_INTERVAL_SECONDS, else
    six hours (see the module docstring for why six). Floored at a
    minute so a typo can't turn this into a busy loop against nflverse."""
    raw = (os.environ.get("ASKMADDEN_REFRESH_INTERVAL_SECONDS") or "").strip()
    if not raw:
        return DEFAULT_INTERVAL_SECONDS
    try:
        return max(MIN_INTERVAL_SECONDS, int(float(raw)))
    except ValueError:
        LOGGER.warning("ignoring unparseable ASKMADDEN_REFRESH_INTERVAL_SECONDS=%r", raw)
        return DEFAULT_INTERVAL_SECONDS


def refresh_enabled() -> bool:
    """Whether the background thread should run at all. On by default --
    a server whose data silently stops updating is the gap this module
    exists to close -- so opting out is explicit."""
    raw = (os.environ.get("ASKMADDEN_REFRESH_ENABLED") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


# ---- what week is it, and which weeks are actually finished ----


def completed_weeks(schedules: pl.DataFrame, season: int, now: datetime | None = None) -> set[int]:
    """Regular-season weeks of `season` that are finished: every game has
    a final result (nflverse's `result` is null until then), or is more
    than COMPLETION_GRACE_DAYS past its scheduled date -- see that
    constant for why the grace clause exists."""
    now = now or _now()
    cutoff = (now - timedelta(days=COMPLETION_GRACE_DAYS)).date()
    season_games = schedules.filter((pl.col("season") == season) & (pl.col("game_type") == matchup_signals.REG_SEASON_TYPE))
    done: set[int] = set()
    for week, games in season_games.group_by("week"):
        week = int(week[0] if isinstance(week, tuple) else week)
        if all(_game_settled(game, cutoff) for game in games.iter_rows(named=True)):
            done.add(week)
    return done


def _game_settled(game: dict, cutoff) -> bool:
    if game.get("result") is not None:
        return True
    gameday = game.get("gameday")
    if not gameday:
        return False
    try:
        return datetime.strptime(str(gameday), "%Y-%m-%d").date() < cutoff
    except ValueError:
        return False


def target_as_of_week(schedules: pl.DataFrame, season: int, now: datetime | None = None) -> int:
    """The week to compute signals for: one past the last fully-completed
    week, so the table sees every finished game and no unfinished one.

    A gap in the middle (week 3 complete, week 4 still being played, week
    5 somehow marked done) must NOT let the target jump past the
    unfinished week -- that would leak a partial week into the trailing
    history -- so this walks up from week 1 and stops at the first
    incomplete week rather than taking max(completed) + 1. Returns 1
    before any game of the season is final, which is correct and honest:
    the table is then all-nulls and Phase 3.6's prior-season fallback
    does the work.
    """
    done = completed_weeks(schedules, season, now)
    week = 1
    while week in done:
        week += 1
    return week


def resolve_season_and_week(
    season: int | None = None, as_of_week: int | None = None, schedules: pl.DataFrame | None = None
) -> tuple[int, int, pl.DataFrame, int]:
    """(season, as_of_week, schedules, completed_week_count) for this
    cycle. Either override may be passed explicitly (the CLI does, for
    recomputing a past week); anything not given comes from nflverse."""
    if season is None:
        season = nflverse.current_season()
    if schedules is None:
        schedules = nflverse.fetch_schedules(season)
    done = completed_weeks(schedules, season)
    if as_of_week is None:
        as_of_week = target_as_of_week(schedules, season)
    return season, as_of_week, schedules, len(done)


# ---- (a) the shared signals table ----


def refresh_signals(
    season: int,
    as_of_week: int,
    schedules: pl.DataFrame,
    signals_dir: Path | None = None,
    weeks: list[int] | None = None,
) -> dict:
    """Recompute and write the league-agnostic signals table(s), pulling
    nflverse pbp and NGS once and reusing them for every requested week.

    `weeks` defaults to just [as_of_week]; --backfill passes 1..as_of_week
    so a fresh deployment isn't missing the earlier weeks that
    ranking.load_signals_table() unions in when deciding whether a player
    has ANY current-season data.
    """
    signals_dir = signals_dir or SIGNALS_DIR
    weeks = weeks or [as_of_week]
    # Phase 6: the league-agnostic weekly stat lines this season and last
    # (the points proxy is computed from them per league at query time).
    # Cheap (one small download each) and rewritten every cycle so a
    # corrected box score reaches the ranking.
    stats_dir = Path(signals_dir).parent / "player_stats"  # data/processed/player_stats in production
    for stats_season in (season, season - 1):
        player_stats.write_weekly_stats(stats_season, stats_dir=stats_dir)
        LOGGER.info("player stats %s written -> %s", stats_season, stats_dir)
    pbp = nflverse.fetch_pbp(season)
    ngs_receiving = ngs.fetch_ngs(season, "receiving")
    ngs_rushing = ngs.fetch_ngs(season, "rushing")
    ngs_passing = ngs.fetch_ngs(season, "passing")

    written = []
    for week in weeks:
        rows = matchup_signals.build_signals_table(
            season,
            week,
            pbp,
            schedules,
            ngs_receiving=ngs_receiving,
            ngs_rushing=ngs_rushing,
            ngs_passing=ngs_passing,
        )
        path = matchup_signals.save_signals_table(rows, season, week, out_dir=signals_dir)
        LOGGER.info("signals: wrote %d rows for %s week %d -> %s", len(rows), season, week, path)
        written.append({"season": season, "as_of_week": week, "rows": len(rows), "path": str(path)})
    return {"status": "ok", "tables": written}


# ---- (b) + (c) per-league Sleeper pull and re-embed ----


def refresh_league(
    league_id: str,
    raw_dir: Path,
    persist_dir: Path,
    signals_dir: Path | None = None,
    week: int | None = None,
) -> dict:
    """One league's Sleeper pull followed by a full re-embed of its Chroma
    collection, then tell a running server to re-read that index.

    `week` is left as None on purpose: sleeper.run() then uses Sleeper's
    own current week (its `week` field -- not `display_week`, which lags
    until midweek; see sleeper.current_week()) for the matchup/transaction
    files, which is the right week for LEAGUE data (whose matchup is live
    right now) even when it differs from the signals table's
    nflverse-derived as-of-week.
    The two are different questions and deliberately answered by
    different authorities.
    """
    signals_dir = signals_dir or SIGNALS_DIR
    result: dict[str, Any] = {"league_id": league_id, "sleeper": "skipped", "embed": "skipped", "error": None}
    try:
        written = sleeper.run(league_id=str(league_id), week=week, out_dir=raw_dir)
        result["sleeper"] = "ok"
        result["files"] = len(written)
        LOGGER.info("league %s: refreshed %d Sleeper file(s) -> %s", league_id, len(written), raw_dir)
    except Exception as e:  # network, a deleted league, a Sleeper outage
        result["sleeper"] = "error"
        result["error"] = f"sleeper: {type(e).__name__}: {e}"
        LOGGER.warning("league %s: Sleeper refresh failed: %s", league_id, e)
        return result

    try:
        collection = embed.embed(
            persist_dir=persist_dir,
            raw_dir=raw_dir,
            signals_dir=signals_dir if signals_dir.exists() else None,
        )
        result["embed"] = "ok"
        result["chunks"] = collection.count()
        LOGGER.info("league %s: re-embedded %d chunk(s) -> %s", league_id, result["chunks"], persist_dir)
    except Exception as e:
        result["embed"] = "error"
        result["error"] = f"embed: {type(e).__name__}: {e}"
        LOGGER.warning("league %s: re-embed failed: %s", league_id, e)
        return result

    _note_reembedded(persist_dir, signals_dir)
    return result


def _note_reembedded(persist_dir: Path, signals_dir: Path) -> None:
    """Tell the API layer this league's collection was just rebuilt, on
    both of the axes it tracks separately:

    - invalidate_chroma() points a SAME-PROCESS server's cached Chroma
      client at the rewritten index. (An out-of-process run needs
      nothing here -- warm_chroma()'s own (mtime, size) stamp catches
      the change on that server's next request.)
    - record_signals_fingerprint() updates the on-disk stamp that
      ensure_league_data() compares against to decide whether a league
      needs re-embedding at all. Skipping it would leave the stamp
      describing the PREVIOUS signals table, so the first request after
      every cycle would redundantly rebuild an already-current
      collection -- reproduced before this call existed.

    Imported lazily so a cron-only run never pays for importing the API
    layer, and never fatal: the worst case without it is the redundant
    rebuild above, not a wrong answer."""
    try:
        from src.api import leagues

        leagues.invalidate_chroma(persist_dir)
        leagues.record_signals_fingerprint(persist_dir, signals_dir)
    except Exception as e:  # pragma: no cover - defensive
        LOGGER.debug("could not tell the API layer about the re-embed of %s: %s", persist_dir, e)


def _league_targets(league_ids: list[str] | None) -> list[tuple[str, Path, Path]]:
    from src.api import leagues

    ids = league_ids if league_ids is not None else leagues.ingested_league_ids()
    return [(league_id, *leagues.league_dirs(league_id)) for league_id in ids]


# ---- one cycle ----


def run_cycle(
    season: int | None = None,
    as_of_week: int | None = None,
    signals_dir: Path | None = None,
    league_ids: list[str] | None = None,
    backfill: bool = False,
    status_path: Any = _UNSET,  # a Path, or None for "don't write a status file"
) -> dict:
    """One full refresh: shared signals once, then every ingested
    league's Sleeper data and Chroma index. Never raises -- a cycle that
    fails records the failure and returns, so the loop survives a
    transient nflverse or Sleeper outage. The returned record is also
    what gets written to the status file; `outcome` is "ok", "partial"
    (signals fine, at least one league failed) or "error" (no signals
    computed, so nothing downstream could have changed)."""
    signals_dir = signals_dir or SIGNALS_DIR
    status_path = STATUS_PATH if status_path is _UNSET else status_path
    global _cycle_started_at
    _cycle_running.set()
    _cycle_started_at = _now().isoformat()
    try:
        return _run_cycle(season, as_of_week, signals_dir, league_ids, backfill, status_path)
    finally:
        _cycle_running.clear()
        _cycle_started_at = None


def _run_cycle(season, as_of_week, signals_dir, league_ids, backfill, status_path) -> dict:
    started = _now()
    record: dict[str, Any] = {
        "started_at": started.isoformat(),
        "outcome": "error",
        "season": season,
        "as_of_week": as_of_week,
        "completed_weeks": None,
        "signals": None,
        "leagues": [],
        "error": None,
    }

    try:
        season, as_of_week, schedules, done = resolve_season_and_week(season, as_of_week)
        record.update(season=season, as_of_week=as_of_week, completed_weeks=done)
        weeks = list(range(1, as_of_week + 1)) if backfill else [as_of_week]
        LOGGER.info(
            "cycle start: season %s, as-of-week %d (%d completed week(s)), signal week(s) %s",
            season, as_of_week, done, weeks,
        )
        record["signals"] = refresh_signals(season, as_of_week, schedules, signals_dir=signals_dir, weeks=weeks)
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        record["signals"] = {"status": "error", "tables": []}
        LOGGER.error("cycle failed before any league refresh: %s", record["error"])
        _finish(record, started, status_path)
        return record

    try:
        targets = _league_targets(league_ids)
    except Exception as e:  # pragma: no cover - defensive
        targets = []
        record["error"] = f"could not enumerate leagues: {type(e).__name__}: {e}"
    if not targets:
        LOGGER.info("no ingested leagues to refresh yet (a league is ingested the first time a session selects it)")

    for league_id, raw_dir, persist_dir in targets:
        record["leagues"].append(refresh_league(league_id, raw_dir, persist_dir, signals_dir=signals_dir))

    failed = [entry for entry in record["leagues"] if entry["error"]]
    # record["error"] may already be set if enumerating the leagues itself
    # failed -- that is a partial cycle too, not a clean one.
    record["outcome"] = "partial" if (failed or record["error"]) else "ok"
    if failed and not record["error"]:
        record["error"] = f"{len(failed)} of {len(record['leagues'])} league(s) failed to refresh"
    _finish(record, started, status_path)
    return record


def _finish(record: dict, started: datetime, status_path: Path | None) -> None:
    finished = _now()
    record["finished_at"] = finished.isoformat()
    record["duration_seconds"] = round((finished - started).total_seconds(), 1)
    LOGGER.info(
        "cycle %s in %ss (season %s week %s, %d league(s))",
        record["outcome"], record["duration_seconds"], record["season"], record["as_of_week"], len(record["leagues"]),
    )
    if status_path is not None:
        write_status(record, status_path)


# ---- status visibility ----


def write_status(record: dict, status_path: Path | None = None) -> Path:
    """Record the cycle so "is the refresh actually working" is one `cat`
    away (or `--status`). Keeps a consecutive-failure count across cycles,
    which is the number that actually tells you something is wrong --
    one failed cycle is a blip, six in a row is an outage. Written via a
    temp file + os.replace so a reader never catches a half-written
    file."""
    status_path = status_path or STATUS_PATH
    previous = read_status(status_path) or {}
    failures = int(previous.get("consecutive_failures") or 0)
    failures = 0 if record.get("outcome") == "ok" else failures + 1
    interval = interval_seconds()
    payload = {
        "last_run": record,
        "consecutive_failures": failures,
        "interval_seconds": interval,
        "next_run_after": (_now() + timedelta(seconds=interval)).isoformat(),
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = status_path.with_suffix(status_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, status_path)
    return status_path


def read_status(status_path: Path | None = None) -> dict | None:
    status_path = status_path or STATUS_PATH
    try:
        return json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ---- the loop ----


def seconds_until_due(status_path: Path | None = None, now: datetime | None = None) -> float:
    """How long until the next cycle is due per the status file's
    `next_run_after` -- 0 when there is no status file, it is unreadable,
    or the time has passed. A freshly (re)started process uses this so a
    deploy does not rerun a cycle that finished twenty minutes ago: every
    deploy was pinning the machine's CPU for minutes, right when someone
    was most likely to be trying the site (2026-09-22: a login took 16s
    and a first-time league open several minutes, on top of the boot
    cycle)."""
    status = read_status(status_path)
    if not status or not status.get("next_run_after"):
        return 0.0
    try:
        due = datetime.fromisoformat(status["next_run_after"])
    except ValueError:
        return 0.0
    if due.tzinfo is None:
        due = due.replace(tzinfo=timezone.utc)
    return max(0.0, (due - (now or _now())).total_seconds())


def cycle_started_at() -> str | None:
    return _cycle_started_at


def start_background_refresh(
    signals_dir: Path | None = None,
    interval: int | None = None,
    run_immediately: bool = True,
    status_path: Path | None = None,
) -> tuple[threading.Thread, threading.Event] | None:
    """Run cycles in a daemon thread until the returned Event is set.
    Returns None (having logged why) when ASKMADDEN_REFRESH_ENABLED is
    off, so a caller can just treat None as "not running".

    Daemon so it can never hold up interpreter shutdown, and the Event so
    a clean shutdown (dev_server's lifespan) stops it between cycles
    rather than mid-write. The first cycle runs immediately by default:
    a server that has just started is exactly when the data is most
    likely to be stale. It is a full pipeline run (an nflverse pbp pull
    and a re-embed per league), so it is backgrounded, not awaited --
    startup is never blocked on it.
    """
    if not refresh_enabled():
        LOGGER.info("background refresh disabled (ASKMADDEN_REFRESH_ENABLED)")
        return None

    stop = threading.Event()
    seconds = interval if interval is not None else interval_seconds()

    def _loop() -> None:
        LOGGER.info("background refresh started: every %ds", seconds)
        try:
            # Linux schedules threads individually, so the refresh can run at
            # low priority and request threads win the CPU while a cycle is
            # re-embedding (the site felt slow during every cycle before).
            # Not supported everywhere; a failure just means normal priority.
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
        except (OSError, AttributeError):
            pass
        if not run_immediately:
            stop.wait(seconds)
        else:
            # "Immediately" means "as soon as it is due": a restart right
            # after a finished cycle waits for the next scheduled one.
            delay = seconds_until_due(status_path)
            if delay > 0:
                LOGGER.info("last cycle is recent; next one due in %ds, not rerunning on start", int(delay))
                stop.wait(delay)
        while not stop.is_set():
            try:
                run_cycle(signals_dir=signals_dir)
            except Exception as e:  # pragma: no cover - run_cycle already traps its own failures
                LOGGER.exception("refresh cycle raised unexpectedly: %s", e)
            stop.wait(seconds)
        LOGGER.info("background refresh stopped")

    thread = threading.Thread(target=_loop, name="askmadden-refresh", daemon=True)
    thread.start()
    return thread, stop


# ---- CLI ----


def _print_status(status_path: Path) -> int:
    status = read_status(status_path)
    if status is None:
        print(f"no refresh has run yet (nothing at {status_path})")
        return 1
    run = status.get("last_run") or {}
    print(f"last run:   {run.get('started_at')} -> {run.get('finished_at')} ({run.get('duration_seconds')}s)")
    print(f"outcome:    {run.get('outcome')}  (consecutive failures: {status.get('consecutive_failures')})")
    print(f"season:     {run.get('season')}, as-of-week {run.get('as_of_week')} "
          f"({run.get('completed_weeks')} completed week(s))")
    for table in ((run.get("signals") or {}).get("tables") or []):
        print(f"  signals:  week {table['as_of_week']}, {table['rows']} rows -> {table['path']}")
    for league in run.get("leagues") or []:
        detail = f"sleeper={league['sleeper']} embed={league['embed']}"
        if league.get("chunks") is not None:
            detail += f" chunks={league['chunks']}"
        print(f"  league {league['league_id']}: {detail}" + (f"  ERROR {league['error']}" if league.get("error") else ""))
    if run.get("error"):
        print(f"error:      {run['error']}")
    print(f"next run:   after {status.get('next_run_after')} (every {status.get('interval_seconds')}s)")
    return 0 if run.get("outcome") == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh nflverse signals + every ingested league's Sleeper data and RAG index")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit (what a cron job should call)")
    parser.add_argument("--status", action="store_true", help="print the last cycle's outcome and exit")
    parser.add_argument("--season", type=int, default=None, help="default: nflverse's latest season")
    parser.add_argument("--as-of-week", type=int, default=None,
                        help="default: one past the last fully-completed week (see target_as_of_week)")
    parser.add_argument("--backfill", action="store_true",
                        help="also compute every earlier week of the season, not just the target week")
    parser.add_argument("--league-id", action="append", dest="league_ids", default=None,
                        help="refresh only this league (repeatable); default is every ingested league")
    parser.add_argument("--interval-seconds", type=int, default=None,
                        help=f"loop interval; default ASKMADDEN_REFRESH_INTERVAL_SECONDS or {DEFAULT_INTERVAL_SECONDS}")
    parser.add_argument("--status-path", type=Path, default=None, help=f"default {STATUS_PATH}")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.status:
        return _print_status(args.status_path or STATUS_PATH)

    if args.once:
        record = run_cycle(
            season=args.season,
            as_of_week=args.as_of_week,
            league_ids=args.league_ids,
            backfill=args.backfill,
            # `or STATUS_PATH`, not the bare argument: run_cycle reads an
            # explicit None as "don't write a status file", and the CLI
            # always wants one written.
            status_path=args.status_path or STATUS_PATH,
        )
        return 0 if record["outcome"] == "ok" else 1

    started = start_background_refresh(interval=args.interval_seconds)
    if started is None:
        print("refresh is disabled via ASKMADDEN_REFRESH_ENABLED; nothing to run")
        return 1
    thread, stop = started
    try:
        while thread.is_alive():
            thread.join(timeout=1)
    except KeyboardInterrupt:
        stop.set()
        thread.join(timeout=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
