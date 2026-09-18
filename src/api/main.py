"""Phase 5.2: the FastAPI app -- thin HTTP endpoints over the existing
reasoning layer. Nothing here reasons about football; every answer
comes from recommend() / generate_report() exactly as the CLI gets it.

    uvicorn src.api.main:app --reload            # API + the UI at http://127.0.0.1:8000/
    python -m web.dev_server                     # same app, bound to 0.0.0.0 for a phone on the LAN

Phase 5.6: this one app is the whole deployment. It serves the responsive
frontend (design/askmadden-ui-mockup.html, its manifest, service worker
and icons) as a static mount at /ui and redirects / there, so the
browser's fetch() calls to /api/* are same-origin and there is no CORS
configuration anywhere. The frontend keeps its /ui prefix on purpose:
the service worker's scope is the directory it is served from, so
/api/ stays outside it and is never intercepted or cached (see
design/sw.js). web/dev_server.py used to compose this mount around the
API from the outside (5.3 was scoped not to touch src/api/); it is now a
thin launcher over this module.

The Phase 5.7 background refresh (src/scheduler/refresh.py) also starts
from this app's lifespan, on by default -- a server whose data quietly
stops updating is the failure that module exists to prevent. Set
ASKMADDEN_REFRESH_ENABLED=0 on a host that runs more than one replica or
sleeps idle ones, and run `python -m src.scheduler.refresh --once` from a
cron job / scheduled worker there instead (refresh.py's docstring says
why an in-process thread is the wrong shape for that).

Endpoints (all JSON):
  GET  /api/health
        -> {ok: true, ...} plus the refresh's last-cycle summary when one
           has run. For a host's health check and for "is the data
           moving" at a glance; always 200 while the process is up
           (stale data is reported, not treated as an outage).
  POST /api/leagues            {username, season?}
        -> log in: Sleeper username -> user + their leagues (with the
           user's roster_id in each), stored. Live Sleeper call.
  GET  /api/leagues/{username}
        -> the stored league list for a user who has logged in.
  POST /api/sessions           {username, league_id, roster_id?}
        -> pick the active league; ingests that league's data on first
           use (Sleeper + nflverse, network). Returns a session_id.
  GET  /api/sessions/{session_id}
  GET  /api/roster?session_id=
        -> the session's own roster, grouped by position.
  GET  /api/reports/{report_type}?session_id=[&season=&as_of_week=]
        -> generate_report() output, verbatim (no Claude call, not
           counted against the query cap).
  POST /api/chat               {session_id, question, messages?}
        -> recommend(), one Claude-backed turn; pass back the returned
           `messages` for multi-turn (Phase 3.5's existing support --
           the API only exposes the parameter). Counted against the
           per-user daily cap (ASKMADDEN_DAILY_QUERY_CAP, default 25).

**data_gaps and stale markers are passed through as their own fields,
never collapsed.** Phase 3.6 (stale/source_season/source_as_of_week)
and Phase 3.7 (data_gaps with reason no_signal_data |
out_of_scope_capability) are three distinct facts a UI must render
distinctly (TODO.md's 5.3 plan says so explicitly). Shapes, as they
actually are in the reasoning layer today (Phase 5.2's investigation):
  - recommend(): `data_gaps` is a top-level list of
    {reason, detail, player_name?} straight from the model's
    submit_recommendation call. Stale/has_signals markers are NOT
    top-level -- they live on each get_player_signals tool result
    inside `tool_calls` ({has_signals, stale, source_season,
    source_as_of_week}). /api/chat therefore returns `data_gaps`
    verbatim AND lifts every resolved get_player_signals result into
    `signals_consulted`, each entry carrying those exact fields
    unmodified -- a projection of real tool output, not a
    reconstruction.
  - generate_report(): every entry already carries {stale,
    source_season, source_as_of_week} and the report carries `notes`;
    there is no data_gaps concept in reports (their gaps are `notes`).
    /api/reports returns the report dict verbatim.

**Whose roster.** The session's roster_id is passed explicitly into
recommend()/generate_report() (Phase 5.2 threads it down to
src/rag/lookup.py); the server never relies on the MY_ROSTER_ID
environment variable, which stays a CLI-only convenience.

**.env** is loaded once at startup (recommend.load_dotenv_once()), not
per request -- see that function for the investigation note.

**Validation status.** Sleeper's API and the Claude API are both
unreachable from the sandbox this was built in, so tests/test_api_main.py
drives every endpoint through FastAPI's TestClient with the Sleeper
login mocked at the auth-module boundary, league ingest mocked to seed
fixture data, and a scripted fake Claude client -- the reasoning layer
and its real tools run for real against real-shaped fixture data. The
live username -> leagues -> ingest -> chat flow still needs one real
run on a machine with network access; see TODO.md's Phase 5.2 entry.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import requests
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.api import auth, leagues
from src.api.storage import DEFAULT_DB_PATH, QueryCapExceeded, Storage
from src.rag import lookup
from src.reasoning import recommend, report
from src.reasoning.league import LeagueMismatchError
from src.scheduler import refresh

DEFAULT_DAILY_QUERY_CAP = 25
# The shared, league-agnostic signals table every league's reports rank
# from (see src/api/leagues.py) -- module-level so tests can point it at
# fixture data.
SIGNALS_DIR = report.SIGNALS_DIR

# Phase 5.6: the one responsive frontend, served by this app. The file
# keeps its name (every doc references it) and its /ui prefix (see the
# module docstring: the service worker's scope must not include /api/).
DESIGN_DIR = Path(__file__).resolve().parents[2] / "design"
UI_PREFIX = "/ui"
UI_FILENAME = "askmadden-ui-mockup.html"
UI_INDEX_PATH = f"{UI_PREFIX}/"

# The background refresh started by the lifespan, kept so /api/health can
# say whether it is running in this process; None when disabled or before
# startup.
_refresh_thread = None


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _refresh_thread
    recommend.load_dotenv_once()  # once per server process, not per request; before refresh reads its env
    started = refresh.start_background_refresh()  # None when ASKMADDEN_REFRESH_ENABLED is off
    _refresh_thread = started[0] if started is not None else None
    try:
        yield
    finally:
        if started is not None:
            _thread, stop = started
            stop.set()  # let the loop end between cycles rather than mid-write
        _refresh_thread = None


app = FastAPI(title="Ask Madden API", version="0.5.6", lifespan=_lifespan)


# ---- dependencies (overridable in tests) ----

_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = Storage(os.environ.get("ASKMADDEN_DB", str(DEFAULT_DB_PATH)))
    return _storage


def daily_query_cap() -> int:
    return int(os.environ.get("ASKMADDEN_DAILY_QUERY_CAP", DEFAULT_DAILY_QUERY_CAP))


def anthropic_client() -> Any:
    """None means recommend() builds the real anthropic.Anthropic() from
    ANTHROPIC_API_KEY; tests override this to inject a fake client."""
    return None


# ---- request/response models ----


class LoginRequest(BaseModel):
    username: str
    season: str | int | None = None


class SessionRequest(BaseModel):
    username: str
    league_id: str
    roster_id: str | int | None = None
    refresh: bool = Field(default=False, description="re-ingest the league's data even if it's already on disk")


class ChatRequest(BaseModel):
    session_id: str
    question: str
    messages: list[dict] | None = Field(
        default=None,
        description="prior turns exactly as returned in a previous /api/chat response's `messages`, for multi-turn",
    )
    season: int | None = None
    as_of_week: int | None = None


# ---- helpers ----


def _session_or_404(storage: Storage, session_id: str) -> dict:
    session = storage.session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session {session_id!r}")
    if not session.get("league_id"):
        raise HTTPException(status_code=409, detail="session has no active league -- POST /api/sessions first")
    return session


def _league_for_session(session: dict):
    try:
        return leagues.ensure_league_data(session["league_id"])
    except LeagueMismatchError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


def _jsonable(obj: Any) -> Any:
    """recommend()'s `messages` contain Anthropic SDK content blocks
    (pydantic models) for assistant turns; convert them to plain dicts so
    the API can return them and accept them back unchanged (the SDK
    accepts dict-shaped blocks in `messages`)."""
    if hasattr(obj, "model_dump"):
        return _jsonable(obj.model_dump(exclude_none=True))
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _signals_consulted(tool_calls: list[dict]) -> list[dict]:
    """Every resolved get_player_signals result the agent actually saw,
    with its stale/has_signals markers exactly as the tool returned them
    -- the per-player counterpart to the top-level data_gaps list."""
    consulted = []
    for call in tool_calls:
        if call.get("name") != "get_player_signals":
            continue
        result = call.get("result") or {}
        if not result.get("resolved"):
            continue
        consulted.append(
            {
                "player_id": result.get("player_id"),
                "player_name": result.get("player_name"),
                "position": result.get("position"),
                "team": result.get("team"),
                "has_signals": result.get("has_signals"),
                "stale": result.get("stale"),
                "source_season": result.get("source_season"),
                "source_as_of_week": result.get("source_as_of_week"),
            }
        )
    return consulted


# ---- endpoints ----


@app.get("/api/health")
def health() -> dict:
    """Process is up -> 200, always. The refresh summary is informational:
    a host's health check must not restart a perfectly good server because
    nflverse was down for a cycle."""
    status = refresh.read_status() or {}
    last = status.get("last_run") or {}
    return {
        "ok": True,
        "version": app.version,
        "refresh": {
            "enabled": refresh.refresh_enabled(),
            "running_in_process": _refresh_thread is not None and _refresh_thread.is_alive(),
            "last_outcome": last.get("outcome"),
            "last_finished_at": last.get("finished_at"),
            "season": last.get("season"),
            "as_of_week": last.get("as_of_week"),
            "consecutive_failures": status.get("consecutive_failures"),
            "next_run_after": status.get("next_run_after"),
        },
    }


# ---- the frontend (Phase 5.6) ----
# Order matters: FastAPI matches routes in registration order, so the
# explicit /ui/ index route must exist before the /ui mount or the
# StaticFiles app would answer /ui/ with a 404 (no index.html there).


@app.get("/", include_in_schema=False)
def _root() -> RedirectResponse:
    return RedirectResponse(UI_INDEX_PATH)


@app.get(UI_INDEX_PATH, include_in_schema=False)
def _ui_index() -> FileResponse:
    return FileResponse(DESIGN_DIR / UI_FILENAME, media_type="text/html")


app.mount(UI_PREFIX, StaticFiles(directory=str(DESIGN_DIR)), name="ui")


@app.post("/api/leagues")
def login(body: LoginRequest, storage: Storage = Depends(get_storage)) -> dict:
    try:
        resolved = auth.resolve_user_leagues(body.username, season=body.season)
    except auth.UnknownSleeperUser as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Sleeper API unreachable: {e}") from e
    storage.save_user_leagues(resolved["user"], resolved["leagues"])
    return {"user": resolved["user"], "leagues": storage.leagues(resolved["user"]["username"])}


@app.get("/api/leagues/{username}")
def stored_leagues(username: str, storage: Storage = Depends(get_storage)) -> dict:
    user = storage.user(username)
    if user is None:
        raise HTTPException(status_code=404, detail=f"{username!r} hasn't logged in -- POST /api/leagues first")
    return {"user": user, "leagues": storage.leagues(username)}


@app.post("/api/sessions")
def create_session(body: SessionRequest, storage: Storage = Depends(get_storage)) -> dict:
    league = storage.league(body.username, body.league_id)
    if league is None:
        raise HTTPException(
            status_code=404, detail=f"league {body.league_id!r} is not one of {body.username!r}'s stored leagues"
        )
    roster_id = body.roster_id if body.roster_id is not None else league.get("roster_id")
    if roster_id is None:
        raise HTTPException(
            status_code=409,
            detail=f"{body.username!r} has no roster in league {body.league_id!r} (commissioner-only?); pass roster_id",
        )
    try:
        config = leagues.ensure_league_data(body.league_id, refresh=body.refresh)
    except LeagueMismatchError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except requests.RequestException as e:  # first-use ingest hits Sleeper (and nflverse)
        raise HTTPException(status_code=502, detail=f"upstream data source unreachable while ingesting league: {e}") from e
    session = storage.create_session(body.username, league_id=body.league_id, roster_id=roster_id)
    return {**session, "league_name": config.name, "scoring_settings": config.scoring_settings}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, storage: Storage = Depends(get_storage)) -> dict:
    return _session_or_404(storage, session_id)


@app.get("/api/roster")
def roster(session_id: str = Query(...), storage: Storage = Depends(get_storage)) -> dict:
    session = _session_or_404(storage, session_id)
    config = _league_for_session(session)
    team = lookup.team_roster_for_roster_id(session["roster_id"], config.raw_dir)
    if team is None:
        raise HTTPException(
            status_code=404, detail=f"roster {session['roster_id']!r} not found in league {config.league_id!r}"
        )
    return {"league_id": config.league_id, "league_name": config.name, **team}


@app.get("/api/reports/{report_type}")
def get_report(
    report_type: str,
    session_id: str = Query(...),
    season: int | None = None,
    as_of_week: int | None = None,
    storage: Storage = Depends(get_storage),
) -> dict:
    session = _session_or_404(storage, session_id)
    config = _league_for_session(session)
    try:
        return report.generate_report(
            report_type,
            config.league_id,
            raw_dir=config.raw_dir,
            persist_dir=config.persist_dir,
            season=season,
            as_of_week=as_of_week,
            signals_dir=SIGNALS_DIR,
            roster_id=session["roster_id"],
        )
    except ValueError as e:  # unknown report_type
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:  # e.g. week not ingested, roster not found
        raise HTTPException(status_code=409, detail=str(e)) from e


@app.post("/api/chat")
def chat(
    body: ChatRequest,
    storage: Storage = Depends(get_storage),
    cap: int = Depends(daily_query_cap),
    client: Any = Depends(anthropic_client),
) -> dict:
    session = _session_or_404(storage, body.session_id)
    config = _league_for_session(session)
    try:
        used = storage.record_query(session["username"], cap)
    except QueryCapExceeded as e:
        raise HTTPException(status_code=429, detail=str(e)) from e

    result = recommend.recommend(
        body.question,
        config.league_id,
        messages=body.messages,
        raw_dir=config.raw_dir,
        persist_dir=config.persist_dir,
        season=body.season,
        as_of_week=body.as_of_week,
        client=client,
        roster_id=session["roster_id"],
    )
    return {
        "recommendation": result["recommendation"],
        "reasoning": result["reasoning"],
        "player_id": result["player_id"],
        # Phase 3.7, verbatim: [{reason, detail, player_name?}] -- [] means fully grounded.
        "data_gaps": result["data_gaps"],
        # Phase 3.6, per player the agent looked up: has_signals / stale / source_season / source_as_of_week.
        "signals_consulted": _signals_consulted(result["tool_calls"]),
        "error": result["error"],
        "league_id": result["league_id"],
        "roster_id": result["roster_id"],
        "messages": _jsonable(result["messages"]),
        "queries_used_today": used,
        "daily_query_cap": cap,
    }
