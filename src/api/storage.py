"""Phase 5.2 storage: SQLite, deliberately tiny.

What it holds (see PROJECT_SPEC.md's Phase 5.2):
  - users:          Sleeper username -> user_id / display name
  - leagues:        username -> the leagues Sleeper listed for them, with
                    the user's own roster_id in each
  - sessions:       an opaque session_id -> username + which league (and
                    which roster_id within it) is currently active
  - query_counts:   per-username, per-UTC-day count of Claude-backed
                    queries, for the daily cap (cost control)

Everything here is rebuildable from Sleeper plus a fresh login, so the
schema is created with CREATE TABLE IF NOT EXISTS on open and there is
no migration framework -- a portfolio deliverable at friend-group
scale. Ingested league data (rosters, matchups, the Chroma index) does
NOT live here; that stays in the per-league data directories
src/api/leagues.py manages.
"""
from __future__ import annotations

import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "askmadden.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username     TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    display_name TEXT,
    updated_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS leagues (
    username  TEXT NOT NULL,
    league_id TEXT NOT NULL,
    name      TEXT,
    season    TEXT,
    roster_id TEXT,
    PRIMARY KEY (username, league_id)
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    username   TEXT NOT NULL,
    league_id  TEXT,
    roster_id  TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS query_counts (
    username TEXT NOT NULL,
    day      TEXT NOT NULL,
    count    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (username, day)
);
"""


class QueryCapExceeded(RuntimeError):
    def __init__(self, username: str, cap: int, used: int):
        super().__init__(f"{username!r} has used {used} of {cap} Claude-backed queries today; try again tomorrow (UTC).")
        self.username, self.cap, self.used = username, cap, used


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _norm_username(username: str) -> str:
    return (username or "").strip().lower()


class Storage:
    def __init__(self, path: Path | str = DEFAULT_DB_PATH):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- users + leagues ----

    def save_user_leagues(self, user: dict, leagues: list[dict]) -> None:
        """Store (replace) what Sleeper reported for this user: the user
        record and the full list of their leagues with their roster_id
        in each. Called after every successful login so the stored list
        never goes stale relative to Sleeper."""
        username = _norm_username(user["username"])
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO users(username, user_id, display_name, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(username) DO UPDATE SET user_id=excluded.user_id, "
                "display_name=excluded.display_name, updated_at=excluded.updated_at",
                (username, str(user["user_id"]), user.get("display_name"), _now()),
            )
            conn.execute("DELETE FROM leagues WHERE username = ?", (username,))
            conn.executemany(
                "INSERT INTO leagues(username, league_id, name, season, roster_id) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        username,
                        str(league["league_id"]),
                        league.get("name"),
                        str(league["season"]) if league.get("season") is not None else None,
                        str(league["roster_id"]) if league.get("roster_id") is not None else None,
                    )
                    for league in leagues
                ],
            )

    def user(self, username: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (_norm_username(username),)).fetchone()
        return dict(row) if row else None

    def leagues(self, username: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT league_id, name, season, roster_id FROM leagues WHERE username = ? ORDER BY name",
                (_norm_username(username),),
            ).fetchall()
        return [dict(row) for row in rows]

    def league(self, username: str, league_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT league_id, name, season, roster_id FROM leagues WHERE username = ? AND league_id = ?",
                (_norm_username(username), str(league_id)),
            ).fetchone()
        return dict(row) if row else None

    # ---- sessions ----

    def create_session(self, username: str, league_id: str | None = None, roster_id: Any | None = None) -> dict:
        """A new opaque session for a known user, optionally with its
        active league/roster set immediately. The user must already be
        stored (i.e. have logged in via save_user_leagues)."""
        username = _norm_username(username)
        if self.user(username) is None:
            raise KeyError(f"unknown user {username!r} -- log in first")
        session_id = secrets.token_urlsafe(24)
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions(session_id, username, league_id, roster_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    username,
                    str(league_id) if league_id is not None else None,
                    str(roster_id) if roster_id is not None else None,
                    now,
                    now,
                ),
            )
        return self.session(session_id)  # type: ignore[return-value]

    def set_active_league(self, session_id: str, league_id: str, roster_id: Any | None = None) -> dict:
        """Switch the session's active league. The league must be one
        Sleeper listed for this session's user (stored by
        save_user_leagues) -- a session can never be pointed at a league
        its user isn't in. roster_id defaults to the stored one for that
        user in that league."""
        session = self.session(session_id)
        if session is None:
            raise KeyError(f"unknown session {session_id!r}")
        league = self.league(session["username"], league_id)
        if league is None:
            raise KeyError(f"league {league_id!r} is not one of {session['username']!r}'s leagues")
        if roster_id is None:
            roster_id = league.get("roster_id")
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET league_id = ?, roster_id = ?, updated_at = ? WHERE session_id = ?",
                (str(league_id), str(roster_id) if roster_id is not None else None, _now(), session_id),
            )
        return self.session(session_id)  # type: ignore[return-value]

    def session(self, session_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    # ---- per-user/day query cap ----

    def queries_today(self, username: str, day: str | None = None) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT count FROM query_counts WHERE username = ? AND day = ?",
                (_norm_username(username), day or _today()),
            ).fetchone()
        return int(row["count"]) if row else 0

    def record_query(self, username: str, cap: int, day: str | None = None) -> int:
        """Count one Claude-backed query against username's daily cap and
        return the new total. Raises QueryCapExceeded (and records
        nothing) if the cap is already reached -- check-and-increment in
        one statement so two concurrent requests can't both slip under
        the cap."""
        username = _norm_username(username)
        day = day or _today()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO query_counts(username, day, count) VALUES (?, ?, 0)", (username, day)
            )
            cursor = conn.execute(
                "UPDATE query_counts SET count = count + 1 WHERE username = ? AND day = ? AND count < ?",
                (username, day, cap),
            )
            if cursor.rowcount == 0:
                used = conn.execute(
                    "SELECT count FROM query_counts WHERE username = ? AND day = ?", (username, day)
                ).fetchone()["count"]
                raise QueryCapExceeded(username, cap, int(used))
            return int(
                conn.execute(
                    "SELECT count FROM query_counts WHERE username = ? AND day = ?", (username, day)
                ).fetchone()["count"]
            )
