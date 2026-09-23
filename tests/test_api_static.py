"""Phase 5.6: the API serves the frontend itself -- one app, one origin.

Everything Phase 5.3/5.4 validated about the frontend assumed the URL
layout web/dev_server.py gave it (/ui/ prefix, relative asset hrefs,
service-worker scope = /ui/). These tests pin that layout now that
src/api/main.py owns it, so a refactor can't quietly move the page to a
path where sw.js would start intercepting /api/, or leave a manifest
href pointing at a file that isn't served.
"""
from __future__ import annotations

import json
import re
import threading

import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.scheduler import refresh


@pytest.fixture
def client(monkeypatch):
    # No lifespan (plain TestClient, not `with`), so no refresh thread and
    # no .env load -- these tests are about routing only.
    return TestClient(main.app)


def test_root_redirects_to_the_ui_index(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/ui/"


def test_ui_index_serves_the_frontend_html(client):
    resp = client.get("/ui/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert 'id="view-landing"' in resp.text
    # The same bytes are reachable by filename -- the manifest's start_url
    # and the service worker's precache list both name the file directly.
    by_name = client.get(f"/ui/{main.UI_FILENAME}")
    assert by_name.status_code == 200
    assert by_name.text == resp.text


def test_ui_without_trailing_slash_lands_on_the_index(client):
    resp = client.get("/ui", follow_redirects=True)
    assert resp.status_code == 200
    assert 'id="view-landing"' in resp.text


@pytest.mark.parametrize(
    "path,content_type",
    [
        ("/ui/manifest.json", "application/json"),
        ("/ui/sw.js", "javascript"),
        ("/ui/icons/icon-192.png", "image/png"),
        ("/ui/icons/icon-512.png", "image/png"),
        ("/ui/icons/apple-touch-icon.png", "image/png"),
    ],
)
def test_pwa_assets_are_served_next_to_the_page(client, path, content_type):
    resp = client.get(path)
    assert resp.status_code == 200, path
    assert content_type in resp.headers["content-type"], (path, resp.headers["content-type"])


def test_every_relative_asset_the_page_references_is_actually_served(client):
    """The HTML's manifest/icon/sw hrefs are relative on purpose (they must
    survive being served from any prefix). That only helps if each one
    resolves to a real file under the mount."""
    html = client.get("/ui/").text
    head = html.split("</head>", 1)[0]
    hrefs = set(re.findall(r'href="([^"#]+)"', head))
    hrefs |= set(re.findall(r"serviceWorker\.register\('([^']+)'\)", html))
    relative = {h for h in hrefs if not h.startswith(("http://", "https://", "/"))}
    assert {"manifest.json", "sw.js"} <= relative, relative  # the two the install path depends on
    for href in sorted(relative):
        assert client.get(f"/ui/{href}").status_code == 200, href


def test_manifest_and_service_worker_stay_inside_the_ui_prefix(client):
    """Why /ui/ and not /: the service worker's scope is the directory it
    is served from. Under / it would intercept /api/*; under /ui/ it can't.
    manifest.json's own scope/start_url are relative for the same reason."""
    manifest = client.get("/ui/manifest.json").json()
    assert not manifest["start_url"].startswith("/")
    assert not manifest["scope"].startswith("/")
    assert main.UI_PREFIX != "/" and main.UI_INDEX_PATH.endswith("/")
    sw = client.get("/ui/sw.js").text
    assert "self.registration.scope" in sw  # the guard that keeps /api/ out


def test_api_routes_are_not_shadowed_by_the_mount(client):
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.get("/api/sessions/nope").json()["detail"].startswith("unknown session")
    assert client.get("/docs").status_code == 200
    assert client.get("/ui/does-not-exist.txt").status_code == 404


# ---- /api/health ----


def test_health_is_ok_even_before_any_refresh_has_run(client, monkeypatch, tmp_path):
    monkeypatch.setattr(refresh, "STATUS_PATH", tmp_path / "refresh_status.json")
    monkeypatch.setattr(main, "_refresh_thread", None)
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["version"] == main.app.version
    assert body["refresh"]["last_outcome"] is None
    assert body["refresh"]["running_in_process"] is False


def test_health_reports_the_last_refresh_cycle_without_failing_on_a_bad_one(client, monkeypatch, tmp_path):
    status_path = tmp_path / "refresh_status.json"
    monkeypatch.setattr(refresh, "STATUS_PATH", status_path)
    status_path.write_text(json.dumps({
        "last_run": {"outcome": "error", "finished_at": "2026-09-18T10:00:00+00:00", "season": 2026, "as_of_week": 3},
        "consecutive_failures": 4,
        "interval_seconds": 21600,
        "next_run_after": "2026-09-18T16:00:00+00:00",
    }))
    resp = client.get("/api/health")
    assert resp.status_code == 200  # the process is fine; the data source isn't -- that's reported, not fatal
    body = resp.json()["refresh"]
    assert body["last_outcome"] == "error"
    assert body["consecutive_failures"] == 4
    assert (body["season"], body["as_of_week"]) == (2026, 3)
    assert body["next_run_after"] == "2026-09-18T16:00:00+00:00"


# ---- lifespan: the refresh thread now belongs to the API app ----


def test_lifespan_starts_the_refresh_and_stops_it_on_shutdown(monkeypatch):
    stop = threading.Event()
    thread = threading.Thread(target=stop.wait, daemon=True)
    thread.start()
    calls = []

    def fake_start(*args, **kwargs):
        calls.append(1)
        return thread, stop

    monkeypatch.setattr(refresh, "start_background_refresh", fake_start)
    with TestClient(main.app) as client:
        assert calls == [1]
        assert client.get("/api/health").json()["refresh"]["running_in_process"] is True
        assert not stop.is_set()
    assert stop.is_set()
    thread.join(timeout=2)
    assert main._refresh_thread is None


def test_lifespan_respects_the_refresh_off_switch(monkeypatch):
    monkeypatch.setenv("ASKMADDEN_REFRESH_ENABLED", "0")
    with TestClient(main.app) as client:
        body = client.get("/api/health").json()["refresh"]
        assert body["enabled"] is False
        assert body["running_in_process"] is False


# ---- Phase 5.5: the landing view is a front door, nothing more ----


def test_landing_view_makes_no_api_calls(client):
    """5.5's checklist: no auth, no API calls from the landing view. Its
    only behavior is navigation into the login view; the first network
    call the page makes is the login form's POST /api/leagues."""
    html = client.get("/ui/").text
    landing = html.split('id="view-landing"', 1)[1].split('id="view-login"', 1)[0]
    assert "api(" not in landing and "fetch(" not in landing
    handlers = set(re.findall(r'onclick="([^"]+)"', landing))
    assert handlers and all(h.startswith("goView('login')") for h in handlers), handlers


def test_remember_me_only_stores_the_username_and_league(client):
    """The page may keep exactly one thing in localStorage: {username,
    league_id}. No session id, no tokens, nothing else."""
    html = client.get("/ui/").text
    assert "localStorage.setItem(REMEMBER_KEY, JSON.stringify({ username, league_id: leagueId }))" in html
    assert html.count("localStorage.setItem(") == 1
    assert "restoreSession()" in html and "function logout()" in html
