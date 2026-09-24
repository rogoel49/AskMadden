"""Usage events and the private /stats page (2026-09-24)."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.api import main
from src.api.storage import Storage
from tests.test_api_main import _login_and_session, _script_grounded_answer, _SEASON, _WEEK, VS30_ID, api  # noqa: F401


def test_record_event_and_usage_stats(tmp_path):
    st = Storage(tmp_path / "db.sqlite3")
    st.save_user_leagues({"user_id": "1", "username": "rohangoel", "display_name": "R"}, [])
    st.record_event("chat", "rohangoel", "9", {"question": "who do I start?"})
    st.record_event("ui:tab", "rohangoel", "9", {"tab": "feed"})
    st.record_event("report", None, None)
    out = st.usage_stats()
    assert {r["kind"]: r["n"] for r in out["events_by_kind"]} == {"chat": 1, "ui:tab": 1, "report": 1}
    assert json.loads(out["recent_questions"][0]["detail"])["question"] == "who do I start?"
    assert out["users"][0]["username"] == "rohangoel"


def test_api_logs_league_open_reports_and_chat_questions(api):
    session_id = _login_and_session(api)
    api["client"].get(f"/api/reports/drop?session_id={session_id}")
    _script_grounded_answer(api["claude"])
    api["client"].post("/api/chat", json={"session_id": session_id, "question": "start Barkley?", "season": _SEASON, "as_of_week": _WEEK})
    kinds = {r["kind"]: r["n"] for r in api["storage"].usage_stats()["events_by_kind"]}
    assert kinds["league_open"] == 1 and kinds["report"] == 1 and kinds["chat"] == 1
    q = json.loads(api["storage"].usage_stats()["recent_questions"][0]["detail"])
    assert q["question"] == "start Barkley?" and "get_my_roster" in q["tools"] and q["turn"] == 1


def test_ui_events_need_a_real_session(api):
    session_id = _login_and_session(api)
    assert api["client"].post("/api/events", json={"session_id": session_id, "kind": "tab", "detail": {"tab": "roster"}}).json() == {"ok": True}
    assert api["client"].post("/api/events", json={"session_id": "nope", "kind": "tab"}).json() == {"ok": False}
    kinds = {r["kind"]: r["n"] for r in api["storage"].usage_stats()["events_by_kind"]}
    assert kinds.get("ui:tab") == 1


def test_stats_page_is_gated_by_the_key(api, monkeypatch):
    monkeypatch.delenv("ASKMADDEN_STATS_KEY", raising=False)
    assert api["client"].get("/stats").status_code == 404  # unset: the page does not exist
    monkeypatch.setenv("ASKMADDEN_STATS_KEY", "s3cret")
    assert api["client"].get("/stats").status_code == 404
    assert api["client"].get("/stats?key=wrong").status_code == 404
    _login_and_session(api)
    page = api["client"].get("/stats?key=s3cret")
    assert page.status_code == 200 and "Ask Madden usage" in page.text and "rogoel49" in page.text
