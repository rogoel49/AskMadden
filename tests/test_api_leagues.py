"""src/api/leagues.py: per-league data directories + ingest-on-first-use.
The ingest itself (Sleeper + nflverse, network) is mocked; what's tested
is the routing (flat CLI dirs for SLEEPER_LEAGUE_ID, per-league dirs
for everyone else) and that ingest runs exactly when data is missing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.api import leagues
from src.rag.embed import CHROMA_DIR, RAW_DIR
from src.reasoning.league import LeagueMismatchError

VS30 = "1389341490030862336"


def _write_league_json(raw_dir: Path, league_id: str, name: str = "Some League") -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "league.json").write_text(
        json.dumps({"data": {"league_id": league_id, "name": name, "scoring_settings": {"rec": 1.0}}})
    )


@pytest.fixture
def per_league_root(tmp_path, monkeypatch):
    monkeypatch.setattr(leagues, "LEAGUES_DIR", tmp_path / "leagues")
    monkeypatch.delenv("SLEEPER_LEAGUE_ID", raising=False)
    return tmp_path / "leagues"


def test_the_developers_own_league_keeps_the_flat_cli_directories(per_league_root, monkeypatch):
    monkeypatch.setenv("SLEEPER_LEAGUE_ID", VS30)
    assert leagues.league_dirs(VS30) == (RAW_DIR, CHROMA_DIR)


def test_every_other_league_gets_its_own_directory(per_league_root):
    raw_dir, persist_dir = leagues.league_dirs("555")
    assert raw_dir == per_league_root / "555" / "sleeper"
    assert persist_dir == per_league_root / "555" / "chroma"


def test_ensure_league_data_ingests_when_nothing_is_on_disk(per_league_root, monkeypatch):
    calls = []

    def fake_ingest(league_id, week=None):
        raw_dir, persist_dir = leagues.league_dirs(league_id)
        _write_league_json(raw_dir, league_id, "Friends PPR")
        persist_dir.mkdir(parents=True)
        calls.append(league_id)
        return raw_dir, persist_dir

    monkeypatch.setattr(leagues, "ingest_league", fake_ingest)

    config = leagues.ensure_league_data("555")

    assert calls == ["555"]
    assert config.league_id == "555" and config.name == "Friends PPR"
    assert config.scoring_settings == {"rec": 1.0}
    assert config.raw_dir == per_league_root / "555" / "sleeper"


def test_ensure_league_data_does_not_re_ingest_when_data_exists(per_league_root, monkeypatch):
    raw_dir, persist_dir = leagues.league_dirs("555")
    _write_league_json(raw_dir, "555")
    persist_dir.mkdir(parents=True)
    monkeypatch.setattr(leagues, "ingest_league", lambda *a, **kw: pytest.fail("ingest must not run"))

    assert leagues.ensure_league_data("555").league_id == "555"


def test_ensure_league_data_refresh_forces_ingest(per_league_root, monkeypatch):
    raw_dir, persist_dir = leagues.league_dirs("555")
    _write_league_json(raw_dir, "555")
    persist_dir.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(leagues, "ingest_league", lambda league_id, week=None: calls.append(league_id) or (raw_dir, persist_dir))

    leagues.ensure_league_data("555", refresh=True)

    assert calls == ["555"]


def test_ensure_league_data_refuses_a_directory_holding_another_league(per_league_root, monkeypatch):
    raw_dir, persist_dir = leagues.league_dirs("555")
    _write_league_json(raw_dir, "999")  # wrong league on disk
    persist_dir.mkdir(parents=True)
    monkeypatch.setattr(leagues, "ingest_league", lambda *a, **kw: pytest.fail("ingest must not run"))

    with pytest.raises(LeagueMismatchError):
        leagues.ensure_league_data("555")


def test_ingest_league_uses_the_existing_ingest_and_embed_pointed_at_the_league_dirs(per_league_root, monkeypatch):
    seen = {}
    monkeypatch.setattr(leagues.sleeper, "run", lambda league_id, week, out_dir: seen.update(run=(league_id, week, out_dir)))
    monkeypatch.setattr(
        leagues.embed, "embed", lambda persist_dir, raw_dir, signals_dir: seen.update(embed=(persist_dir, raw_dir, signals_dir))
    )

    raw_dir, persist_dir = leagues.ingest_league("555", week=3)

    assert seen["run"] == ("555", 3, per_league_root / "555" / "sleeper")
    assert seen["embed"][:2] == (per_league_root / "555" / "chroma", per_league_root / "555" / "sleeper")
    assert (raw_dir, persist_dir) == (per_league_root / "555" / "sleeper", per_league_root / "555" / "chroma")
