import json
from pathlib import Path

from src.rag import embed, retrieve


def _write(raw_dir: Path, filename: str, data) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / filename).write_text(json.dumps({"fetched_at": "2026-01-01T00:00:00Z", "source": "sleeper", "data": data}))


def _seed_raw_dir(raw_dir: Path) -> None:
    _write(
        raw_dir,
        "league.json",
        {
            "name": "Victorious Secret 3.0",
            "season": "2026",
            "settings": {"num_teams": 12, "waiver_budget": 1000, "playoff_teams": 8, "playoff_week_start": 15},
        },
    )
    _write(
        raw_dir,
        "teams.json",
        [
            {
                "roster_id": 1,
                "owner_id": "u1",
                "display_name": "rogoel49",
                "team_name": "Victorious Secret",
                "players": ["4046", "5850"],
                "starters": ["4046"],
                "settings": {"wins": 2},
            },
            {
                "roster_id": 2,
                "owner_id": "u2",
                "display_name": "otheruser",
                "team_name": "Team Two",
                "players": ["4029", "5945", "4217"],
                "starters": ["5945"],
                "settings": {"wins": 1},
            },
        ],
    )
    _write(
        raw_dir,
        "players.json",
        {
            "4046": {"full_name": "Patrick Mahomes", "position": "QB", "team": "KC"},
            "5850": {"full_name": "Cooper Kupp", "position": "WR", "team": "LAR"},
            "4029": {"full_name": "Christian McCaffrey", "position": "RB", "team": "SF"},
            "5945": {"full_name": "Brock Purdy", "position": "QB", "team": "SF"},
            "4217": {"full_name": "George Kittle", "position": "TE", "team": "SF"},
        },
    )
    _write(raw_dir, "matchups_week_3.json", [{"roster_id": 1, "matchup_id": 1, "points": 110.5}, {"roster_id": 2, "matchup_id": 1, "points": 98.2}])
    _write(raw_dir, "transactions_week_3.json", [{"type": "waiver", "adds": {"5850": 1}, "drops": {"4046": 1}}])


def test_build_chunks_covers_league_teams_matchups_and_transactions(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_raw_dir(raw_dir)

    chunks = embed.build_chunks(raw_dir)

    ids = {c["id"] for c in chunks}
    assert "league:settings" in ids
    assert "team:1" in ids
    assert "matchup:week3:1" in ids
    assert "transaction:week3:0" in ids

    team_chunk = next(c for c in chunks if c["id"] == "team:1")
    assert "Patrick Mahomes (QB, KC)" in team_chunk["text"]
    assert "Cooper Kupp (WR, LAR)" in team_chunk["text"]
    assert team_chunk["metadata"] == {"type": "team_roster", "roster_id": 1}

    txn_chunk = next(c for c in chunks if c["id"] == "transaction:week3:0")
    assert "added: Cooper Kupp (WR, LAR)" in txn_chunk["text"]
    assert "dropped: Patrick Mahomes (QB, KC)" in txn_chunk["text"]


def test_embed_and_query_roundtrip(tmp_path):
    raw_dir = tmp_path / "sleeper"
    persist_dir = tmp_path / "chroma"
    _seed_raw_dir(raw_dir)

    chunks = embed.build_chunks(raw_dir)
    embed.embed(chunks, persist_dir=persist_dir)

    results = retrieve.query("who is on the Victorious Secret roster", n_results=1, persist_dir=persist_dir)

    assert len(results) == 1
    assert results[0]["id"] == "team:1"
    assert "Mahomes" in results[0]["text"]


def test_embed_is_idempotent_on_rebuild(tmp_path):
    raw_dir = tmp_path / "sleeper"
    persist_dir = tmp_path / "chroma"
    _seed_raw_dir(raw_dir)
    chunks = embed.build_chunks(raw_dir)

    collection = embed.embed(chunks, persist_dir=persist_dir)
    assert collection.count() == len(chunks)

    collection = embed.embed(chunks, persist_dir=persist_dir)
    assert collection.count() == len(chunks)
