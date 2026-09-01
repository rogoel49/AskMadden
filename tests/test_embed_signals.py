import polars as pl

from src.rag import embed

_SIGNAL_ROW_FULL = {
    "player_id": "RB1",
    "player_name": "RB One",
    "team": "A",
    "season": 2024,
    "as_of_week": 6,
    "epa_trend": 0.35,
    "red_zone_share": 0.5,
    "target_share": None,
    "opponent": "B",
    "run_funnel_rate_vs_avg": 0.1,
    "implied_total": 24.5,
    "adot": None,
    "ryoe_per_att": 1.2,
    "cpoe": None,
}

_SIGNAL_ROW_SPARSE = {
    "player_id": "WR9",
    "player_name": "WR Nine",
    "team": "C",
    "season": 2024,
    "as_of_week": 6,
    "epa_trend": None,
    "red_zone_share": None,
    "target_share": None,
    "opponent": None,
    "run_funnel_rate_vs_avg": None,
    "implied_total": None,
    "adot": None,
    "ryoe_per_att": None,
    "cpoe": None,
}


def test_build_signal_chunks_includes_available_signals():
    chunks = embed.build_signal_chunks([_SIGNAL_ROW_FULL])

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["id"] == "signal:2024:week6:RB1"
    assert chunk["metadata"] == {"type": "player_signal", "player_id": "RB1", "week": 6, "season": 2024}
    assert "RB One (A)" in chunk["text"]
    assert "trending up" in chunk["text"]
    assert "red zone role share 50.00%" in chunk["text"]
    assert "rush yards over expected 1.20/attempt" in chunk["text"]
    # target_share is None on this row -- must not be fabricated as a fact.
    assert "target share" not in chunk["text"]


def test_build_signal_chunks_handles_no_signals_available():
    chunks = embed.build_signal_chunks([_SIGNAL_ROW_SPARSE])

    assert "no signals available yet this season" in chunks[0]["text"]


def test_load_signal_chunks_reads_every_parquet_in_dir(tmp_path):
    pl.DataFrame([_SIGNAL_ROW_FULL]).write_parquet(tmp_path / "signals_2024_week5.parquet")
    pl.DataFrame([_SIGNAL_ROW_SPARSE]).write_parquet(tmp_path / "signals_2024_week6.parquet")

    chunks = embed.load_signal_chunks(tmp_path)

    ids = {c["id"] for c in chunks}
    assert ids == {"signal:2024:week6:RB1", "signal:2024:week6:WR9"}


def test_embed_includes_signal_chunks_when_signals_dir_given(tmp_path):
    raw_dir = tmp_path / "sleeper"
    persist_dir = tmp_path / "chroma"
    signals_dir = tmp_path / "signals"
    signals_dir.mkdir()
    pl.DataFrame([_SIGNAL_ROW_FULL]).write_parquet(signals_dir / "signals_2024_week6.parquet")

    from tests.test_embed import _seed_raw_dir

    _seed_raw_dir(raw_dir)
    league_chunks = embed.build_chunks(raw_dir)

    collection = embed.embed(persist_dir=persist_dir, raw_dir=raw_dir, signals_dir=signals_dir)

    assert collection.count() == len(league_chunks) + 1
    assert "signal:2024:week6:RB1" in collection.get()["ids"]
