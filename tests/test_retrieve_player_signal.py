from src.rag import embed, retrieve

# Same shape as the real bug report: two players who share a surname get
# near-identical chunk text ("entering 2024 week 8: recent efficiency
# trending up...") that would be easy for embedding similarity to
# confuse. query_player_signal must never rely on that similarity -- it
# resolves by exact player_id metadata match instead.
_CHRISTIAN = {
    "player_id": "00-0033280",
    "player_name": "C.McCaffrey",
    "team": "SF",
    "season": 2024,
    "as_of_week": 8,
    "epa_trend": 0.2,
    "red_zone_share": 0.4,
    "target_share": 0.15,
    "opponent": "SEA",
    "run_funnel_rate_vs_avg": 0.05,
    "implied_total": 24.0,
    "adot": None,
    "ryoe_per_att": 1.1,
    "cpoe": None,
}
_LUKE = {
    "player_id": "00-0039355",
    "player_name": "L.McCaffrey",
    "team": "WAS",
    "season": 2024,
    "as_of_week": 8,
    "epa_trend": 0.2,
    "red_zone_share": 0.4,
    "target_share": 0.15,
    "opponent": "NYG",
    "run_funnel_rate_vs_avg": 0.05,
    "implied_total": 24.0,
    "adot": None,
    "ryoe_per_att": None,
    "cpoe": None,
}
_CHRISTIAN_WEEK9 = {**_CHRISTIAN, "as_of_week": 9, "epa_trend": 0.4}


def _seeded_collection(tmp_path, rows):
    persist_dir = tmp_path / "chroma"
    chunks = embed.build_signal_chunks(rows)
    embed.embed(chunks, persist_dir=persist_dir)
    return persist_dir


def test_query_player_signal_never_returns_the_wrong_mccaffrey(tmp_path):
    persist_dir = _seeded_collection(tmp_path, [_CHRISTIAN, _LUKE])

    result = retrieve.query_player_signal("00-0033280", persist_dir=persist_dir)

    assert result is not None
    assert result["id"] == "signal:2024:week8:00-0033280"
    assert result["metadata"]["player_id"] == "00-0033280"
    assert "L.McCaffrey" not in result["text"]


def test_query_player_signal_resolves_luke_independently(tmp_path):
    persist_dir = _seeded_collection(tmp_path, [_CHRISTIAN, _LUKE])

    result = retrieve.query_player_signal("00-0039355", persist_dir=persist_dir)

    assert result["metadata"]["player_id"] == "00-0039355"
    assert result["metadata"]["player_id"] != "00-0033280"


def test_query_player_signal_defaults_to_most_recent_week(tmp_path):
    persist_dir = _seeded_collection(tmp_path, [_CHRISTIAN, _CHRISTIAN_WEEK9])

    result = retrieve.query_player_signal("00-0033280", persist_dir=persist_dir)

    assert result["metadata"]["week"] == 9


def test_query_player_signal_can_pin_a_specific_week(tmp_path):
    persist_dir = _seeded_collection(tmp_path, [_CHRISTIAN, _CHRISTIAN_WEEK9])

    result = retrieve.query_player_signal("00-0033280", season=2024, as_of_week=8, persist_dir=persist_dir)

    assert result["metadata"]["week"] == 8


def test_query_player_signal_returns_none_when_player_has_no_chunk(tmp_path):
    persist_dir = _seeded_collection(tmp_path, [_CHRISTIAN])

    result = retrieve.query_player_signal("00-0099999", persist_dir=persist_dir)

    assert result is None
