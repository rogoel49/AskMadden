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


# ---- query_player_signal_with_fallback: prior-season fallback (Phase 3.6) ----

_CHRISTIAN_PRIOR_SEASON = {**_CHRISTIAN, "season": 2023, "as_of_week": 19, "epa_trend": -0.5}


def test_fallback_returns_none_not_stale_when_player_has_no_data_at_all(tmp_path):
    persist_dir = _seeded_collection(tmp_path, [_LUKE])  # some other player, not this one

    chunk, stale = retrieve.query_player_signal_with_fallback(
        "00-0099999", season=2024, as_of_week=8, persist_dir=persist_dir
    )

    assert chunk is None
    assert stale is False


def test_fallback_uses_prior_season_when_current_season_has_nothing(tmp_path):
    persist_dir = _seeded_collection(tmp_path, [_CHRISTIAN_PRIOR_SEASON])  # only 2023 data on record

    chunk, stale = retrieve.query_player_signal_with_fallback(
        "00-0033280", season=2024, as_of_week=8, persist_dir=persist_dir
    )

    assert stale is True
    assert chunk["metadata"]["season"] == 2023
    assert chunk["metadata"]["player_id"] == "00-0033280"


def test_fallback_never_used_when_current_season_data_exists(tmp_path):
    persist_dir = _seeded_collection(tmp_path, [_CHRISTIAN, _CHRISTIAN_PRIOR_SEASON])

    chunk, stale = retrieve.query_player_signal_with_fallback(
        "00-0033280", season=2024, as_of_week=8, persist_dir=persist_dir
    )

    assert stale is False
    assert chunk["metadata"]["season"] == 2024


def test_fallback_never_leaks_a_later_current_season_week(tmp_path):
    """The same-season check must be bounded to week <= as_of_week -- a
    later week's chunk existing locally (e.g. computed for a different
    purpose) must never leak into an as-of-week-filtered lookup."""
    later_week = {**_CHRISTIAN, "as_of_week": 12}  # later than the as_of_week=8 requested below
    persist_dir = _seeded_collection(tmp_path, [later_week, _CHRISTIAN_PRIOR_SEASON])

    chunk, stale = retrieve.query_player_signal_with_fallback(
        "00-0033280", season=2024, as_of_week=8, persist_dir=persist_dir
    )

    # No week-8-or-earlier chunk for 2024 exists, so this must fall back
    # to the prior season, never pick up week 12 as if it were valid data
    # for an as-of-week-8 lookup.
    assert stale is True
    assert chunk["metadata"]["season"] == 2023
