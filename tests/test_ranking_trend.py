"""The early-season EPA trend, in prose: a null trend with no baseline
must read as "no trend yet", never as "trending down (+0.00)" (what
every card on the Feed said the morning after week 1, 2026), and an
exactly-zero trend, should one ever occur, is "flat"."""
from src.rag import embed
from src.reasoning import ranking

_EARLY = {"player_id": "x", "epa_trend": None, "epa_baseline_plays": 0, "red_zone_share": 0.4, "target_share": 0.2, "season_plays": 30}
_FLAT = {**_EARLY, "epa_trend": 0.0, "epa_baseline_plays": 12}
_LATE_ABSENT = {**_EARLY, "epa_baseline_plays": 12}  # no plays in the trailing window: the pre-existing null


def test_fmt_signal_row_says_no_trend_yet_instead_of_trending_down():
    text = ranking.fmt_signal_row(_EARLY)
    assert ranking.NO_TREND_YET in text
    assert "trending" not in text
    assert "red zone role share 40%" in text  # the other signals still print


def test_fmt_signal_row_calls_an_exact_zero_flat_not_down():
    assert "efficiency trending flat (+0.00 EPA/play)" in ranking.fmt_signal_row(_FLAT)


def test_fmt_signal_row_stays_silent_for_the_other_kind_of_null():
    text = ranking.fmt_signal_row(_LATE_ABSENT)
    assert "trend" not in text


def test_weakness_reasons_never_blame_a_missing_or_zero_trend():
    assert not [r for r in ranking.weakness_reasons(_EARLY) if "efficiency" in r]
    assert not [r for r in ranking.weakness_reasons(_FLAT) if "efficiency" in r]


def test_opportunity_score_ignores_the_null_trend_but_still_ranks():
    assert ranking.opportunity_score(_EARLY) == ranking.opportunity_score({**_EARLY, "epa_trend": None})
    assert ranking.opportunity_score(_EARLY) is not None


def test_signal_chunk_sentence_states_no_trend_yet_and_flat():
    early = embed._signal_sentence({**_EARLY, "as_of_week": 2})
    assert "no efficiency trend yet" in early and "trending" not in early
    assert "trending flat" in embed._signal_sentence({**_FLAT, "as_of_week": 2})
