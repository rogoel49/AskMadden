"""The early-season EPA trend, in prose: a null trend with no baseline
must read as "no trend yet", never as "trending down (+0.00)" (what
every card on the Feed said the morning after week 1, 2026), and an
exactly-zero trend, should one ever occur, is "flat"."""
import pytest

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


# ---- passers get a score of their own (2026-09-20) ----


def test_a_passer_row_is_rankable_on_cpoe_and_implied_total():
    from src.reasoning import ranking

    qb = {"epa_trend": None, "epa_baseline_plays": 0, "red_zone_share": None, "target_share": None,
          "target_share_adjusted": None, "cpoe": 10.0, "implied_total": 25.0}
    assert ranking.opportunity_score(qb) == pytest.approx(0.05 * 10.0 + 0.04 * 25.0)
    assert "completion % over expected +10.0" in ranking.fmt_signal_row(qb)
    # the same row without the passer stat is what a QB used to be: unrankable
    assert ranking.opportunity_score({**qb, "cpoe": None}) is None


def test_two_passers_rank_by_the_better_environment_and_accuracy():
    from src.reasoning import ranking

    maye = {"player_id": "m", "name": "Drake Maye", "position": "QB", "team": "NE",
            "row": {"cpoe": 3.2, "implied_total": 21.0, "epa_trend": None, "epa_baseline_plays": 0}}
    mayfield = {"player_id": "b", "name": "Baker Mayfield", "position": "QB", "team": "TB",
                "row": {"cpoe": 15.4, "implied_total": 24.5, "epa_trend": None, "epa_baseline_plays": 0}}
    out = ranking.rank_candidates([maye, mayfield])
    assert out["verdict"] == "clear"
    assert [r["name"] for r in out["ranked"]] == ["Baker Mayfield", "Drake Maye"]
    assert "CPOE" in out["score_description"]


# ---- the trend term needs a sample (2026-09-20) ----


def test_a_trend_on_a_handful_of_plays_does_not_count_toward_the_score():
    """Tahj Washington, the first real waiver report's #1: 5 plays all of
    2025, +1.35 EPA/play, 1% target share, 1% red-zone share -> 2.74 under
    the old score, above every player with a real role."""
    row = {"season_plays": 5, "epa_trend": 1.3476, "red_zone_share": 0.0074, "target_share": 0.0108, "target_share_adjusted": None}
    assert ranking.opportunity_score(row) == pytest.approx(3.0 * 0.0074 + 2.0 * 0.0108)
    assert "on only 5 plays (too few to count)" in ranking.fmt_signal_row(row)
    assert "trending up" not in ranking.fmt_signal_row(row)
    # the same trend with a real sample behind it counts as before
    assert ranking.opportunity_score({**row, "season_plays": 40}) == pytest.approx(2.0 * 1.3476 + 3.0 * 0.0074 + 2.0 * 0.0108)
    assert "trending up" in ranking.fmt_signal_row({**row, "season_plays": 40})
