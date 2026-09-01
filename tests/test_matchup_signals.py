import polars as pl
import pytest

from src.signals import matchup_signals as ms

_DEFAULTS = {
    "season": 2024,
    "season_type": "REG",
    "week": 1,
    "posteam": None,
    "defteam": None,
    "rush": 0,
    "pass": 0,
    "yards_gained": 0.0,
    "yardline_100": 50.0,
    "epa": 0.0,
    "rusher_player_id": None,
    "rusher_player_name": None,
    "receiver_player_id": None,
    "receiver_player_name": None,
    "passer_player_id": None,
    "passer_player_name": None,
}


def _play(**kwargs) -> dict:
    return {**_DEFAULTS, **kwargs}


def _sample_pbp() -> pl.DataFrame:
    rows = [
        # Week 1: A's offense (RB1 rush + WR1 target) vs B's defense
        _play(week=1, posteam="A", defteam="B", rush=1, rusher_player_id="RB1", rusher_player_name="RB One",
              yards_gained=5, yardline_100=15, epa=0.5),
        _play(week=1, posteam="A", defteam="B", **{"pass": 1}, receiver_player_id="WR1", receiver_player_name="WR One",
              yards_gained=10, yardline_100=50, epa=0.3),
        # Week 1: B's offense (RB2 rush only) vs A's defense
        _play(week=1, posteam="B", defteam="A", rush=1, rusher_player_id="RB2", rusher_player_name="RB Two",
              yards_gained=3, yardline_100=80, epa=-0.1),
        # Week 1: C's offense (RB3 rush) vs D's defense
        _play(week=1, posteam="C", defteam="D", rush=1, rusher_player_id="RB3", rusher_player_name="RB Three",
              yards_gained=20, yardline_100=10, epa=1.0),
        # Week 1: D's offense (WR2 target only) vs C's defense
        _play(week=1, posteam="D", defteam="C", **{"pass": 1}, receiver_player_id="WR2", receiver_player_name="WR Two",
              yards_gained=8, yardline_100=60, epa=0.2),
        # Week 1: E's offense (WR3 target, on bye in week 3) vs Z's defense
        _play(week=1, posteam="E", defteam="Z", **{"pass": 1}, receiver_player_id="WR3", receiver_player_name="WR Three",
              yards_gained=6, yardline_100=45, epa=0.1),
        # Week 2: same matchups, different production (to create a trend)
        _play(week=2, posteam="A", defteam="B", rush=1, rusher_player_id="RB1", rusher_player_name="RB One",
              yards_gained=2, yardline_100=18, epa=-0.5),
        _play(week=2, posteam="A", defteam="B", **{"pass": 1}, receiver_player_id="WR1", receiver_player_name="WR One",
              yards_gained=15, yardline_100=5, epa=0.7),
        _play(week=2, posteam="B", defteam="A", rush=1, rusher_player_id="RB2", rusher_player_name="RB Two",
              yards_gained=4, yardline_100=70, epa=0.0),
        _play(week=2, posteam="C", defteam="D", rush=1, rusher_player_id="RB3", rusher_player_name="RB Three",
              yards_gained=1, yardline_100=90, epa=-0.2),
        _play(week=2, posteam="D", defteam="C", **{"pass": 1}, receiver_player_id="WR2", receiver_player_name="WR Two",
              yards_gained=5, yardline_100=40, epa=0.1),
        # Week 3 (postseason): must never leak into as_of_week=3 history
        _play(week=3, season_type="POST", posteam="A", defteam="B", rush=1, rusher_player_id="RB1",
              rusher_player_name="RB One", yards_gained=999, yardline_100=1, epa=99),
        # Week 3 (future reg season week): must never leak into as_of_week=3 history
        _play(week=3, posteam="A", defteam="B", rush=1, rusher_player_id="RB1", rusher_player_name="RB One",
              yards_gained=-999, yardline_100=99, epa=-99),
        # Traded player: TR1 played for A in week 1, then for C in week 2
        # (simulates a mid-season trade, e.g. Davante Adams' LV->NYJ move).
        # Kept out of the red zone (yardline_100=50) so it doesn't perturb
        # the red-zone-share assertions above.
        _play(week=1, posteam="A", defteam="B", rush=1, rusher_player_id="TR1", rusher_player_name="Traded One",
              yards_gained=4, yardline_100=50, epa=0.1),
        _play(week=2, posteam="C", defteam="D", rush=1, rusher_player_id="TR1", rusher_player_name="Traded One",
              yards_gained=6, yardline_100=50, epa=0.2),
        # Name-formatting inconsistency: same player, same team, two
        # different name spellings across weeks in nflverse's raw pbp
        # (e.g. "Di.Johnson" vs "Dio.Johnson" for Diontae Johnson).
        _play(week=1, posteam="A", defteam="B", rush=1, rusher_player_id="NM1", rusher_player_name="N.Mixed",
              yards_gained=2, yardline_100=50, epa=0.05),
        _play(week=2, posteam="A", defteam="B", rush=1, rusher_player_id="NM1", rusher_player_name="Na.Mixed",
              yards_gained=3, yardline_100=50, epa=0.15),
    ]
    return pl.DataFrame(rows)


def _sample_schedules() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"season": 2024, "week": 3, "home_team": "A", "away_team": "C", "spread_line": 3.0, "total_line": 45.0},
            {"season": 2024, "week": 3, "home_team": "B", "away_team": "D", "spread_line": -2.0, "total_line": 40.0},
        ]
    )


def test_history_excludes_future_and_postseason_weeks():
    hist = ms._history(_sample_pbp(), as_of_week=3)
    assert hist["week"].max() == 2
    assert set(hist["season_type"].unique().to_list()) == {"REG"}


def test_defense_run_funnel_rate_flags_run_and_pass_funnel_defenses():
    result = ms.defense_run_funnel_rate(_sample_pbp(), as_of_week=3)
    by_team = {row["team"]: row for row in result.to_dicts()}

    # defteam=A only ever allowed rushing (from B's offense) -> pure run funnel
    assert by_team["A"]["run_funnel_rate"] == pytest.approx(1.0)
    # defteam=C only ever allowed passing (from D's offense) -> pure pass funnel
    assert by_team["C"]["run_funnel_rate"] == pytest.approx(0.0)
    assert by_team["A"]["run_funnel_rate_vs_avg"] > by_team["C"]["run_funnel_rate_vs_avg"]


def test_red_zone_role_share():
    result = ms.red_zone_role_share(_sample_pbp(), as_of_week=3)
    by_player = {row["player_id"]: row for row in result.to_dicts()}

    # RB1's week1 (yardline_100=15) and week2 (yardline_100=18) rushes are both red zone touches;
    # WR1's week2 target (yardline_100=5) is also red zone -- 3 of team A's 3 red zone plays.
    assert by_player["RB1"]["red_zone_touches"] == 2
    assert by_player["WR1"]["red_zone_touches"] == 1
    assert by_player["RB1"]["team_red_zone_plays"] == 3
    assert by_player["RB1"]["red_zone_share"] == pytest.approx(2 / 3)


def test_recent_efficiency_trend_computes_trailing_vs_season():
    result = ms.recent_efficiency_trend(_sample_pbp(), as_of_week=3, trailing_games=1)
    by_player = {row["player_id"]: row for row in result.to_dicts()}

    rb1 = by_player["RB1"]
    assert rb1["season_epa_per_play"] == pytest.approx((0.5 + -0.5) / 2)
    assert rb1["trailing_epa_per_play"] == pytest.approx(-0.5)
    assert rb1["epa_trend"] == pytest.approx(-0.5 - 0.0)


def test_opponent_adjusted_target_share_nulls_out_bye_week_teams():
    result = ms.opponent_adjusted_target_share(_sample_pbp(), _sample_schedules(), as_of_week=3)
    by_player = {row["player_id"]: row for row in result.to_dicts()}

    # WR1 is A's only target through week 2 -> 100% target share; A's week-3 opponent is C.
    assert by_player["WR1"]["target_share"] == pytest.approx(1.0)
    assert by_player["WR1"]["opponent"] == "C"
    assert by_player["WR1"]["target_share_adjusted"] is not None

    # E is on a bye in week 3 (not in the schedule) -> no opponent, no adjustment.
    assert by_player["WR3"]["opponent"] is None
    assert by_player["WR3"]["target_share_adjusted"] is None


def test_build_signals_table_combines_all_signals_per_player():
    rows = ms.build_signals_table(2024, as_of_week=3, pbp=_sample_pbp(), schedules=_sample_schedules())
    by_player = {row["player_id"]: row for row in rows}

    rb1 = by_player["RB1"]
    assert rb1["season"] == 2024
    assert rb1["as_of_week"] == 3
    assert rb1["team"] == "A"
    assert rb1["opponent"] == "C"
    assert rb1["red_zone_share"] == pytest.approx(2 / 3)
    assert rb1["implied_total"] == pytest.approx(45.0 / 2 + 3.0 / 2)

    # WR3 (team E, on bye in week 3) still gets a row -- just with nulls for anything
    # that depends on having an opponent this week.
    wr3 = by_player["WR3"]
    assert wr3["opponent"] is None
    assert wr3["implied_total"] is None


def test_current_player_reference_resolves_trade_and_name_variants():
    hist = ms._history(_sample_pbp(), as_of_week=3)
    canonical = ms._current_player_reference(hist)
    by_id = {row["player_id"]: row for row in canonical.to_dicts()}

    # TR1's most recent (week 2) team is C, even though he played for A in week 1.
    assert by_id["TR1"]["team"] == "C"
    assert canonical.filter(pl.col("player_id") == "TR1").height == 1

    # NM1's most recent (week 2) name spelling wins, and there's exactly one row.
    assert by_id["NM1"]["player_name"] == "Na.Mixed"
    assert by_id["NM1"]["team"] == "A"
    assert canonical.filter(pl.col("player_id") == "NM1").height == 1


def test_recent_efficiency_trend_dedupes_traded_and_renamed_players():
    result = ms.recent_efficiency_trend(_sample_pbp(), as_of_week=3, trailing_games=1)
    ids = result["player_id"].to_list()

    # Exactly one row each, not one per team/name variant.
    assert ids.count("TR1") == 1
    assert ids.count("NM1") == 1

    by_player = {row["player_id"]: row for row in result.to_dicts()}
    # TR1 is attributed to his current team (C); his week-1 A-team play is
    # excluded from his own stats, which is why season_plays == 1.
    assert by_player["TR1"]["team"] == "C"
    assert by_player["TR1"]["season_plays"] == 1
    assert by_player["NM1"]["player_name"] == "Na.Mixed"
    assert by_player["NM1"]["season_plays"] == 2


def test_build_signals_table_has_one_row_per_player_despite_trades_and_renames():
    rows = ms.build_signals_table(2024, as_of_week=3, pbp=_sample_pbp(), schedules=_sample_schedules())
    player_ids = [row["player_id"] for row in rows]

    # This is the regression this test guards: duplicate player_id rows
    # here previously fanned out through build_signals_table's joins and
    # crashed embed.py's chromadb.add() with DuplicateIDError downstream.
    assert len(player_ids) == len(set(player_ids))

    by_player = {row["player_id"]: row for row in rows}
    assert by_player["TR1"]["team"] == "C"
    assert by_player["NM1"]["player_name"] == "Na.Mixed"


def test_build_signal_chunks_produces_unique_ids_for_traded_and_renamed_players():
    from src.rag import embed

    rows = ms.build_signals_table(2024, as_of_week=3, pbp=_sample_pbp(), schedules=_sample_schedules())
    chunk_ids = [c["id"] for c in embed.build_signal_chunks(rows)]

    assert len(chunk_ids) == len(set(chunk_ids))
