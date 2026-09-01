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
