import polars as pl
import pytest

from src.ingest import odds


def _schedules() -> pl.DataFrame:
    return pl.DataFrame(
        [
            # home team favored by 3, over/under 46 -> home implied 24.5, away implied 21.5
            {"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL", "spread_line": 3.0, "total_line": 46.0},
            # home team is a 3-point underdog -> away favored, home implied lower than away
            {"season": 2024, "week": 1, "home_team": "IND", "away_team": "HOU", "spread_line": -3.0, "total_line": 49.0},
        ]
    )


def test_implied_team_totals_uses_home_favored_convention():
    result = odds.implied_team_totals(_schedules())
    by_team = {row["team"]: row for row in result.to_dicts()}

    assert by_team["KC"]["implied_total"] == pytest.approx(24.5)
    assert by_team["BAL"]["implied_total"] == pytest.approx(21.5)
    assert by_team["KC"]["implied_total"] + by_team["BAL"]["implied_total"] == pytest.approx(46.0)


def test_implied_team_totals_handles_home_underdog():
    result = odds.implied_team_totals(_schedules())
    by_team = {row["team"]: row for row in result.to_dicts()}

    # IND (home) is the underdog, so HOU (away, favored) should have the higher implied total.
    assert by_team["HOU"]["implied_total"] > by_team["IND"]["implied_total"]
    assert by_team["IND"]["implied_total"] + by_team["HOU"]["implied_total"] == pytest.approx(49.0)


def test_implied_team_totals_returns_one_row_per_team_per_game():
    result = odds.implied_team_totals(_schedules())
    assert result.height == 4
    assert set(result["team"].to_list()) == {"KC", "BAL", "IND", "HOU"}
