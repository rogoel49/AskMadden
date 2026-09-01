import polars as pl

from src.ingest import realtime


def test_current_injury_status_keeps_latest_report_per_player():
    injuries = pl.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "gsis_id": "00-001",
                "report_status": "Questionable",
                "date_modified": "2024-09-04T12:00:00Z",
            },
            {
                "season": 2024,
                "week": 1,
                "gsis_id": "00-001",
                "report_status": "Out",
                "date_modified": "2024-09-06T18:00:00Z",
            },
            {
                "season": 2024,
                "week": 2,
                "gsis_id": "00-001",
                "report_status": "Doubtful",
                "date_modified": "2024-09-11T12:00:00Z",
            },
        ]
    )

    result = realtime.current_injury_status(injuries, season=2024, week=1)

    assert result.height == 1
    assert result.row(0, named=True)["report_status"] == "Out"


def test_fetch_weather_selects_relevant_columns():
    schedules = pl.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "home_team": "GB",
                "away_team": "CHI",
                "roof": "outdoors",
                "surface": "grass",
                "temp": 45,
                "wind": 12,
                "spread_line": -3.0,
            }
        ]
    )

    result = realtime.fetch_weather(schedules)

    assert set(result.columns) == {"season", "week", "home_team", "away_team", "roof", "surface", "temp", "wind"}
    assert result.row(0, named=True)["wind"] == 12
