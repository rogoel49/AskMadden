from unittest.mock import patch

import polars as pl

from src.ingest import nflverse

FAKE_DF = pl.DataFrame(
    [{"player_display_name": "Patrick Mahomes", "position": "QB", "week": 1, "passing_yards": 300}]
)
FAKE_PBP_DF = pl.DataFrame([{"season": 2024, "week": 1, "posteam": "KC", "epa": 0.5}])
FAKE_SCHEDULES_DF = pl.DataFrame(
    [{"season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL", "spread_line": 3.0, "total_line": 46.0}]
)


def test_save_weekly_stats_writes_parquet(tmp_path):
    with patch.object(nflverse, "fetch_weekly_stats", return_value=FAKE_DF) as mock_fetch:
        path = nflverse.save_weekly_stats(2024, out_dir=tmp_path)

    mock_fetch.assert_called_once_with(2024)
    assert path == tmp_path / "weekly_2024.parquet"
    assert path.exists()

    loaded = pl.read_parquet(path)
    assert loaded.row(0, named=True)["player_display_name"] == "Patrick Mahomes"


def test_save_pbp_writes_parquet(tmp_path):
    with patch.object(nflverse, "fetch_pbp", return_value=FAKE_PBP_DF) as mock_fetch:
        path = nflverse.save_pbp(2024, out_dir=tmp_path)

    mock_fetch.assert_called_once_with(2024)
    assert path == tmp_path / "pbp_2024.parquet"
    loaded = pl.read_parquet(path)
    assert loaded.row(0, named=True)["posteam"] == "KC"


def test_save_schedules_writes_parquet(tmp_path):
    with patch.object(nflverse, "fetch_schedules", return_value=FAKE_SCHEDULES_DF) as mock_fetch:
        path = nflverse.save_schedules(2024, out_dir=tmp_path)

    mock_fetch.assert_called_once_with(2024)
    assert path == tmp_path / "schedules_2024.parquet"
    loaded = pl.read_parquet(path)
    assert loaded.row(0, named=True)["home_team"] == "KC"
