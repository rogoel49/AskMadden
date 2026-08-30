from unittest.mock import patch

import pandas as pd

from src.ingest import nflverse

FAKE_DF = pd.DataFrame(
    [{"player_display_name": "Patrick Mahomes", "position": "QB", "week": 1, "passing_yards": 300}]
)


def test_save_weekly_stats_writes_parquet(tmp_path):
    with patch.object(nflverse, "fetch_weekly_stats", return_value=FAKE_DF) as mock_fetch:
        path = nflverse.save_weekly_stats(2024, out_dir=tmp_path)

    mock_fetch.assert_called_once_with(2024)
    assert path == tmp_path / "weekly_2024.parquet"
    assert path.exists()

    loaded = pd.read_parquet(path)
    assert loaded.iloc[0]["player_display_name"] == "Patrick Mahomes"
