from unittest.mock import patch

import polars as pl
import pytest

from src.ingest import ngs

FAKE_NGS_DF = pl.DataFrame([{"season": 2024, "week": 1, "player_display_name": "Justin Jefferson"}])


def test_fetch_ngs_rejects_unknown_stat_type():
    with pytest.raises(ValueError, match="stat_type must be one of"):
        ngs.fetch_ngs(2024, "kicking")


@pytest.mark.parametrize("stat_type", ["passing", "receiving", "rushing"])
def test_save_ngs_writes_parquet(tmp_path, stat_type):
    with patch("nflreadpy.load_nextgen_stats", return_value=FAKE_NGS_DF) as mock_load:
        path = ngs.save_ngs(2024, stat_type, out_dir=tmp_path)

    mock_load.assert_called_once_with(seasons=[2024], stat_type=stat_type)
    assert path == tmp_path / f"ngs_{stat_type}_2024.parquet"
    loaded = pl.read_parquet(path)
    assert loaded.row(0, named=True)["player_display_name"] == "Justin Jefferson"


def test_save_all_writes_all_three_stat_types(tmp_path):
    with patch("nflreadpy.load_nextgen_stats", return_value=FAKE_NGS_DF):
        written = ngs.save_all(2024, out_dir=tmp_path)

    assert set(written) == {"passing", "receiving", "rushing"}
    assert all(path.exists() for path in written.values())
