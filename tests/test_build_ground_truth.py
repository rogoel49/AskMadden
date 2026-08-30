from unittest.mock import patch

import pandas as pd

from evals import build_ground_truth as bgt

# half-PPR-ish scoring, subset of a real Sleeper league's scoring_settings
SCORING = {
    "pass_yd": 0.04,
    "pass_td": 4.0,
    "pass_int": -1.0,
    "rush_yd": 0.1,
    "rush_td": 6.0,
    "rec": 0.5,
    "rec_yd": 0.1,
    "rec_td": 6.0,
    "fum_lost": -2.0,
}

FAKE_WEEKLY = pd.DataFrame(
    [
        {
            "week": 5,
            "position": "RB",
            "player_display_name": "Christian McCaffrey",
            "recent_team": "SF",
            "rushing_yards": 100,
            "rushing_tds": 1,
            "receptions": 3,
            "receiving_yards": 20,
            "receiving_tds": 0,
            "rushing_fumbles_lost": 1,
            "receiving_fumbles_lost": 0,
            "sack_fumbles_lost": 0,
            "passing_yards": 0,
            "passing_tds": 0,
            "interceptions": 0,
        },
        {
            # different week, should be excluded when weeks=[5]
            "week": 6,
            "position": "RB",
            "player_display_name": "Someone Else",
            "recent_team": "KC",
            "rushing_yards": 50,
            "rushing_tds": 0,
            "receptions": 0,
            "receiving_yards": 0,
            "receiving_tds": 0,
            "rushing_fumbles_lost": 0,
            "receiving_fumbles_lost": 0,
            "sack_fumbles_lost": 0,
            "passing_yards": 0,
            "passing_tds": 0,
            "interceptions": 0,
        },
        {
            # non-skill position, should be excluded
            "week": 5,
            "position": "DEF",
            "player_display_name": "49ers D/ST",
            "recent_team": "SF",
            "rushing_yards": 0,
            "rushing_tds": 0,
            "receptions": 0,
            "receiving_yards": 0,
            "receiving_tds": 0,
            "rushing_fumbles_lost": 0,
            "receiving_fumbles_lost": 0,
            "sack_fumbles_lost": 0,
            "passing_yards": 0,
            "passing_tds": 0,
            "interceptions": 0,
        },
    ]
)


def test_compute_points_matches_league_scoring():
    row = FAKE_WEEKLY.iloc[0].to_dict()

    points = bgt.compute_points(row, SCORING)

    # 100 rush yd * 0.1 + 1 rush td * 6 + 3 rec * 0.5 + 20 rec yd * 0.1 - 1 fumble lost * 2
    assert points == 10.0 + 6.0 + 1.5 + 2.0 - 2.0


def test_build_filters_to_requested_weeks_and_skill_positions():
    with patch.object(bgt.nflverse, "fetch_weekly_stats", return_value=FAKE_WEEKLY):
        entries = bgt.build(season=2024, weeks=[5], scoring=SCORING)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["player_name"] == "Christian McCaffrey"
    assert entry["week"] == 5
    assert entry["measured"] is True
    assert entry["source"] == "nflverse"
    assert entry["fantasy_points"] == 17.5
