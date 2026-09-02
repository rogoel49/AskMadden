import json
from pathlib import Path

import polars as pl
import pytest

from src.rag import embed
from src.reasoning import report

# Real nflverse identities (same fixtures test_recommend.py already uses for
# the McCaffrey-brothers bug, plus a couple more for a 3-candidate RB group).
_CHRISTIAN_ROW = {
    "gsis_id": "00-0033280", "display_name": "Christian McCaffrey", "position": "RB",
    "latest_team": "SF", "last_season": 2026,
}
_LUKE_ROW = {
    "gsis_id": "00-0039355", "display_name": "Luke McCaffrey", "position": "WR",
    "latest_team": "WAS", "last_season": 2026,
}
_BARKLEY_ROW = {
    "gsis_id": "00-0034844", "display_name": "Saquon Barkley", "position": "RB",
    "latest_team": "PHI", "last_season": 2026,
}
_COOK_ROW = {
    "gsis_id": "00-0037248", "display_name": "James Cook", "position": "RB",
    "latest_team": "BUF", "last_season": 2026,
}
_WEAK_WR_ROW = {
    "gsis_id": "00-0034115", "display_name": "Jaylen Wilson", "position": "WR",
    "latest_team": "MIA", "last_season": 2026,
}
_UNROSTERED_WR_ROW = {
    "gsis_id": "00-0036900", "display_name": "Ja'Marr Chase", "position": "WR",
    "latest_team": "CIN", "last_season": 2026,
}

_SEASON, _WEEK = 2024, 5

# Real 2024 week-5 signal values (pulled from the committed
# data/processed/signals/signals_2024_week5.parquet) so tests exercise the
# same numbers the real report is validated against.
_CHRISTIAN_SIGNAL_ROW = {
    "player_id": "00-0033280", "player_name": "C.McCaffrey", "team": "SF", "season": _SEASON, "as_of_week": _WEEK,
    "season_plays": 40, "epa_trend": 0.2, "red_zone_share": 0.4, "target_share": 0.15,
    "target_share_adjusted": 0.15, "opponent": "SEA", "run_funnel_rate_vs_avg": 0.05, "implied_total": 24.0,
}
_BARKLEY_SIGNAL_ROW = {
    "player_id": "00-0034844", "player_name": "S.Barkley", "team": "PHI", "season": _SEASON, "as_of_week": _WEEK,
    "season_plays": 90, "epa_trend": 0.006847, "red_zone_share": 0.48, "target_share": 0.121951,
    "target_share_adjusted": None, "opponent": None, "run_funnel_rate_vs_avg": None, "implied_total": None,
}
_COOK_SIGNAL_ROW = {
    "player_id": "00-0037248", "player_name": "J.Cook", "team": "BUF", "season": _SEASON, "as_of_week": _WEEK,
    "season_plays": 60, "epa_trend": 0.119438, "red_zone_share": 0.264706, "target_share": 0.102041,
    "target_share_adjusted": 0.098492, "opponent": "HOU", "run_funnel_rate_vs_avg": 0.02, "implied_total": 23.5,
}
_WEAK_WR_SIGNAL_ROW = {
    "player_id": "00-0034115", "player_name": "J.Wilson", "team": "MIA", "season": _SEASON, "as_of_week": _WEEK,
    "season_plays": 10, "epa_trend": -0.336411, "red_zone_share": None, "target_share": 0.015873,
    "target_share_adjusted": 0.017512, "opponent": "NE", "run_funnel_rate_vs_avg": -0.08, "implied_total": 19.5,
}
_CHASE_SIGNAL_ROW = {
    "player_id": "00-0036900", "player_name": "J.Chase", "team": "CIN", "season": _SEASON, "as_of_week": _WEEK,
    "season_plays": 24, "epa_trend": -0.045437, "red_zone_share": 0.116279, "target_share": 0.181818,
    "target_share_adjusted": 0.186837, "opponent": "BAL", "run_funnel_rate_vs_avg": -0.03, "implied_total": 23.0,
}


def _write(raw_dir: Path, filename: str, data) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / filename).write_text(json.dumps({"fetched_at": "2026-01-01T00:00:00Z", "source": "sleeper", "data": data}))


def _seed_league(raw_dir: Path, roster_players: dict, second_team_players: dict | None = None) -> None:
    _write(
        raw_dir,
        "league.json",
        {"name": "Victorious Secret 3.0", "season": "2024", "settings": {"num_teams": 12}, "scoring_settings": {"rec": 0.5}},
    )
    teams = [
        {
            "roster_id": 1, "owner_id": "u1", "display_name": "rogoel49", "team_name": "Victorious Secret",
            "players": list(roster_players), "starters": list(roster_players), "settings": {"wins": 3, "losses": 1, "ties": 0},
        }
    ]
    all_players = dict(roster_players)
    if second_team_players:
        teams.append(
            {
                "roster_id": 2, "owner_id": "u2", "display_name": "rival", "team_name": "Rival Team",
                "players": list(second_team_players), "starters": [], "settings": {"wins": 1, "losses": 3, "ties": 0},
            }
        )
        all_players.update(second_team_players)
    _write(raw_dir, "teams.json", teams)
    _write(raw_dir, "players.json", all_players)
    _write(raw_dir, "nfl_state.json", {"season": "2024", "display_week": _WEEK})
    _write(
        raw_dir,
        "matchups_week_5.json",
        [{"roster_id": 1, "matchup_id": 1, "points": 0}, {"roster_id": 2, "matchup_id": 1, "points": 0}],
    )


def _setup(tmp_path, monkeypatch, roster_players, signal_rows, players_df, second_team_players=None):
    raw_dir = tmp_path / "sleeper"
    persist_dir = tmp_path / "chroma"
    signals_dir = tmp_path / "signals"
    _seed_league(raw_dir, roster_players, second_team_players)
    embed.embed(embed.build_signal_chunks(signal_rows), persist_dir=persist_dir)
    signals_dir.mkdir()
    pl.DataFrame(signal_rows).write_parquet(signals_dir / f"signals_{_SEASON}_week{_WEEK}.parquet")

    import src.rag.player_index as player_index_module

    monkeypatch.setattr(player_index_module.nflverse, "fetch_players", lambda: players_df)
    monkeypatch.setenv("MY_ROSTER_ID", "1")
    return raw_dir, persist_dir, signals_dir


# ---- start_sit ----


def test_start_sit_recommends_the_stronger_rb_with_grounded_reasoning(tmp_path, monkeypatch):
    roster = {
        "sleeper_barkley": {"full_name": "Saquon Barkley", "position": "RB", "team": "PHI"},
        "sleeper_cook": {"full_name": "James Cook", "position": "RB", "team": "BUF"},
    }
    players_df = pl.DataFrame([_BARKLEY_ROW, _COOK_ROW])
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, roster, [_BARKLEY_SIGNAL_ROW, _COOK_SIGNAL_ROW], players_df
    )

    result = report.generate_report(
        "start_sit", raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    assert result["report_type"] == "start_sit"
    assert result["team_name"] == "Victorious Secret"
    assert result["record"]["wins"] == 3
    assert len(result["entries"]) == 1
    entry = result["entries"][0]
    assert entry["position"] == "RB"
    # Barkley: red_zone_share 0.48, epa_trend ~0.007, target_share 0.122
    # Cook: red_zone_share 0.265, epa_trend 0.119, target_share_adjusted 0.098
    # Composite score favors Barkley's much larger red-zone share.
    assert entry["recommended_starter"]["name"] == "Saquon Barkley"
    assert [alt["name"] for alt in entry["alternatives_considered"]] == ["James Cook"]
    assert "48%" in entry["reasoning"]  # Barkley's real red zone share, cited
    assert "James Cook" in entry["reasoning"]


def test_start_sit_skips_position_with_only_one_rostered_player(tmp_path, monkeypatch):
    roster = {"sleeper_cmc": {"full_name": "Christian McCaffrey", "position": "RB", "team": "SF"}}
    players_df = pl.DataFrame([_CHRISTIAN_ROW])
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [_CHRISTIAN_SIGNAL_ROW], players_df)

    result = report.generate_report(
        "start_sit", raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    assert result["entries"] == []


def test_start_sit_requires_my_roster_id(tmp_path, monkeypatch):
    roster = {"sleeper_cmc": {"full_name": "Christian McCaffrey", "position": "RB", "team": "SF"}}
    players_df = pl.DataFrame([_CHRISTIAN_ROW])
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [_CHRISTIAN_SIGNAL_ROW], players_df)
    monkeypatch.delenv("MY_ROSTER_ID")

    with pytest.raises(RuntimeError, match="MY_ROSTER_ID"):
        report.generate_report(
            "start_sit", raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
        )


# ---- drop ----


def test_drop_identifies_the_weak_contributor_with_concrete_reasons(tmp_path, monkeypatch):
    roster = {
        "sleeper_cmc": {"full_name": "Christian McCaffrey", "position": "RB", "team": "SF"},
        "sleeper_weak": {"full_name": "Jaylen Wilson", "position": "WR", "team": "MIA"},
    }
    players_df = pl.DataFrame([_CHRISTIAN_ROW, _WEAK_WR_ROW])
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, roster, [_CHRISTIAN_SIGNAL_ROW, _WEAK_WR_SIGNAL_ROW], players_df
    )

    result = report.generate_report(
        "drop", raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    assert result["report_type"] == "drop"
    assert len(result["entries"]) == 2  # both candidates, ranked weakest-first
    weakest = result["entries"][0]
    assert weakest["name"] == "Jaylen Wilson"
    assert any("efficiency trending down" in r for r in weakest["weakness_reasons"])
    assert any("low target share" in r for r in weakest["weakness_reasons"])
    assert result["entries"][1]["name"] == "Christian McCaffrey"


def test_drop_excludes_players_with_no_computed_signals(tmp_path, monkeypatch):
    roster = {
        "sleeper_cmc": {"full_name": "Christian McCaffrey", "position": "RB", "team": "SF"},
        "sleeper_nosig": {"full_name": "James Cook", "position": "RB", "team": "BUF"},
    }
    players_df = pl.DataFrame([_CHRISTIAN_ROW, _COOK_ROW])
    # Only seed a signal row for McCaffrey -- Cook has none.
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [_CHRISTIAN_SIGNAL_ROW], players_df)

    result = report.generate_report(
        "drop", raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    assert [e["name"] for e in result["entries"]] == ["Christian McCaffrey"]
    assert any("James Cook" in note for note in result["notes"])


# ---- waiver_pickups ----


def test_waiver_pickups_excludes_every_rostered_player_across_the_league(tmp_path, monkeypatch):
    my_roster = {"sleeper_cmc": {"full_name": "Christian McCaffrey", "position": "RB", "team": "SF"}}
    rival_roster = {"sleeper_cook": {"full_name": "James Cook", "position": "RB", "team": "BUF"}}
    players_df = pl.DataFrame([_CHRISTIAN_ROW, _COOK_ROW, _UNROSTERED_WR_ROW])
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path,
        monkeypatch,
        my_roster,
        [_CHRISTIAN_SIGNAL_ROW, _COOK_SIGNAL_ROW, _CHASE_SIGNAL_ROW],
        players_df,
        second_team_players=rival_roster,
    )

    result = report.generate_report(
        "waiver_pickups", raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    names = [e["name"] for e in result["entries"]]
    assert "Christian McCaffrey" not in names  # rostered on my team
    assert "James Cook" not in names  # rostered on the rival team
    assert names == ["Ja'Marr Chase"]
    assert result["entries"][0]["opportunity_score"] > 0


def test_waiver_pickups_excludes_pool_players_with_no_signal_data(tmp_path, monkeypatch):
    my_roster = {"sleeper_cmc": {"full_name": "Christian McCaffrey", "position": "RB", "team": "SF"}}
    # Cook is in the player pool but never given a signal row and never rostered.
    players_df = pl.DataFrame([_CHRISTIAN_ROW, _COOK_ROW])
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, my_roster, [_CHRISTIAN_SIGNAL_ROW], players_df)

    result = report.generate_report(
        "waiver_pickups", raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    assert result["entries"] == []  # James Cook has no signal row, so nothing to ground a pickup in


def test_waiver_pickups_does_not_require_my_roster_id(tmp_path, monkeypatch):
    my_roster = {"sleeper_cmc": {"full_name": "Christian McCaffrey", "position": "RB", "team": "SF"}}
    players_df = pl.DataFrame([_CHRISTIAN_ROW, _UNROSTERED_WR_ROW])
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, my_roster, [_CHRISTIAN_SIGNAL_ROW, _CHASE_SIGNAL_ROW], players_df
    )
    monkeypatch.delenv("MY_ROSTER_ID")

    result = report.generate_report(
        "waiver_pickups", raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    assert result["entries"]


# ---- misc ----


def test_unknown_report_type_raises(tmp_path, monkeypatch):
    roster = {"sleeper_cmc": {"full_name": "Christian McCaffrey", "position": "RB", "team": "SF"}}
    players_df = pl.DataFrame([_CHRISTIAN_ROW])
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [_CHRISTIAN_SIGNAL_ROW], players_df)

    with pytest.raises(ValueError, match="Unknown report_type"):
        report.generate_report(
            "trade_suggestions",
            raw_dir=raw_dir,
            persist_dir=persist_dir,
            season=_SEASON,
            as_of_week=_WEEK,
            signals_dir=signals_dir,
        )


def test_requires_sleeper_ingest_to_have_run(tmp_path):
    with pytest.raises(RuntimeError, match="run `python -m src.ingest.sleeper`"):
        report.generate_report("start_sit", raw_dir=tmp_path / "nonexistent")
