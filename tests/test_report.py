import json
from pathlib import Path

import polars as pl
import pytest

from src.rag import embed
from src.reasoning import report

_LEAGUE_ID = "1389341490030862336"  # Victorious Secret 3.0 -- see tests/test_recommend.py

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
        {
            "league_id": _LEAGUE_ID,
            "name": "Victorious Secret 3.0",
            "season": "2024",
            "settings": {"num_teams": 12},
            "scoring_settings": {"rec": 0.5},
        },
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


def _setup(
    tmp_path, monkeypatch, roster_players, signal_rows, players_df, second_team_players=None, prior_signal_rows=None
):
    raw_dir = tmp_path / "sleeper"
    persist_dir = tmp_path / "chroma"
    signals_dir = tmp_path / "signals"
    _seed_league(raw_dir, roster_players, second_team_players)
    embed.embed(embed.build_signal_chunks(signal_rows), persist_dir=persist_dir)
    signals_dir.mkdir()
    if signal_rows:
        pl.DataFrame(signal_rows).write_parquet(signals_dir / f"signals_{_SEASON}_week{_WEEK}.parquet")
    if prior_signal_rows:
        pl.DataFrame(prior_signal_rows).write_parquet(signals_dir / f"signals_{_SEASON - 1}_week19.parquet")

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
        "start_sit", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
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
        "start_sit", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    assert result["entries"] == []


def test_start_sit_requires_my_roster_id(tmp_path, monkeypatch):
    """delenv alone is NOT enough here: generate_report() calls
    load_dotenv() itself, which would restore MY_ROSTER_ID from a
    developer's real .env (it did, on Rohan's machine). tests/conftest.py's
    autouse fixture makes .env discovery find nothing for every test --
    see tests/test_dotenv_isolation.py for the mechanism and the guard."""
    roster = {"sleeper_cmc": {"full_name": "Christian McCaffrey", "position": "RB", "team": "SF"}}
    players_df = pl.DataFrame([_CHRISTIAN_ROW])
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [_CHRISTIAN_SIGNAL_ROW], players_df)
    monkeypatch.delenv("MY_ROSTER_ID")

    with pytest.raises(RuntimeError, match="MY_ROSTER_ID"):
        report.generate_report(
            "start_sit", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
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
        "drop", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
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
        "drop", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
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
        "waiver_pickups", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
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
        "waiver_pickups", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    assert result["entries"] == []  # James Cook has no signal row, so nothing to ground a pickup in


def test_waiver_pickups_does_not_require_my_roster_id(tmp_path, monkeypatch):
    my_roster = {"sleeper_cmc": {"full_name": "Christian McCaffrey", "position": "RB", "team": "SF"}}
    players_df = pl.DataFrame([_CHRISTIAN_ROW, _UNROSTERED_WR_ROW])
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, my_roster, [_CHRISTIAN_SIGNAL_ROW, _CHASE_SIGNAL_ROW], players_df
    )
    monkeypatch.delenv("MY_ROSTER_ID")  # stays deleted thanks to tests/conftest.py's .env isolation

    result = report.generate_report(
        "waiver_pickups", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    assert result["entries"]


# ---- Phase 3.6: prior-season signal fallback ----

# Real 2025 season-end signal values for Saquon Barkley (pulled from the
# committed data/processed/signals/signals_2025_week19.parquet) -- a
# distinguishable stand-in "prior season" row, deliberately different
# from _BARKLEY_SIGNAL_ROW's 2024 numbers so tests can tell which one a
# report actually used.
_BARKLEY_PRIOR_SIGNAL_ROW = {
    "player_id": "00-0034844", "player_name": "S.Barkley", "team": "PHI", "season": _SEASON - 1, "as_of_week": 19,
    "season_plays": 411, "epa_trend": 0.170673, "red_zone_share": 0.358621, "target_share": 0.109442,
    "target_share_adjusted": None, "opponent": None, "run_funnel_rate_vs_avg": None, "implied_total": None,
}


def test_drop_falls_back_to_stale_prior_season_when_no_current_season_data(tmp_path, monkeypatch):
    """A player with ONLY prior-season data on record gets the
    stale-labeled fallback, never an empty report."""
    roster = {"sleeper_barkley": {"full_name": "Saquon Barkley", "position": "RB", "team": "PHI"}}
    players_df = pl.DataFrame([_BARKLEY_ROW])
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, roster, [], players_df, prior_signal_rows=[_BARKLEY_PRIOR_SIGNAL_ROW]
    )

    result = report.generate_report(
        "drop", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    assert len(result["entries"]) == 1
    entry = result["entries"][0]
    assert entry["name"] == "Saquon Barkley"
    assert entry["stale"] is True
    assert entry["source_season"] == _SEASON - 1
    assert entry["signals_summary"].startswith("[STALE")
    assert any("stale" in reason.lower() for reason in entry["weakness_reasons"])
    assert any(f"{_SEASON - 1} season-end" in note for note in result["notes"])


def test_drop_never_falls_back_when_current_season_data_exists(tmp_path, monkeypatch):
    """A player with enough current-season data must never see the stale
    fallback, even when prior-season data also exists on record."""
    roster = {"sleeper_barkley": {"full_name": "Saquon Barkley", "position": "RB", "team": "PHI"}}
    players_df = pl.DataFrame([_BARKLEY_ROW])
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path,
        monkeypatch,
        roster,
        [_BARKLEY_SIGNAL_ROW],
        players_df,
        prior_signal_rows=[_BARKLEY_PRIOR_SIGNAL_ROW],
    )

    result = report.generate_report(
        "drop", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    entry = result["entries"][0]
    assert entry["stale"] is False
    assert entry["source_season"] is None
    assert "STALE" not in entry["signals_summary"]
    # _BARKLEY_SIGNAL_ROW's real 2024 red zone share (48%), not the prior
    # season fixture's 36% -- proves the current-season row won, not a
    # blend of the two.
    assert "48%" in entry["signals_summary"]
    assert not any("stale" in note.lower() for note in result["notes"])


def test_drop_handles_a_player_with_no_signal_data_at_all_gracefully(tmp_path, monkeypatch):
    """Even with a prior-season file present in signals_dir (for a
    different player), a roster player who appears in NEITHER the
    current-season nor any prior-season table must degrade to "no
    signal, excluded" rather than crash."""
    roster = {
        "sleeper_barkley": {"full_name": "Saquon Barkley", "position": "RB", "team": "PHI"},
        "sleeper_cook": {"full_name": "James Cook", "position": "RB", "team": "BUF"},
    }
    players_df = pl.DataFrame([_BARKLEY_ROW, _COOK_ROW])
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path,
        monkeypatch,
        roster,
        [_BARKLEY_SIGNAL_ROW],
        players_df,
        prior_signal_rows=[_BARKLEY_PRIOR_SIGNAL_ROW],
    )
    # Cook has no row in the current-season file above, and none in the
    # prior-season file either (it only has Barkley).

    result = report.generate_report(
        "drop", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    assert [e["name"] for e in result["entries"]] == ["Saquon Barkley"]
    assert any("James Cook" in note for note in result["notes"])


def test_waiver_pickups_skips_last_season_only_players_once_the_season_has_data(tmp_path, monkeypatch):
    """The first real waiver report's top three were players with 4-8 plays
    in all of 2025 and none in 2026. Once a current-season table exists,
    a player whose only row is last season's has no role now and is not
    a pickup; the report says how many it set aside."""
    my_roster = {"sleeper_cmc": {"full_name": "Christian McCaffrey", "position": "RB", "team": "SF"}}
    players_df = pl.DataFrame([_CHRISTIAN_ROW, _BARKLEY_ROW, _UNROSTERED_WR_ROW])
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, my_roster, [_CHRISTIAN_SIGNAL_ROW, _CHASE_SIGNAL_ROW], players_df,
        prior_signal_rows=[_BARKLEY_PRIOR_SIGNAL_ROW],
    )

    result = report.generate_report(
        "waiver_pickups", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    assert [e["name"] for e in result["entries"]] == ["Ja'Marr Chase"]
    assert any(note.startswith(f"1 unrostered player(s) with no {_SEASON} plays yet were not ranked") for note in result["notes"])


def test_waiver_pickups_uses_last_season_labeled_stale_before_the_season_has_any_data(tmp_path, monkeypatch):
    """Before the first current-season table exists, last season is all
    there is: still listed, still labeled stale (Phase 3.6)."""
    my_roster = {"sleeper_cmc": {"full_name": "Christian McCaffrey", "position": "RB", "team": "SF"}}
    players_df = pl.DataFrame([_CHRISTIAN_ROW, _BARKLEY_ROW])
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, my_roster, [], players_df, prior_signal_rows=[_BARKLEY_PRIOR_SIGNAL_ROW],
    )

    result = report.generate_report(
        "waiver_pickups", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    barkley = next(e for e in result["entries"] if e["name"] == "Saquon Barkley")
    assert barkley["stale"] is True
    assert barkley["source_season"] == _SEASON - 1
    assert "[STALE" in barkley["reasoning"]
    assert any("stale" in note.lower() for note in result["notes"])


def test_reports_say_kickers_and_defenses_have_no_signals_instead_of_failing_to_resolve_them(tmp_path, monkeypatch):
    roster = {
        "sleeper_barkley": {"full_name": "Saquon Barkley", "position": "RB", "team": "PHI"},
        "sleeper_cook": {"full_name": "James Cook", "position": "RB", "team": "BUF"},
        "sleeper_bates": {"full_name": "Jake Bates", "position": "K", "team": "DET"},
    }
    players_df = pl.DataFrame([_BARKLEY_ROW, _COOK_ROW])
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [_BARKLEY_SIGNAL_ROW, _COOK_SIGNAL_ROW], players_df)

    result = report.generate_report(
        "drop", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    assert "No matchup signals exist for kickers or defenses, so they aren't ranked: Jake Bates (K)." in result["notes"]
    assert not any("identity-resolve" in note for note in result["notes"])


def test_drop_report_explains_an_unrankable_player_once_not_twice(tmp_path, monkeypatch):
    """Roschon Johnson: 2 plays in 2025, no scored signal. The old notes
    said he 'fell back to stale 2025 data' AND 'had no computed signals',
    which reads as a contradiction."""
    roster = {
        "sleeper_barkley": {"full_name": "Saquon Barkley", "position": "RB", "team": "PHI"},
        "sleeper_cook": {"full_name": "James Cook", "position": "RB", "team": "BUF"},
        "sleeper_rj": {"full_name": "Roschon Johnson", "position": "RB", "team": "CHI"},
    }
    rj_index = {**_BARKLEY_ROW, "gsis_id": "00-0000099", "display_name": "Roschon Johnson"}
    rj_prior = {**_BARKLEY_PRIOR_SIGNAL_ROW, "player_id": "00-0000099", "player_name": "R.Johnson", "season_plays": 2,
                "epa_trend": None, "red_zone_share": None, "target_share": None, "target_share_adjusted": None}
    players_df = pl.DataFrame([_BARKLEY_ROW, _COOK_ROW, rj_index])
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, roster, [_BARKLEY_SIGNAL_ROW, _COOK_SIGNAL_ROW], players_df, prior_signal_rows=[rj_prior]
    )

    result = report.generate_report(
        "drop", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    notes = result["notes"]
    assert any(note.startswith("Not enough usage on record to rank") and f"Roschon Johnson (2 play(s) in {_SEASON - 1})" in note for note in notes), notes
    assert not any("fell back to stale" in note for note in notes), notes


# ---- misc ----


def test_unknown_report_type_raises(tmp_path, monkeypatch):
    roster = {"sleeper_cmc": {"full_name": "Christian McCaffrey", "position": "RB", "team": "SF"}}
    players_df = pl.DataFrame([_CHRISTIAN_ROW])
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [_CHRISTIAN_SIGNAL_ROW], players_df)

    with pytest.raises(ValueError, match="Unknown report_type"):
        report.generate_report(
            "trade_suggestions", _LEAGUE_ID,
            raw_dir=raw_dir,
            persist_dir=persist_dir,
            season=_SEASON,
            as_of_week=_WEEK,
            signals_dir=signals_dir,
        )


def test_requires_sleeper_ingest_to_have_run(tmp_path):
    with pytest.raises(RuntimeError, match=r"run `python -m src.ingest.sleeper"):
        report.generate_report("start_sit", _LEAGUE_ID, raw_dir=tmp_path / "nonexistent")


def test_generate_report_loads_dotenv_itself_not_only_via_cli_main(tmp_path, monkeypatch):
    """Same env-loading gap as recommend.py's recommend() (see
    tests/test_recommend.py's equivalent regression test): generate_report()
    reads MY_ROSTER_ID for start_sit/drop, which used to only get loaded
    from .env via this module's own CLI main() -- calling generate_report()
    directly (e.g. from a future API layer) skipped that entirely.
    Confirms it now calls load_dotenv() itself."""
    roster = {"sleeper_cmc": {"full_name": "Christian McCaffrey", "position": "RB", "team": "SF"}}
    players_df = pl.DataFrame([_CHRISTIAN_ROW])
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [_CHRISTIAN_SIGNAL_ROW], players_df)

    calls = []
    # Phase 5.2: generate_report() goes through recommend.load_dotenv_once(),
    # which calls recommend.py's load_dotenv reference (once per process;
    # tests/conftest.py resets the memo per test).
    from src.reasoning import recommend

    monkeypatch.setattr(recommend, "load_dotenv", lambda *a, **kw: calls.append((a, kw)))

    report.generate_report(
        "drop", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    assert len(calls) == 1  # load_dotenv() was actually invoked by generate_report() itself


# ---- start_sit follows the league's actual starting slots ----


def _league_with_slots(raw_dir: Path, roster_positions: list[str]) -> None:
    envelope = json.loads((raw_dir / "league.json").read_text())  # {"fetched_at", "source", "data"} -- see test_league._write
    envelope["data"]["roster_positions"] = roster_positions
    (raw_dir / "league.json").write_text(json.dumps(envelope))


def _rb_row(pid: str, name: str, rz: float) -> dict:
    return {**_BARKLEY_SIGNAL_ROW, "player_id": pid, "player_name": name, "red_zone_share": rz, "target_share": 0.05,
            "target_share_adjusted": 0.05, "epa_trend": 0.0}


def _rb_index_row(pid: str, name: str) -> dict:
    return {**_BARKLEY_ROW, "gsis_id": pid, "display_name": name}


def test_start_sit_recommends_as_many_starters_as_the_league_has_slots(tmp_path, monkeypatch):
    """The first friend's league started two RBs and the report recommended
    one -- a wrong answer, not a simplification. RB, RB + FLEX with four
    RBs: two RB starters, the third best in the FLEX, the fourth sits."""
    rbs = [("00-0000001", "Alpha Back", 0.50), ("00-0000002", "Bravo Back", 0.40), ("00-0000003", "Charlie Back", 0.30), ("00-0000004", "Delta Back", 0.20)]
    roster = {f"s_{pid}": {"full_name": name, "position": "RB", "team": "PHI"} for pid, name, _ in rbs}
    players_df = pl.DataFrame([_rb_index_row(pid, name) for pid, name, _ in rbs])
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [_rb_row(pid, name, rz) for pid, name, rz in rbs], players_df)
    _league_with_slots(raw_dir, ["QB", "RB", "RB", "WR", "TE", "FLEX", "K", "BN", "BN"])

    result = report.generate_report(
        "start_sit", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    by_slot = {e["position"]: e for e in result["entries"]}
    assert set(by_slot) == {"RB", "FLEX"}
    rb = by_slot["RB"]
    assert rb["slots"] == 2
    assert [s["name"] for s in rb["recommended_starters"]] == ["Alpha Back", "Bravo Back"]
    assert rb["recommended_starter"]["name"] == "Alpha Back"  # the top pick keeps its name -- Chat's verdict anchor
    assert [a["name"] for a in rb["alternatives_considered"]] == ["Charlie Back", "Delta Back"]
    assert rb["reasoning"].startswith("Start Alpha Back and Bravo Back at RB (2 slots).")
    flex = by_slot["FLEX"]
    assert flex["slots"] == 1 and flex["eligible_positions"] == ["RB", "WR", "TE"]
    assert [s["name"] for s in flex["recommended_starters"]] == ["Charlie Back"]
    assert [a["name"] for a in flex["alternatives_considered"]] == ["Delta Back"]
    assert all(s["position"] == "RB" for s in flex["recommended_starters"])


def test_start_sit_with_exactly_as_many_players_as_slots_has_nothing_to_decide(tmp_path, monkeypatch):
    roster = {
        "sleeper_barkley": {"full_name": "Saquon Barkley", "position": "RB", "team": "PHI"},
        "sleeper_cook": {"full_name": "James Cook", "position": "RB", "team": "BUF"},
    }
    players_df = pl.DataFrame([_BARKLEY_ROW, _COOK_ROW])
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [_BARKLEY_SIGNAL_ROW, _COOK_SIGNAL_ROW], players_df)
    _league_with_slots(raw_dir, ["QB", "RB", "RB", "WR", "BN"])

    result = report.generate_report(
        "start_sit", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    assert result["entries"] == []  # both start; the old one-slot assumption would have benched Cook


def test_start_sit_falls_back_to_one_slot_per_position_when_the_league_has_no_slot_structure(tmp_path, monkeypatch):
    roster = {
        "sleeper_barkley": {"full_name": "Saquon Barkley", "position": "RB", "team": "PHI"},
        "sleeper_cook": {"full_name": "James Cook", "position": "RB", "team": "BUF"},
    }
    players_df = pl.DataFrame([_BARKLEY_ROW, _COOK_ROW])
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [_BARKLEY_SIGNAL_ROW, _COOK_SIGNAL_ROW], players_df)
    _league_with_slots(raw_dir, [])

    result = report.generate_report(
        "start_sit", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    assert [(e["position"], e["slots"], e["recommended_starter"]["name"]) for e in result["entries"]] == [("RB", 1, "Saquon Barkley")]


def test_start_sit_leaves_out_players_who_cannot_play_and_says_so(tmp_path, monkeypatch):
    rbs = [("00-0000001", "Alpha Back", 0.50), ("00-0000002", "Bravo Back", 0.40), ("00-0000003", "Charlie Back", 0.30)]
    roster = {f"s_{pid}": {"full_name": name, "position": "RB", "team": "PHI"} for pid, name, _ in rbs}
    roster["s_00-0000001"]["injury_status"] = "IR"          # the best back, but on IR
    roster["s_00-0000002"]["injury_status"] = "Questionable"  # stays in -- that IS the call to help with
    players_df = pl.DataFrame([_rb_index_row(pid, name) for pid, name, _ in rbs])
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [_rb_row(pid, name, rz) for pid, name, rz in rbs], players_df)
    _league_with_slots(raw_dir, ["QB", "RB", "WR", "BN"])

    result = report.generate_report(
        "start_sit", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    entry = result["entries"][0]
    assert entry["recommended_starter"]["name"] == "Bravo Back"
    assert entry["recommended_starter"]["injury_status"] == "Questionable"
    assert [a["name"] for a in entry["alternatives_considered"]] == ["Charlie Back"]
    assert any(note == "Not available this week, left out of the lineup: Alpha Back (IR)." for note in result["notes"])
