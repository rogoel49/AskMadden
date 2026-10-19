"""The Ferguson-over-Likely RCA fixes (2026-10-14): a player whose game this
week is already final is counted at his actual points; a Questionable
player's Sleeper designation is combined with the NFL report's practice
status; a player who changed teams is shrunk less toward last season; and
both surfaces say when the games are and what the lock-in sequence is."""
from __future__ import annotations

import polars as pl

from src.reasoning import ranking, recommend, report
from src.reasoning.ranking import SignalTables, rank_candidates
from src.signals import availability as av
from src.signals import matchup_signals as ms

RATES = {"rates": {"Questionable|unknown|*": {"n": 100, "rate": 0.68}, "Questionable|full|*": {"n": 100, "rate": 0.71},
                   "Questionable|full|TE": {"n": 100, "rate": 0.77}},
         "factors": {"Questionable|unknown|*": {"n": 100, "factor": 0.86}, "Questionable|full|*": {"n": 100, "factor": 0.91}}}


def _row(pid, ppg, **extra):
    return {"player_id": pid, "position": "TE", "ppg": ppg, "games_played": 4, "target_share": 0.2, "red_zone_share": 0.1,
            "epa_trend": None, "epa_baseline_plays": 0, "season_plays": 40, "stale": False, "source_season": None,
            "source_as_of_week": None, **extra}


def test_sleeper_designation_takes_the_reports_practice_status(monkeypatch):
    monkeypatch.setattr(av, "load_play_rates", lambda path=None: RATES)
    # The report lists him (full participation, knee) with no game designation; Sleeper says Questionable.
    row = _row("a", 10.0, report_status=None, practice_status="full practice", injury="Knee", play_probability=None)
    fields = ranking.availability_of(row, "Questionable", "TE")
    assert fields["play_probability"] == 0.77 and fields["practice_status"] == "full practice" and fields["injury"] == "Knee"
    assert fields["expected_output"] == round(0.77 * 0.91, 2)
    # Nothing from the report at all: the pooled rate, as before.
    assert ranking.availability_of(_row("b", 10.0), "Questionable", "TE")["play_probability"] == 0.68


def test_a_player_whose_game_is_final_is_counted_at_his_actual_points():
    thursday = _row("f", 9.0, ppg_prior_season=9.0, already_played=True, this_week_points=3.4, game_day="Thu 10/08 8:15pm ET",
                    kickoff="2026-10-08 20:15 ET")
    sunday = _row("l", 10.5, ppg_prior_season=4.0, team="NYG", team_this_season="NYG", team_prior_season="BAL",
                  game_day="Sun 10/11 1:00pm ET", kickoff="2026-10-11 13:00 ET")
    # Without the game being played, Ferguson's blended 9.0 beats Likely's team-change blend ((4*10.5+2*4)/6 = 8.33)...
    assert ranking.opportunity_score({**thursday, "already_played": False, "this_week_points": None}) > ranking.opportunity_score(sunday)
    # ...but once his 3.4 is in the books, the points term uses 3.4 and Likely leads.
    assert ranking.opportunity_score(thursday) < ranking.opportunity_score(sunday)
    out = rank_candidates([{"player_id": "f", "name": "F", "position": "TE", "team": "DAL", "row": thursday},
                           {"player_id": "l", "name": "L", "position": "TE", "team": "NYG", "row": sunday}])
    assert out["recommended"]["name"] == "L"
    f = next(e for e in out["ranked"] if e["name"] == "F")
    assert f["already_played"] is True and f["this_week_points"] == 3.4 and f["game_day"] == "Thu 10/08 8:15pm ET"
    assert "ALREADY PLAYED this week (Thu 10/08 8:15pm ET): 3.4 pts" in f["signals_summary"]
    assert f["key_stats"]["already_played"] is True
    # An already-played player carries no availability discount even if Sleeper still tags him.
    assert ranking.availability_of(thursday, "Questionable", "TE")["play_probability"] is None
    l = next(e for e in out["ranked"] if e["name"] == "L")
    assert "plays Sun 10/11 1:00pm ET" in l["signals_summary"]
    assert "with BAL -- a different team, so last season counts as only 2 game(s)" in l["signals_summary"]


def test_team_change_shrinks_less_toward_last_season():
    same = {"ppg": 10.5, "games_played": 4, "ppg_prior_season": 4.0, "team": "BAL", "team_prior_season": "BAL"}
    moved = {**same, "team": "NYG", "team_this_season": "NYG"}
    assert ranking.blended_ppg(same) == (4 * 10.5 + 4 * 4.0) / 8
    assert ranking.blended_ppg(moved) == (4 * 10.5 + 2 * 4.0) / 6
    assert ranking.team_changed(moved) and not ranking.team_changed(same)
    # A stale fallback row carries last season's team and no current one: never a change.
    assert not ranking.team_changed({"team": "BAL", "team_prior_season": "BAL", "stale": True})
    assert not ranking.team_changed({"team": "NYG"})


def test_format_game_day():
    assert ranking.format_game_day("2026-10-08", "20:15", "Thursday") == "Thu 10/08 8:15pm ET"
    assert ranking.format_game_day("2026-10-11", "13:00", None) == "Sun 10/11 1:00pm ET"
    assert ranking.format_game_day(None, "13:00", "Sunday") is None


def test_opponent_map_carries_kickoff_columns_when_the_schedule_has_them():
    sched = pl.DataFrame([
        {"season": 2026, "week": 5, "home_team": "DAL", "away_team": "TB", "gameday": "2026-10-08", "gametime": "20:15", "weekday": "Thursday"},
        {"season": 2026, "week": 5, "home_team": "NYG", "away_team": "WAS", "gameday": "2026-10-11", "gametime": "13:00", "weekday": "Sunday"},
    ])
    m = {r["team"]: r for r in ms._opponent_map(sched, 5).to_dicts()}
    assert m["TB"]["opponent"] == "DAL" and m["TB"]["gameday"] == "2026-10-08" and m["WAS"]["weekday"] == "Sunday"
    # Fixtures without the columns still work (the older tables and tests).
    plain = pl.DataFrame([{"season": 2026, "week": 5, "home_team": "DAL", "away_team": "TB"}])
    assert ms._opponent_map(plain, 5).columns == ["team", "opponent"]


def test_signal_tables_know_which_games_are_final(tmp_path):
    signals = tmp_path / "signals"
    signals.mkdir()
    (tmp_path / "player_stats").mkdir()
    pl.DataFrame([
        {"player_id": "f", "player_name": "J.Ferguson", "team": "DAL", "opponent": "TB", "implied_total": 28.5, "season": 2026,
         "as_of_week": 5, "gameday": "2026-10-08", "gametime": "20:15", "weekday": "Thursday", "target_share": 0.1, "red_zone_share": 0.09,
         "epa_trend": None, "epa_baseline_plays": 2, "season_plays": 15},
        {"player_id": "l", "player_name": "I.Likely", "team": "NYG", "opponent": "WAS", "implied_total": 20.25, "season": 2026,
         "as_of_week": 5, "gameday": "2026-10-11", "gametime": "13:00", "weekday": "Sunday", "target_share": 0.31, "red_zone_share": 0.18,
         "epa_trend": None, "epa_baseline_plays": 8, "season_plays": 35},
    ]).write_parquet(signals / "signals_2026_week5.parquet")
    cols = ["player_id", "player_display_name", "position", "team", "opponent_team", "season", "week", "receptions", "receiving_yards", "receiving_tds"]
    lines = [["f", "Jake Ferguson", "TE", "DAL", "HOU", 2026, 4, 3.0, 11.0, 0.0], ["l", "Isaiah Likely", "TE", "NYG", "ARI", 2026, 4, 7.0, 66.0, 0.0],
             ["f", "Jake Ferguson", "TE", "DAL", "TB", 2026, 5, 3.0, 19.0, 0.0]]  # the Thursday game is in; the Giants' is not
    pl.DataFrame([dict(zip(cols, r)) for r in lines]).write_parquet(tmp_path / "player_stats" / "player_stats_2026.parquet")
    pl.DataFrame([dict(zip(cols, ["l", "Isaiah Likely", "TE", "BAL", "PIT", 2025, 18, 2.0, 20.0, 0.0]))]).write_parquet(
        tmp_path / "player_stats" / "player_stats_2025.parquet")
    scoring = {"rec": 0.5, "rec_yd": 0.1, "rec_td": 6}
    strict = SignalTables.load(signals, 2026, 5, scoring, stats_dir=tmp_path / "player_stats", injuries_dir=tmp_path / "none")
    assert strict.teams_played_this_week == set() and strict.row_for("f")["already_played"] is False
    assert strict.row_for("f")["game_day"] == "Thu 10/08 8:15pm ET" and strict.row_for("l")["kickoff"] == "2026-10-11 13:00 ET"
    live = SignalTables.load(signals, 2026, 5, scoring, stats_dir=tmp_path / "player_stats", injuries_dir=tmp_path / "none",
                             points_through_week=6)
    assert live.teams_played_this_week == {"DAL", "TB"}
    f, l = live.row_for("f"), live.row_for("l")
    assert f["already_played"] is True and f["this_week_points"] == 3.4 and l["already_played"] is False and l["this_week_points"] is None
    assert l["team_prior_season"] == "BAL" and l["team_this_season"] == "NYG" and ranking.team_changed(l)
    assert live.game_fields("f")["this_week_points"] == 3.4 and live.game_fields("nobody")["already_played"] is None


def test_report_and_chat_say_the_lock_in_sequence():
    starter = {"name": "Likely", "player_id": "l", "play_probability": 0.77, "report_status": "Questionable",
               "kickoff": "2026-10-11 13:00 ET", "game_day": "Sun 10/11 1:00pm ET"}
    alt = {"name": "Ferguson", "player_id": "f", "play_probability": None, "kickoff": "2026-10-08 20:15 ET", "game_day": "Thu 10/08 8:15pm ET"}
    s = report.timing_sentence([starter], [alt])
    assert "Ferguson plays Thu 10/08 8:15pm ET, before Likely (Sun 10/11 1:00pm ET, Questionable) is settled" in s
    assert "held until Thu 10/08 8:15pm ET" in s
    assert report.timing_sentence([alt], [starter]) is None  # the healthy player starts: nothing to sequence
    note = recommend._timing_note([starter, alt, {"name": "Kraft", "player_id": "k", "play_probability": None,
                                                  "kickoff": "2026-10-11 16:25 ET", "game_day": "Sun 10/11 4:25pm ET"}])
    assert "Kraft (Sun 10/11 4:25pm ET) would still be available as the fallback" in note
    assert recommend._timing_note([alt]) is None


def test_prompt_has_the_game_timing_rules():
    prompt = recommend._build_system_prompt({"name": "L", "roster_positions": ["QB", "TE"]}, {"rec": 0.5}, 2026, 5)
    assert "GAME TIMING" in prompt and "already_played" in prompt and "Never project a player whose game is over" in prompt
