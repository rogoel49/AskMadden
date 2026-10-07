"""2026-10-05, from three real screenshots on a Monday afternoon.

1. Case Keenum -- one 24-point spot start, no 2025 stat line, QB2 behind
   Tyson Bagent on Sleeper's depth chart -- was the #1 waiver QB in two
   leagues and started over Patrick Mahomes in a superflex lineup.
2. De'Von Achane -- Sleeper: IR, "Knee - ACL" -- was the #1 waiver target.
   The waiver pool read nflverse usage only; Sleeper's injury designation
   and depth chart were on disk the whole time.
3. The stat lines behind points per game stopped at last week while this
   week's Sunday games were already final.

The fixes: ranking.is_backup_qb + a sort group in rank_candidates, the
waiver pool reading lookup.sleeper_player_info_by_name, and a points
horizon the live product sets (count_games_played_this_week) that every
backtest leaves off.
"""
import json
from pathlib import Path

import polars as pl

from src.reasoning import ranking, recommend, report
from src.reasoning.ranking import SignalTables, rank_candidates
from src.scheduler import refresh
from tests.test_report import (
    _CHASE_SIGNAL_ROW,
    _LEAGUE_ID,
    _SEASON,
    _UNROSTERED_WR_ROW,
    _WEAK_WR_ROW,
    _WEAK_WR_SIGNAL_ROW,
    _WEEK,
    _setup,
)
from tests.test_refresh import SCHEDULES, SEASON, _ingest_one_league, cycle  # noqa: F401 -- the fixture

KEENUM_ROW = {"gsis_id": "00-0031800", "display_name": "Case Keenum", "position": "QB", "latest_team": "CHI", "last_season": 2026}
MAHOMES_ROW = {"gsis_id": "00-0033873", "display_name": "Patrick Mahomes", "position": "QB", "latest_team": "KC", "last_season": 2026}
KEENUM_SIGNAL = {
    "player_id": "00-0031800", "player_name": "C.Keenum", "team": "CHI", "season": _SEASON, "as_of_week": _WEEK,
    "season_plays": 35, "epa_trend": None, "red_zone_share": 0.02, "target_share": None, "target_share_adjusted": None,
    "opponent": "NYJ", "run_funnel_rate_vs_avg": 0.01, "implied_total": 23.5, "cpoe": 1.1,
}
MAHOMES_SIGNAL = {
    "player_id": "00-0033873", "player_name": "P.Mahomes", "team": "KC", "season": _SEASON, "as_of_week": _WEEK,
    "season_plays": 140, "epa_trend": None, "red_zone_share": 0.01, "target_share": None, "target_share_adjusted": None,
    "opponent": "JAX", "run_funnel_rate_vs_avg": -0.02, "implied_total": 27.0, "cpoe": 0.4,
}


def _qb(name, pid, depth, ppg, games=3, stale=False):
    return {
        "player_id": pid, "name": name, "position": "QB", "team": "X", "depth_chart_order": depth,
        "row": {"player_id": pid, "ppg": ppg, "games_played": games, "ppg_prior_season": None, "cpoe": 1.0,
                "implied_total": 24.0, "stale": stale, "source_season": 2026 if not stale else 2025,
                "source_as_of_week": 5, "season_plays": 100, "epa_trend": None},
    }


# ---- ranking ----


def test_a_backup_qb_ranks_behind_every_starter_whatever_his_numbers():
    keenum = _qb("Case Keenum", "k", depth=2, ppg=24.5, games=1)
    mahomes = _qb("Patrick Mahomes", "m", depth=1, ppg=18.9)
    result = rank_candidates([keenum, mahomes])
    assert result["verdict"] == "clear"
    assert [e["name"] for e in result["ranked"]] == ["Patrick Mahomes", "Case Keenum"]
    k = result["ranked"][1]
    assert k["backup_qb"] is True and k["depth_chart_order"] == 2
    assert k["signals_summary"].startswith("listed QB2 on Sleeper's depth chart")
    assert k["key_stats"]["depth_chart_order"] == 2
    assert result["ranked"][0]["backup_qb"] is False
    assert "depth_chart_description" in result


def test_a_backup_qb_sorts_behind_a_non_qb_in_a_superflex_pool_too():
    keenum = _qb("Case Keenum", "k", depth=2, ppg=24.5, games=1)
    rb = {"player_id": "r", "name": "Some Back", "position": "RB", "team": "X", "depth_chart_order": 3,
          "row": {"player_id": "r", "ppg": 6.0, "games_played": 3, "ppg_prior_season": 5.0, "red_zone_share": 0.2,
                  "target_share": 0.08, "stale": False, "source_season": 2026, "source_as_of_week": 5, "season_plays": 80}}
    result = rank_candidates([keenum, rb])
    assert [e["name"] for e in result["ranked"]] == ["Some Back", "Case Keenum"]
    assert result["ranked"][0]["backup_qb"] is False  # an RB3 still plays snaps; the rule is QB-only


def test_a_backup_qb_and_a_starter_on_equal_scores_are_not_a_tie():
    a = _qb("Starter", "a", depth=1, ppg=20.0)
    b = _qb("Backup", "b", depth=2, ppg=20.0)
    result = rank_candidates([b, a])
    assert result["verdict"] == "clear" and result["recommended"]["name"] == "Starter"


def test_depth_chart_rule_is_quarterbacks_only_and_tolerates_missing_or_bad_values():
    assert ranking.is_backup_qb("QB", 2) and ranking.is_backup_qb("qb", "3")
    assert not ranking.is_backup_qb("QB", 1) and not ranking.is_backup_qb("QB", None)
    assert not ranking.is_backup_qb("RB", 4) and not ranking.is_backup_qb("QB", "n/a")


# ---- the points horizon ----


def test_signal_tables_points_horizon_defaults_to_strictly_before_the_week(tmp_path, monkeypatch):
    seen = []

    def fake_proxy(stats, scoring, as_of_week=None):
        seen.append(as_of_week)
        return {}

    from src.reasoning import points_proxy
    from src.signals import player_stats

    monkeypatch.setattr(points_proxy, "season_points_proxy", fake_proxy)
    monkeypatch.setattr(player_stats, "load_weekly_stats", lambda season, stats_dir=None: None)
    SignalTables.load(tmp_path, 2026, 5, scoring_settings={"rec": 0.5})
    assert seen == [5, None]  # this season as-of week 5, last season whole
    seen.clear()
    tables = SignalTables.load(tmp_path, 2026, 5, scoring_settings={"rec": 0.5}, points_through_week=6)
    assert seen == [6, None] and tables.points_through_week == 6


def test_the_live_report_counts_this_weeks_finished_games_and_says_so(tmp_path, monkeypatch):
    roster = {"sleeper_wilson": {"full_name": "Jaylen Wilson", "position": "WR", "team": "MIA"}}
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [_WEAK_WR_SIGNAL_ROW], pl.DataFrame([_WEAK_WR_ROW]))
    seen = {}
    real = SignalTables.load

    def spy(*args, **kwargs):
        seen["points_through_week"] = kwargs.get("points_through_week")
        return real(*args, **kwargs)

    monkeypatch.setattr(report.SignalTables, "load", spy)
    strict = report.generate_report("drop", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON,
                                    as_of_week=_WEEK, signals_dir=signals_dir)
    assert seen["points_through_week"] is None and strict["points_through_week"] is None
    live = report.generate_report("drop", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON,
                                  as_of_week=_WEEK, signals_dir=signals_dir, count_games_played_this_week=True)
    assert seen["points_through_week"] == _WEEK + 1 and live["points_through_week"] == _WEEK + 1
    assert any(f"week-{_WEEK} games already final" in n for n in live["notes"])


def test_recommend_context_carries_the_horizon_only_when_asked(tmp_path, monkeypatch):
    roster = {"sleeper_wilson": {"full_name": "Jaylen Wilson", "position": "WR", "team": "MIA"}}
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [_WEAK_WR_SIGNAL_ROW], pl.DataFrame([_WEAK_WR_ROW]))
    captured = {}

    class FakeClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("stop here")

    real_ctx = recommend.RecommendContext

    def spy_ctx(*args, **kwargs):
        captured["points_through_week"] = kwargs.get("points_through_week")
        return real_ctx(*args, **kwargs)

    monkeypatch.setattr(recommend, "RecommendContext", spy_ctx)
    for flag, expected in ((False, None), (True, _WEEK + 1)):
        try:
            recommend.recommend("q", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK,
                                client=FakeClient(), signals_dir=signals_dir, count_games_played_this_week=flag)
        except RuntimeError:
            pass
        assert captured["points_through_week"] == expected


# ---- the waiver pool reads Sleeper's designation and depth chart ----


def _add_to_players_json(raw_dir: Path, extra: dict) -> None:
    path = raw_dir / "players.json"
    payload = json.loads(path.read_text())
    payload["data"].update(extra)
    path.write_text(json.dumps(payload))


def test_waiver_pickups_leave_out_ir_players_and_backup_qbs_and_name_them(tmp_path, monkeypatch):
    my_roster = {"sleeper_wilson": {"full_name": "Jaylen Wilson", "position": "WR", "team": "MIA"}}
    players_df = pl.DataFrame([_WEAK_WR_ROW, _UNROSTERED_WR_ROW, KEENUM_ROW, MAHOMES_ROW])
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, my_roster, [_WEAK_WR_SIGNAL_ROW, _CHASE_SIGNAL_ROW, KEENUM_SIGNAL, MAHOMES_SIGNAL], players_df
    )
    _add_to_players_json(raw_dir, {
        "s_chase": {"full_name": "Ja'Marr Chase", "position": "WR", "team": "CIN", "injury_status": "IR", "injury_body_part": "Knee - ACL"},
        "s_keenum": {"full_name": "Case Keenum", "position": "QB", "team": "CHI", "depth_chart_order": 2},
        "s_mahomes": {"full_name": "Patrick Mahomes", "position": "QB", "team": "KC", "depth_chart_order": 1, "injury_status": "Questionable"},
    })

    result = report.generate_report(
        "waiver_pickups", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    names = [e["name"] for e in result["entries"]]
    assert "Ja'Marr Chase" not in names, "a player Sleeper lists IR is not a pickup"
    assert "Case Keenum" not in names, "a QB2 is not a pickup"
    assert names == ["Patrick Mahomes"]
    assert any("Ja'Marr Chase (IR)" in n for n in result["notes"])
    assert any("Case Keenum (QB2)" in n for n in result["notes"])
    entry = result["entries"][0]
    assert entry["injury_status"] == "Questionable" and entry["depth_chart_order"] == 1
    # A Questionable pickup carries the same availability fields a start/sit ref does (a Sleeper-only designation
    # gets the pooled rate when no report row exists), and is counted at expected output.
    assert entry["play_probability"] is not None and entry["key_stats"]["play_probability"] == entry["play_probability"]
    assert "base rate" in entry["reasoning"]


def test_start_sit_sits_a_backup_qb_behind_the_starter_and_notes_the_depth_chart(tmp_path, monkeypatch):
    roster = {
        "s_keenum": {"full_name": "Case Keenum", "position": "QB", "team": "CHI", "depth_chart_order": 2},
        "s_mahomes": {"full_name": "Patrick Mahomes", "position": "QB", "team": "KC", "depth_chart_order": 1},
    }
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [KEENUM_SIGNAL, MAHOMES_SIGNAL], pl.DataFrame([KEENUM_ROW, MAHOMES_ROW]))

    result = report.generate_report(
        "start_sit", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    entry = result["entries"][0]
    assert entry["recommended_starter"]["name"] == "Patrick Mahomes"
    assert entry["recommended_starter"]["backup_qb"] is False
    alt = entry["alternatives_considered"][0]
    assert alt["name"] == "Case Keenum" and alt["backup_qb"] is True and alt["depth_chart_order"] == 2
    assert any("Case Keenum (QB2)" in n for n in result["notes"])
    # The chat's roster tool exposes the same field, so rank_players sees it too.
    ctx = recommend.RecommendContext(raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, player_idx=None)
    players = {p["name"]: p for p in recommend.dispatch_tool("get_my_roster", {}, ctx)["players"]}
    assert players["Case Keenum"]["depth_chart_order"] == 2


def test_the_system_prompt_states_the_backup_qb_rule():
    prompt = recommend._build_system_prompt({"name": "L"}, {"rec": 0.5}, 2026, 5)
    assert "backup_qb" in prompt and "QB2" in prompt
    assert "picking up a player on IR" in prompt


# ---- the refresh writes next week's table early ----


def _schedules_with_week3_started() -> pl.DataFrame:
    rows = SCHEDULES.to_dicts()
    rows[[i for i, r in enumerate(rows) if r["week"] == 3][0]]["result"] = 7  # one week-3 game is final
    return pl.DataFrame(rows)


def test_next_weeks_table_is_written_early_once_the_target_week_has_started(cycle, monkeypatch):
    _ingest_one_league(cycle)
    monkeypatch.setattr(refresh.nflverse, "fetch_schedules", lambda season: _schedules_with_week3_started())

    record = refresh.run_cycle(signals_dir=cycle["signals_dir"], status_path=cycle["status_path"])

    assert record["outcome"] == "ok"
    assert (record["as_of_week"], record["early_as_of_week"]) == (3, 4), "the strict target stays 3; week 4 is the early table"
    assert [t["as_of_week"] for t in record["signals"]["tables"]] == [3, 4]
    assert (cycle["signals_dir"] / f"signals_{SEASON}_week4.parquet").exists()
    status = json.loads(cycle["status_path"].read_text())
    assert status["last_run"]["early_as_of_week"] == 4


def test_no_early_table_before_the_target_week_kicks_off_or_for_an_explicit_week(cycle):
    _ingest_one_league(cycle)
    record = refresh.run_cycle(signals_dir=cycle["signals_dir"], status_path=cycle["status_path"])
    assert record["early_as_of_week"] is None
    assert [t["as_of_week"] for t in record["signals"]["tables"]] == [3]
    record = refresh.run_cycle(as_of_week=2, signals_dir=cycle["signals_dir"], status_path=cycle["status_path"])
    assert record["early_as_of_week"] is None and [t["as_of_week"] for t in record["signals"]["tables"]] == [2]


def test_week_has_started_reads_results_only_for_that_regular_season_week():
    assert refresh.week_has_started(SCHEDULES, SEASON, 2)
    assert not refresh.week_has_started(SCHEDULES, SEASON, 3)
    assert refresh.week_has_started(_schedules_with_week3_started(), SEASON, 3)


def test_the_frontend_groups_a_superflex_qb_under_qb_and_shows_the_depth_chart_chip():
    html = Path("design/askmadden-ui-mockup.html").read_text()
    assert "flexQbs" in html and "shown under QB" in html
    assert "function depthChip" in html and "on depth chart" in html


# ---- what the first real run after the fix surfaced (same day) ----


def test_a_proxy_only_row_with_no_current_season_points_is_stale(tmp_path, monkeypatch):
    from src.reasoning import points_proxy
    from src.signals import player_stats

    monkeypatch.setattr(player_stats, "load_weekly_stats", lambda season, stats_dir=None: None)
    monkeypatch.setattr(points_proxy, "season_points_proxy",
                        lambda stats, scoring, as_of_week=None: {} if as_of_week is not None else {"fa": {"ppg": -0.3, "games_played": 3}})
    tables = SignalTables.load(tmp_path, 2026, 5, scoring_settings={"rec": 0.5})
    row = tables.row_for("fa")
    assert row["proxy_only"] and row["stale"] is True and row["source_season"] == 2025
    assert row["ppg"] is None and row["ppg_prior_season"] == -0.3


def test_waiver_pickups_skip_free_agents_unlisted_qbs_and_players_with_almost_no_plays(tmp_path, monkeypatch):
    my_roster = {"sleeper_wilson": {"full_name": "Jaylen Wilson", "position": "WR", "team": "MIA"}}
    cameo = {**_CHASE_SIGNAL_ROW, "season_plays": 2}
    players_df = pl.DataFrame([_WEAK_WR_ROW, _UNROSTERED_WR_ROW, KEENUM_ROW, MAHOMES_ROW])
    raw_dir, persist_dir, signals_dir = _setup(
        tmp_path, monkeypatch, my_roster, [_WEAK_WR_SIGNAL_ROW, cameo, KEENUM_SIGNAL, MAHOMES_SIGNAL], players_df
    )
    _add_to_players_json(raw_dir, {
        "s_chase": {"full_name": "Ja'Marr Chase", "position": "WR", "team": "CIN"},
        "s_keenum": {"full_name": "Case Keenum", "position": "QB", "team": None, "depth_chart_order": None},  # a free agent
        # Mahomes: unmatched/unlisted on the depth chart -> not a pickup either
        "s_mahomes": {"full_name": "Patrick Mahomes", "position": "QB", "team": "KC", "depth_chart_order": None},
    })

    result = report.generate_report(
        "waiver_pickups", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )

    assert [e["name"] for e in result["entries"]] == []
    notes = " ".join(result["notes"])
    assert f"fewer than {report.MIN_PICKUP_PLAYS} plays" in notes
    assert "not on an NFL roster in Sleeper's data: Case Keenum" in notes
    assert "no depth-chart listing" in notes


# ---- bye weeks (found by the frontend check of this same fix: KC on bye, Mahomes started in SUPER_FLEX) ----


def test_on_bye_reads_this_weeks_table_first_and_the_bye_map_otherwise():
    assert ranking.on_bye({"season": 2026, "as_of_week": 5, "opponent": None}, 2026, 5)
    assert not ranking.on_bye({"season": 2026, "as_of_week": 5, "opponent": "DET"}, 2026, 5, "KC", {"KC": 5})
    # a stale / proxy-only row says nothing about this week: fall back to the bye map by team
    assert ranking.on_bye({"season": 2025, "as_of_week": 19, "opponent": None, "stale": True}, 2026, 5, "KC", {"KC": 5})
    assert not ranking.on_bye({"player_id": "x", "proxy_only": True}, 2026, 5, "KC", {"KC": 6})
    assert not ranking.on_bye(None, 2026, 5, "KC", None)


def test_start_sit_leaves_a_bye_week_player_out_and_says_so(tmp_path, monkeypatch):
    roster = {
        "s_keenum": {"full_name": "Case Keenum", "position": "QB", "team": "CHI", "depth_chart_order": 1},
        "s_mahomes": {"full_name": "Patrick Mahomes", "position": "QB", "team": "KC", "depth_chart_order": 1},
    }
    mahomes_on_bye = {**MAHOMES_SIGNAL, "opponent": None, "implied_total": None, "run_funnel_rate_vs_avg": None}
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [KEENUM_SIGNAL, mahomes_on_bye], pl.DataFrame([KEENUM_ROW, MAHOMES_ROW]))
    monkeypatch.setattr(recommend.RecommendContext, "bye_weeks", lambda self: {"KC": _WEEK})

    result = report.generate_report(
        "start_sit", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )
    assert result["entries"] == []  # one QB left after the bye exclusion: nothing to decide
    assert any(f"On bye in week {_WEEK}, left out of the lineup: Patrick Mahomes (KC)" in n for n in result["notes"])

    ctx = recommend.RecommendContext(raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK,
                                     player_idx=__import__("src.rag.player_index", fromlist=["x"]).build_player_index(_SEASON),
                                     signals_dir=signals_dir)
    ranked = recommend.dispatch_tool("rank_players", {"player_names": ["Patrick Mahomes", "Case Keenum"]}, ctx)
    assert [u["name"] for u in ranked["unavailable"]] == ["Patrick Mahomes"] and ranked["unavailable"][0]["reason"] == "bye"
    assert [e["name"] for e in ranked["ranked"]] == ["Case Keenum"]
    assert "on bye" in recommend._build_system_prompt({"name": "L"}, {"rec": 0.5}, 2026, 5)


# ---- 2026-10-06: "Saquon Barkley shows no injury status and is healthy" (he was Out, Hamstring in Sleeper's data) ----


def test_get_player_signals_carries_sleepers_designation_for_any_player_and_when_it_was_pulled(tmp_path, monkeypatch):
    from tests.test_report import _BARKLEY_ROW, _BARKLEY_SIGNAL_ROW

    my_roster = {"sleeper_wilson": {"full_name": "Jaylen Wilson", "position": "WR", "team": "MIA"}}
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, my_roster, [_WEAK_WR_SIGNAL_ROW, _BARKLEY_SIGNAL_ROW],
                                               pl.DataFrame([_WEAK_WR_ROW, _BARKLEY_ROW]))
    # Barkley is rostered by nobody in this fixture league; Sleeper's data still carries his designation.
    _add_to_players_json(raw_dir, {"s_barkley": {"full_name": "Saquon Barkley", "position": "RB", "team": "PHI",
                                                 "injury_status": "Out", "injury_body_part": "Hamstring", "depth_chart_order": 1}})
    from src.rag import player_index
    ctx = recommend.RecommendContext(raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK,
                                     player_idx=player_index.build_player_index(_SEASON), signals_dir=signals_dir)

    out = recommend.dispatch_tool("get_player_signals", {"player_name": "Saquon Barkley"}, ctx)

    assert out["resolved"] and out["has_signals"]
    assert out["injury_status"] == "Out" and out["injury_body_part"] == "Hamstring" and out["depth_chart_order"] == 1
    assert out["unavailable_this_week"] is True
    assert out["sleeper_data_as_of"] == "2026-01-01T00:00:00Z"  # the fixture's fetched_at
    assert out["report_status"] is None  # the NFL report has nothing: the two sources are kept apart
    healthy = recommend.dispatch_tool("get_player_signals", {"player_name": "Jaylen Wilson"}, ctx)
    assert healthy["injury_status"] is None and healthy["unavailable_this_week"] is False
    roster = recommend.dispatch_tool("get_my_roster", {}, ctx)
    assert roster["sleeper_data_as_of"] == "2026-01-01T00:00:00Z"
    prompt = recommend._build_system_prompt({"name": "L"}, {"rec": 0.5}, 2026, 5)
    assert "NEVER 'healthy'" in prompt and "sleeper_data_as_of" in prompt and "unavailable_this_week" in prompt
