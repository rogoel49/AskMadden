"""Kickers and team defenses (2026-10-06): "what about the Bucs defense?
that surely can be updated". Weekly lines from nflverse (unit_stats),
scored per league (points_proxy), joined into the signal tables as
proxy-only rows, ranked on points per game alone in start/sit and the
waiver pool, and resolvable in chat by team name."""
import polars as pl

from src.rag import teams
from src.reasoning import points_proxy, recommend, report
from src.reasoning.ranking import SignalTables
from src.signals import unit_stats
from tests.test_report import _COOK_ROW, _COOK_SIGNAL_ROW, _LEAGUE_ID, _SEASON, _WEEK, _setup

SCORING = {"rec": 0.5, "sack": 1, "int": 2, "ff": 1, "fum_rec": 2, "def_td": 6, "def_st_td": 6, "safe": 2, "blk_kick": 2,
           "pts_allow_0": 10, "pts_allow_1_6": 7, "pts_allow_7_13": 4, "pts_allow_14_20": 1, "pts_allow_21_27": 0,
           "pts_allow_28_34": -1, "pts_allow_35p": -4,
           "fgm_0_19": 3, "fgm_20_29": 3, "fgm_30_39": 3, "fgm_40_49": 4, "fgm_50_59": 5, "fgm_60p": 6, "fgmiss": -1,
           "xpm": 1, "xpmiss": -1}


def _kickers():
    return pl.DataFrame({
        "player_id": ["k1", "k1", "k2"], "player_display_name": ["Chris Boswell", "Chris Boswell", "Jake Moody"],
        "position": ["K", "K", "K"], "team": ["PIT", "PIT", "SF"], "opponent_team": ["CIN", "CLE", "LA"],
        "season": [_SEASON] * 3, "week": [1, 2, 1],
        "fg_made_0_19": [0.0, 0, 0], "fg_made_20_29": [1.0, 0, 0], "fg_made_30_39": [0.0, 1, 0], "fg_made_40_49": [1.0, 0, 0],
        "fg_made_50_59": [0.0, 1, 0], "fg_made_60_": [0.0, 0, 0], "fg_made": [2.0, 2, 0], "fg_missed": [0.0, 1, 2],
        "fg_blocked": [0.0, 0, 1], "pat_made": [3.0, 1, 2], "pat_missed": [0.0, 0, 1], "pat_blocked": [0.0, 0, 0],
    })


def _defenses():
    return pl.DataFrame({
        "player_id": ["TB", "TB", "KC"], "player_display_name": ["TB", "TB", "KC"], "position": ["DEF"] * 3,
        "team": ["TB", "TB", "KC"], "opponent_team": ["CIN", "CLE", "LAC"], "season": [_SEASON] * 3, "week": [1, 2, 1],
        "sacks": [3.0, 2, 1], "interceptions": [1.0, 0, 0], "fumbles_forced": [1.0, 0, 0], "fumbles_recovered": [1.0, 0, 0],
        "defensive_tds": [1.0, 0, 0], "special_teams_tds": [0.0, 0, 0], "safeties": [0.0, 0, 0], "blocked_kicks": [0.0, 1, 0],
        "points_allowed": [33.0, 10, 24],
    })


def test_kicker_and_defense_lines_score_under_a_leagues_settings():
    k = _kickers().select("player_id", "week", points_proxy.kicker_points_expr(SCORING)).to_dicts()
    # Boswell wk1: 20-29 (3) + 40-49 (4) + 3 XP = 10; wk2: 30-39 (3) + 50-59 (5) - 1 miss + 1 XP = 8
    # Moody wk1: 2 misses + 1 blocked = -3, 2 XP + 1 XP missed = 1 -> -2
    assert {(r["player_id"], r["week"]): r["points"] for r in k} == {("k1", 1): 10.0, ("k1", 2): 8.0, ("k2", 1): -2.0}
    d = _defenses().select("player_id", "week", points_proxy.defense_points_expr(SCORING)).to_dicts()
    # TB wk1: 3 sacks + 2 int + 1 ff + 2 fr + 6 td, 33 allowed (-1) = 13; wk2: 2 sacks + 2 blk, 10 allowed (4) = 8
    # KC wk1: 1 sack, 24 allowed (0) = 1
    assert {(r["player_id"], r["week"]): r["points"] for r in d} == {("TB", 1): 13.0, ("TB", 2): 8.0, ("KC", 1): 1.0}
    proxy = points_proxy.unit_points_proxy(_kickers(), _defenses(), SCORING, as_of_week=3)
    assert proxy["TB"] == {"games_played": 2, "season_points_so_far_proxy": 21.0, "ppg": 10.5, "team": "TB", "unit": "DEF"}
    assert proxy["k1"]["ppg"] == 9.0 and proxy["k1"]["unit"] == "K" and proxy["k1"]["team"] == "PIT"
    assert points_proxy.unit_points_proxy(_kickers(), _defenses(), SCORING, as_of_week=2)["TB"]["games_played"] == 1  # as-of


def test_a_league_that_scores_fifty_plus_as_one_bucket_is_honored():
    scoring = {"fgm_50p": 5, "fgm_0_19": 3, "fgm_20_29": 3, "fgm_30_39": 3, "fgm_40_49": 4}
    k = _kickers().select("player_id", "week", points_proxy.kicker_points_expr(scoring)).to_dicts()
    assert {(r["player_id"], r["week"]): r["points"] for r in k}[("k1", 2)] == 8.0  # 30-39 (3) + 50-59 via fgm_50p (5)


def test_build_defense_stats_joins_the_opponents_fumbles_and_the_final_score():
    ts = pl.DataFrame({
        "season": [2026, 2026], "week": [1, 1], "game_id": ["g", "g"], "season_type": ["REG", "REG"],
        "team": ["TB", "CIN"], "opponent_team": ["CIN", "TB"],
        "def_sacks": [3, 1], "def_interceptions": [1, 0], "def_fumbles_forced": [1, 0], "def_tds": [1, 0],
        "special_teams_tds": [0, 0], "def_safeties": [0, 0], "def_punt_blocks": [0, 0], "def_pat_blocks": [0, 1], "def_fg_blocks": [0, 0],
        "rushing_fumbles_lost": [0, 1], "receiving_fumbles_lost": [0, 0], "sack_fumbles_lost": [1, 0],
    })
    sched = pl.DataFrame({"game_id": ["g"], "home_team": ["TB"], "away_team": ["CIN"], "home_score": [27], "away_score": [33]})
    out = {r["player_id"]: r for r in unit_stats.build_defense_stats(ts, sched).to_dicts()}
    assert out["TB"]["fumbles_recovered"] == 1.0 and out["TB"]["points_allowed"] == 33.0 and out["TB"]["sacks"] == 3.0
    assert out["CIN"]["fumbles_recovered"] == 1.0 and out["CIN"]["points_allowed"] == 27.0 and out["CIN"]["blocked_kicks"] == 1.0
    assert out["TB"]["position"] == "DEF" and out["TB"]["player_id"] == "TB"


def test_team_names_resolve_to_the_defense_unit():
    assert teams.resolve_team("bucs defense") == "TB" and teams.resolve_team("Tampa Bay D/ST") == "TB"
    assert teams.resolve_team("the Jets") == "NYJ" and teams.resolve_team("New York") is None
    assert teams.resolve_team("Patrick Mahomes") is None and teams.resolve_team("Josh Jacobs") is None
    assert teams.defense_display_name("TB") == "Buccaneers D/ST"


def _write_units(signals_dir, raw_dir=None):
    units_dir = signals_dir.parent / "unit_stats"
    units_dir.mkdir()
    _kickers().write_parquet(unit_stats.kickers_path(_SEASON, units_dir))
    _defenses().write_parquet(unit_stats.defenses_path(_SEASON, units_dir))
    if raw_dir is not None:  # give the fixture league real K/DEF scoring keys
        import json
        path = raw_dir / "league.json"
        payload = json.loads(path.read_text())
        payload["data"]["scoring_settings"].update(SCORING)
        path.write_text(json.dumps(payload))


def test_signal_tables_carry_unit_rows_with_the_matchup_as_context_only(tmp_path, monkeypatch):
    roster = {"sleeper_cook": {"full_name": "James Cook", "position": "RB", "team": "BUF"}}
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [{**_COOK_SIGNAL_ROW, "team": "TB", "opponent": "KC"},
                                                                                {**_COOK_SIGNAL_ROW, "player_id": "x", "team": "KC", "opponent": "TB", "implied_total": 20.0}],
                                               pl.DataFrame([_COOK_ROW]))
    _write_units(signals_dir)
    tables = SignalTables.load(signals_dir, _SEASON, _WEEK, scoring_settings=SCORING)
    tb = tables.row_for("TB")
    assert tb["unit"] == "DEF" and tb["ppg"] == 10.5 and tb["games_played"] == 2 and tb["stale"] is False
    assert tb["opponent"] == "KC" and tb["opponent_implied_total"] == 20.0 and "implied_total" not in tb
    from src.reasoning.ranking import opportunity_score, PPG_WEIGHT
    assert abs(opportunity_score(tb) - PPG_WEIGHT * 10.5) < 1e-9  # points per game alone
    assert tables.row_for("k1")["unit"] == "K" and tables.row_for("k1")["team"] == "PIT"


def test_start_sit_ranks_two_defenses_and_the_waiver_list_offers_the_rest(tmp_path, monkeypatch):
    roster = {"TB": {"position": "DEF", "team": "TB"}, "KC": {"position": "DEF", "team": "KC"},
              "sleeper_boswell": {"full_name": "Chris Boswell", "position": "K", "team": "PIT"}}
    players_df = pl.DataFrame([_COOK_ROW, {"gsis_id": "k1", "display_name": "Chris Boswell", "position": "K", "latest_team": "PIT", "last_season": 2026},
                               {"gsis_id": "k2", "display_name": "Jake Moody", "position": "K", "latest_team": "SF", "last_season": 2026}])
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [_COOK_SIGNAL_ROW], players_df)
    _write_units(signals_dir, raw_dir)
    monkeypatch.setattr(report.player_index, "build_player_index", lambda season, players=None: pl.DataFrame(
        {"player_id": ["00-0037248", "k1", "k2"], "player_name": ["James Cook", "Chris Boswell", "Jake Moody"],
         "position": ["RB", "K", "K"], "team": ["BUF", "PIT", "SF"]}))

    ss = report.generate_report("start_sit", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir)
    entry = [e for e in ss["entries"] if e["position"] == "DEF"][0]
    assert entry["recommended_starter"]["name"] == "Buccaneers D/ST" and entry["recommended_starter"]["sleeper_id"] == "TB"
    assert entry["alternatives_considered"][0]["name"] == "Chiefs D/ST"
    assert "10.5 pts/game" in entry["reasoning"]

    wv = report.generate_report("waiver_pickups", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir)
    names = [(e["name"], e["position"]) for e in wv["entries"]]
    assert ("Jake Moody", "K") in names  # the unrostered kicker with a line this season
    assert not any(n in ("Buccaneers D/ST", "Chiefs D/ST") for n, _ in names)  # rostered defenses excluded
    assert all(p != "DEF" for _, p in names)  # no other defense has a line in this fixture

    drop = report.generate_report("drop", _LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir)
    assert all(e["position"] not in ("K", "DEF") for e in drop["entries"])
    assert any("streamed" in n or "aren't drop candidates" in n for n in drop["notes"])


def test_chat_resolves_a_defense_by_team_name_and_ranks_it_on_points(tmp_path, monkeypatch):
    roster = {"sleeper_cook": {"full_name": "James Cook", "position": "RB", "team": "BUF"}}
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch, roster, [_COOK_SIGNAL_ROW], pl.DataFrame([_COOK_ROW]))
    _write_units(signals_dir)
    from src.rag import player_index
    from src.reasoning.league import load_league
    ctx = recommend.RecommendContext(raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK,
                                     player_idx=player_index.build_player_index(_SEASON), signals_dir=signals_dir,
                                     league=load_league(_LEAGUE_ID, raw_dir=raw_dir, persist_dir=persist_dir))
    ctx.league.scoring_settings.update(SCORING)
    sig = recommend.dispatch_tool("get_player_signals", {"player_name": "bucs defense"}, ctx)
    assert sig["resolved"] and sig["player_id"] == "TB" and sig["position"] == "DEF" and sig["has_signals"]
    assert sig["ppg"] == 10.5 and "unit_note" in sig
    ranked = recommend.dispatch_tool("rank_players", {"player_names": ["Tampa Bay D/ST", "Chiefs"]}, ctx)
    assert ranked["verdict"] == "clear" and ranked["recommended"]["player_id"] == "TB"
    assert "KICKERS AND DEFENSES" in recommend._build_system_prompt({"name": "L"}, {"rec": 0.5}, 2026, 5)
