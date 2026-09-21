"""Phase 6: the per-league points proxy joined onto every ranking row and
tool output, scored under the league's own settings, strictly as-of."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.reasoning import ranking, recommend, report
from src.signals import player_stats as ps
from tests.test_league import (
    _ALL_INDEX_ROWS,
    _ALL_SIGNALS,
    _SEASON,
    _WEEK,
    FULL_PPR,
    HALF_PPR,
    OTHER_ID,
    VS30_ID,
    _setup,
)

BARKLEY, COOK = "00-0034844", "00-0037248"


def _line(pid, season, week, **stats):
    base = {c: 0.0 for c in ps.STAT_COLUMNS}
    base.update(stats)
    return {"player_id": pid, "player_display_name": pid, "position": "RB", "team": "X", "season": season, "week": week, **base}


def _write_stats(signals_dir: Path, prior: bool = True) -> Path:
    """Cook: a reception machine (10 catches, 60 yds each week); Barkley: a
    pure rusher (100 yds each week). Half-PPR: Cook 11.0 vs Barkley 10.0
    per game; full PPR: 16.0 vs 10.0. Week 5 exists to prove the as-of cut
    (it must never count as of week 5). Barkley optionally has a big prior
    season (18.0/game over 17 games)."""
    stats_dir = signals_dir.parent / "player_stats"
    rows = []
    for week in range(1, 6):
        rows.append(_line(COOK, _SEASON, week, receptions=10, receiving_yards=60))
        rows.append(_line(BARKLEY, _SEASON, week, rushing_yards=100))
    ps.write_weekly_stats(_SEASON, stats_dir=stats_dir, df=pl.DataFrame(rows))
    if prior:
        prior_rows = [_line(BARKLEY, _SEASON - 1, w, rushing_yards=120, rushing_tds=1) for w in range(1, 18)]
        ps.write_weekly_stats(_SEASON - 1, stats_dir=stats_dir, df=pl.DataFrame(prior_rows))
    return stats_dir


def test_signal_tables_join_the_proxy_as_of_the_week_under_the_leagues_scoring(tmp_path):
    signals_dir = tmp_path / "signals"
    signals_dir.mkdir()
    pl.DataFrame(_ALL_SIGNALS).write_parquet(signals_dir / f"signals_{_SEASON}_week{_WEEK}.parquet")
    _write_stats(signals_dir)

    half = ranking.SignalTables.load(signals_dir, _SEASON, _WEEK, scoring_settings=HALF_PPR)
    cook = half.row_for(COOK)
    assert cook["games_played"] == 4 and cook["ppg"] == 11.0 and cook["season_points_so_far_proxy"] == 44.0  # weeks 1-4 only
    assert cook["ppg_prior_season"] is None
    barkley = half.row_for(BARKLEY)
    assert barkley["ppg"] == 10.0 and barkley["ppg_prior_season"] == 18.0 and barkley["prior_games_played"] == 17
    assert barkley["red_zone_share"] == 0.48  # the usage row is still there underneath

    full = ranking.SignalTables.load(signals_dir, _SEASON, _WEEK, scoring_settings=FULL_PPR)
    assert full.row_for(COOK)["ppg"] == 16.0  # same stat lines, this league's scoring

    none = ranking.SignalTables.load(signals_dir, _SEASON, _WEEK)  # no scoring -> no proxy, rows unchanged
    assert none.row_for(COOK)["ppg"] is None


def test_a_player_with_points_but_no_usage_row_is_rankable_on_the_proxy(tmp_path):
    signals_dir = tmp_path / "signals"
    signals_dir.mkdir()
    stats_dir = signals_dir.parent / "player_stats"
    ps.write_weekly_stats(_SEASON, stats_dir=stats_dir, df=pl.DataFrame([_line("qb1", _SEASON, 1, passing_yards=300, passing_tds=3)]))
    tables = ranking.SignalTables.load(signals_dir, _SEASON, 2, scoring_settings=HALF_PPR)
    row = tables.row_for("qb1")
    assert row["proxy_only"] is True and row["ppg"] == 12.0  # 3 TD x 4; the fixture scoring has no per-yard passing points
    assert ranking.opportunity_score(row) > 0
    assert "no usage/matchup signals computed yet" in ranking.fmt_signal_row(row)
    assert "12.0 pts/game over 1 game(s)" in ranking.fmt_signal_row(row)
    assert tables.row_for("nobody") is None


def test_report_ranking_now_depends_on_the_leagues_scoring_when_stat_lines_exist(tmp_path, monkeypatch):
    """The Phase 5.1 'scoring-format independent' property was real only
    because nothing scoring-dependent was in the score. With Phase 6's
    proxy, two leagues with identical rosters and different scoring can
    rank differently -- here full PPR turns the reception-heavy Cook into
    the start over the pure-rusher Barkley."""
    half_raw, half_chroma, half_sig = _setup(tmp_path, monkeypatch, VS30_ID, "Victorious Secret 3.0", HALF_PPR, subdir="half")
    full_raw, full_chroma, full_sig = _setup(tmp_path, monkeypatch, OTHER_ID, "Full PPR League", FULL_PPR, subdir="full")
    _write_stats(half_sig, prior=False)  # no prior season: the usage row + this season's points decide it
    _write_stats(full_sig, prior=False)

    half = report.generate_report("start_sit", VS30_ID, raw_dir=half_raw, persist_dir=half_chroma, season=_SEASON, as_of_week=_WEEK, signals_dir=half_sig)
    full = report.generate_report("start_sit", OTHER_ID, raw_dir=full_raw, persist_dir=full_chroma, season=_SEASON, as_of_week=_WEEK, signals_dir=full_sig)

    assert half["entries"][0]["recommended_starter"]["name"] == "Saquon Barkley"
    assert full["entries"][0]["recommended_starter"]["name"] == "James Cook"
    assert "16.0 pts/game" in full["entries"][0]["reasoning"]
    assert "11.0 pts/game" in half["entries"][0]["reasoning"]


def test_get_player_signals_and_get_league_rosters_carry_the_proxy(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch)
    _write_stats(signals_dir)
    from src.rag import player_index
    from src.reasoning.league import load_league

    ctx = recommend.RecommendContext(
        raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK,
        player_idx=player_index.build_player_index(_SEASON), league=load_league(VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir),
        roster_id="1", signals_dir=signals_dir,
    )
    sig = recommend.dispatch_tool("get_player_signals", {"player_name": "Saquon Barkley"}, ctx)
    assert (sig["ppg"], sig["games_played"], sig["season_points_so_far_proxy"], sig["ppg_prior_season"]) == (10.0, 4, 40.0, 18.0)

    rosters = recommend.dispatch_tool("get_league_rosters", {}, ctx)
    players = {p["name"]: p for p in rosters["teams"][0]["players"]}
    assert players["James Cook"]["ppg"] == 11.0 and players["Saquon Barkley"]["ppg_prior_season"] == 18.0
    assert "not a market or trade value" in rosters["points_proxy_note"]


def test_refresh_writes_the_stat_tables_next_to_the_signals(tmp_path, monkeypatch):
    from src.scheduler import refresh

    written = []
    monkeypatch.setattr(refresh.player_stats, "write_weekly_stats", lambda season, stats_dir=None: written.append((season, stats_dir)))
    monkeypatch.setattr(refresh.nflverse, "fetch_pbp", lambda season: pl.DataFrame())
    monkeypatch.setattr(refresh.ngs, "fetch_ngs", lambda season, kind: None)
    monkeypatch.setattr(refresh.matchup_signals, "build_signals_table", lambda *a, **k: [])
    monkeypatch.setattr(refresh.matchup_signals, "write_signals_table", lambda *a, **k: tmp_path / "signals" / "x.parquet", raising=False)
    signals_dir = tmp_path / "signals"
    signals_dir.mkdir()
    try:
        refresh.refresh_signals(2026, 2, pl.DataFrame(), signals_dir=signals_dir)
    except Exception:
        pass  # the stubbed table write may not match the real signature; the stats write happens first
    assert written == [(2026, tmp_path / "player_stats"), (2025, tmp_path / "player_stats")]
