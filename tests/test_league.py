"""Phase 5.1: league/scoring parameterization.

What these tests pin down, and what they deliberately don't claim:
- recommend()/generate_report() take an explicit league_id and refuse
  (LeagueMismatchError, before any model call) when the ingested data
  on disk belongs to a different league -- the silent wrong-league
  answer Phase 5.1 exists to prevent.
- Scoring settings come from the league's own ingested league.json,
  verbatim, and reach the system prompt: two leagues with different
  scoring produce different prompts, and a non-PPR league's prompt
  never mentions a reception value it doesn't have. There is no
  hardcoded half-PPR anywhere to bypass (Phase 5.1's trace confirmed
  that; see src/reasoning/league.py's docstring).
- generate_report()'s ranking is scoring-format-INDEPENDENT today: two
  leagues differing only in scoring get identical report entries. That
  test documents a real property of the current code (report.py never
  consumed scoring settings; the opportunity score is usage/efficiency
  only) rather than asserting a scoring-aware ranking that doesn't
  exist -- see generate_report()'s docstring.
- evals/build_ground_truth.py reads scoring the same way, through the
  same loader, so ground truth and the agent can't be scored against
  two different leagues' settings.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from evals import build_ground_truth as bgt
from src.rag import embed
from src.reasoning import league as league_module
from src.reasoning import recommend, report
from src.reasoning.league import LeagueConfig, LeagueMismatchError, load_league

VS30_ID = "1389341490030862336"  # Victorious Secret 3.0 (half-PPR)
OTHER_ID = "9999999999999999999"  # a second, full-PPR league (fixture only -- no real second league exists in this sandbox)

HALF_PPR = {"rec": 0.5, "pass_td": 4, "rush_td": 6, "rec_td": 6, "rec_yd": 0.1, "rush_yd": 0.1, "fum_lost": -2}
FULL_PPR = {**HALF_PPR, "rec": 1.0, "bonus_rec_te": 0.5}
STANDARD = {k: v for k, v in HALF_PPR.items() if k != "rec"}  # no per-reception value at all

_SEASON, _WEEK = 2024, 5

_BARKLEY_ROW = {
    "gsis_id": "00-0034844", "display_name": "Saquon Barkley", "position": "RB", "latest_team": "PHI", "last_season": 2026,
}
_COOK_ROW = {
    "gsis_id": "00-0037248", "display_name": "James Cook", "position": "RB", "latest_team": "BUF", "last_season": 2026,
}
# An unrostered player so waiver_pickups has a non-empty pool (same fixture tests/test_report.py uses).
_CHASE_ROW = {
    "gsis_id": "00-0036900", "display_name": "Ja'Marr Chase", "position": "WR", "latest_team": "CIN", "last_season": 2026,
}
# Real 2024 week-5 values from the committed signals table (same rows tests/test_report.py uses).
_BARKLEY_SIGNAL = {
    "player_id": "00-0034844", "player_name": "S.Barkley", "team": "PHI", "season": _SEASON, "as_of_week": _WEEK,
    "season_plays": 90, "epa_trend": 0.006847, "red_zone_share": 0.48, "target_share": 0.121951,
    "target_share_adjusted": None, "opponent": None, "run_funnel_rate_vs_avg": None, "implied_total": None,
}
_COOK_SIGNAL = {
    "player_id": "00-0037248", "player_name": "J.Cook", "team": "BUF", "season": _SEASON, "as_of_week": _WEEK,
    "season_plays": 60, "epa_trend": 0.119438, "red_zone_share": 0.264706, "target_share": 0.102041,
    "target_share_adjusted": 0.098492, "opponent": "HOU", "run_funnel_rate_vs_avg": 0.02, "implied_total": 23.5,
}
_CHASE_SIGNAL = {
    "player_id": "00-0036900", "player_name": "J.Chase", "team": "CIN", "season": _SEASON, "as_of_week": _WEEK,
    "season_plays": 24, "epa_trend": -0.045437, "red_zone_share": 0.116279, "target_share": 0.181818,
    "target_share_adjusted": 0.186837, "opponent": "BAL", "run_funnel_rate_vs_avg": -0.03, "implied_total": 23.0,
}
_ALL_SIGNALS = [_BARKLEY_SIGNAL, _COOK_SIGNAL, _CHASE_SIGNAL]
_ALL_INDEX_ROWS = [_BARKLEY_ROW, _COOK_ROW, _CHASE_ROW]


def _write(raw_dir: Path, filename: str, data) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / filename).write_text(json.dumps({"fetched_at": "2026-01-01T00:00:00Z", "source": "sleeper", "data": data}))


def _seed_league(raw_dir: Path, league_id: str, name: str, scoring: dict) -> None:
    """One ingested league in raw_dir, shaped like src/ingest/sleeper.py's
    output. Roster/players are identical across leagues on purpose, so
    any difference in output between two seeded leagues is attributable
    to league identity/scoring alone."""
    _write(
        raw_dir,
        "league.json",
        {
            "league_id": league_id,
            "name": name,
            "season": str(_SEASON),
            "settings": {"num_teams": 12},
            "scoring_settings": scoring,
            "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN"],
        },
    )
    _write(
        raw_dir,
        "teams.json",
        [
            {
                "roster_id": 1, "owner_id": "u1", "display_name": "rogoel49", "team_name": "Victorious Secret",
                "players": ["s_barkley", "s_cook"], "starters": ["s_barkley"], "settings": {"wins": 3, "losses": 1, "ties": 0},
            }
        ],
    )
    _write(
        raw_dir,
        "players.json",
        {
            "s_barkley": {"full_name": "Saquon Barkley", "position": "RB", "team": "PHI"},
            "s_cook": {"full_name": "James Cook", "position": "RB", "team": "BUF"},
        },
    )
    _write(raw_dir, "nfl_state.json", {"season": str(_SEASON), "display_week": _WEEK})
    _write(raw_dir, f"matchups_week_{_WEEK}.json", [{"roster_id": 1, "matchup_id": 1, "points": 0}])


def _setup(tmp_path, monkeypatch, league_id=VS30_ID, name="Victorious Secret 3.0", scoring=HALF_PPR, subdir="a"):
    raw_dir = tmp_path / subdir / "sleeper"
    persist_dir = tmp_path / subdir / "chroma"
    signals_dir = tmp_path / subdir / "signals"
    _seed_league(raw_dir, league_id, name, scoring)
    embed.embed(embed.build_signal_chunks(_ALL_SIGNALS), persist_dir=persist_dir)
    signals_dir.mkdir(parents=True)
    pl.DataFrame(_ALL_SIGNALS).write_parquet(signals_dir / f"signals_{_SEASON}_week{_WEEK}.parquet")

    import src.rag.player_index as player_index_module

    monkeypatch.setattr(player_index_module.nflverse, "fetch_players", lambda: pl.DataFrame(_ALL_INDEX_ROWS))
    monkeypatch.setenv("MY_ROSTER_ID", "1")
    return raw_dir, persist_dir, signals_dir


# ---- fake Claude client that records what it was shown ----


def _tool_use_block(name, input_, id_="tool_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_)


class _RecordingClient:
    """Records every `system` prompt it's given; replays scripted responses."""

    def __init__(self, responses):
        self.system_prompts: list[str] = []
        self.calls = 0
        outer = self

        class _Messages:
            def create(self_inner, **kwargs):
                outer.calls += 1
                outer.system_prompts.append(kwargs["system"])
                return responses.pop(0)

        self.messages = _Messages()


def _submit_only_client():
    return _RecordingClient(
        [
            SimpleNamespace(
                content=[
                    _tool_use_block(
                        "submit_recommendation",
                        {"recommendation": "Start Saquon Barkley.", "reasoning": "grounded", "player_id": "00-0034844"},
                    )
                ]
            )
        ]
    )


# ---- load_league: the one place scoring settings are read ----


def test_load_league_returns_the_leagues_real_scoring_settings_verbatim(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_league(raw_dir, VS30_ID, "Victorious Secret 3.0", HALF_PPR)

    cfg = load_league(VS30_ID, raw_dir=raw_dir, persist_dir=tmp_path / "chroma")

    assert isinstance(cfg, LeagueConfig)
    assert cfg.league_id == VS30_ID
    assert cfg.name == "Victorious Secret 3.0"
    assert cfg.scoring_settings == HALF_PPR  # exactly what's on disk -- no defaults merged in, no format label
    assert cfg.roster_positions[0] == "QB"
    assert cfg.raw_dir == raw_dir


@pytest.mark.parametrize("scoring", [HALF_PPR, FULL_PPR, STANDARD], ids=["half_ppr", "full_ppr", "standard"])
def test_load_league_passes_through_any_scoring_format_not_just_half_ppr(tmp_path, scoring):
    """Half-PPR is the first parameterized case, not a special one: the
    loader has no idea what format it's reading and never fills in 0.5."""
    raw_dir = tmp_path / "sleeper"
    _seed_league(raw_dir, OTHER_ID, "Some Other League", scoring)

    cfg = load_league(OTHER_ID, raw_dir=raw_dir)

    assert cfg.scoring_settings == scoring
    assert cfg.scoring_settings.get("rec") == scoring.get("rec")


def test_load_league_refuses_a_directory_holding_a_different_league(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_league(raw_dir, OTHER_ID, "Some Other League", FULL_PPR)

    with pytest.raises(LeagueMismatchError) as excinfo:
        load_league(VS30_ID, raw_dir=raw_dir)

    message = str(excinfo.value)
    assert OTHER_ID in message and VS30_ID in message  # names both, so the fix is obvious
    assert f"--league-id {VS30_ID}" in message  # copy-pasteable re-ingest command


def test_load_league_refuses_a_league_json_with_no_league_id_rather_than_assuming(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _write(raw_dir, "league.json", {"name": "No ID League", "scoring_settings": HALF_PPR})

    with pytest.raises(LeagueMismatchError, match="no league_id field"):
        load_league(VS30_ID, raw_dir=raw_dir)


def test_load_league_requires_ingest_to_have_run(tmp_path):
    with pytest.raises(RuntimeError, match=r"run `python -m src.ingest.sleeper --league-id"):
        load_league(VS30_ID, raw_dir=tmp_path / "nonexistent")


@pytest.mark.parametrize("bad", [None, "", "   "])
def test_load_league_requires_a_league_id(tmp_path, bad):
    raw_dir = tmp_path / "sleeper"
    _seed_league(raw_dir, VS30_ID, "Victorious Secret 3.0", HALF_PPR)
    with pytest.raises(ValueError, match="league_id is required"):
        load_league(bad, raw_dir=raw_dir)


def test_load_league_matches_numeric_ids_by_string_value(tmp_path):
    """Sleeper returns league_id as a string; a caller passing the same
    number as an int (e.g. from a config file) must not be treated as a
    different league."""
    raw_dir = tmp_path / "sleeper"
    _seed_league(raw_dir, VS30_ID, "Victorious Secret 3.0", HALF_PPR)
    assert load_league(int(VS30_ID), raw_dir=raw_dir).league_id == VS30_ID


# ---- recommend(): league_id is explicit, verified, and echoed ----


def test_recommend_refuses_wrong_league_before_any_model_call(tmp_path, monkeypatch):
    raw_dir, persist_dir, _ = _setup(tmp_path, monkeypatch)  # holds Victorious Secret 3.0
    client = _submit_only_client()

    with pytest.raises(LeagueMismatchError):
        recommend.recommend("Should I start Saquon Barkley?", OTHER_ID, raw_dir=raw_dir, persist_dir=persist_dir, client=client)

    assert client.calls == 0  # refused before spending a model call on the wrong league


def test_recommend_result_echoes_the_league_it_answered_for(tmp_path, monkeypatch):
    raw_dir, persist_dir, _ = _setup(tmp_path, monkeypatch)
    client = _submit_only_client()

    result = recommend.recommend(
        "Should I start Saquon Barkley?", VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir,
        season=_SEASON, as_of_week=_WEEK, client=client,
    )

    assert result["league_id"] == VS30_ID
    assert result["recommendation"] == "Start Saquon Barkley."


def test_recommend_system_prompt_carries_that_leagues_own_scoring_not_half_ppr(tmp_path, monkeypatch):
    """Same roster, same signals, two leagues that differ only in scoring
    -> the model is shown each league's own settings. Neither prompt is
    a hardcoded half-PPR: the non-PPR league's prompt contains no
    reception value at all."""
    half_raw, half_chroma, _ = _setup(tmp_path, monkeypatch, VS30_ID, "Victorious Secret 3.0", HALF_PPR, subdir="half")
    std_raw, std_chroma, _ = _setup(tmp_path, monkeypatch, OTHER_ID, "Standard League", STANDARD, subdir="std")

    half_client, std_client = _submit_only_client(), _submit_only_client()
    recommend.recommend("q", VS30_ID, raw_dir=half_raw, persist_dir=half_chroma, season=_SEASON, as_of_week=_WEEK, client=half_client)
    recommend.recommend("q", OTHER_ID, raw_dir=std_raw, persist_dir=std_chroma, season=_SEASON, as_of_week=_WEEK, client=std_client)

    half_prompt, std_prompt = half_client.system_prompts[0], std_client.system_prompts[0]
    assert half_prompt != std_prompt
    assert '"Victorious Secret 3.0"' in half_prompt and "rec=0.5" in half_prompt
    assert '"Standard League"' in std_prompt and "rec=" not in std_prompt  # no invented reception value
    assert "0.5" not in std_prompt


def test_recommend_full_ppr_league_shows_its_own_values_including_keys_half_ppr_lacks(tmp_path, monkeypatch):
    raw_dir, persist_dir, _ = _setup(tmp_path, monkeypatch, OTHER_ID, "Full PPR League", FULL_PPR)
    client = _submit_only_client()

    recommend.recommend("q", OTHER_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, client=client)

    prompt = client.system_prompts[0]
    assert "rec=1.0" in prompt and "bonus_rec_te=0.5" in prompt
    assert "rec=0.5" not in prompt


def test_recommend_requires_league_id_positionally_or_by_keyword(tmp_path, monkeypatch):
    raw_dir, persist_dir, _ = _setup(tmp_path, monkeypatch)
    with pytest.raises(TypeError):
        recommend.recommend("q", raw_dir=raw_dir, persist_dir=persist_dir, client=_submit_only_client())  # type: ignore[call-arg]


def test_recommend_context_records_the_verified_league(tmp_path, monkeypatch):
    raw_dir, persist_dir, _ = _setup(tmp_path, monkeypatch)
    seen = {}
    real_dispatch = recommend.dispatch_tool

    def spy(name, tool_input, ctx):
        seen["league"] = ctx.league
        return real_dispatch(name, tool_input, ctx)

    monkeypatch.setattr(recommend, "dispatch_tool", spy)
    client = _RecordingClient(
        [
            SimpleNamespace(content=[_tool_use_block("get_my_roster", {})]),
            SimpleNamespace(
                content=[_tool_use_block("submit_recommendation", {"recommendation": "ok", "reasoning": "ok"}, "t2")]
            ),
        ]
    )
    recommend.recommend("q", VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, client=client)

    assert seen["league"].league_id == VS30_ID
    assert seen["league"].scoring_settings == HALF_PPR


# ---- generate_report(): same explicit, verified league ----


def test_generate_report_refuses_wrong_league(tmp_path, monkeypatch):
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch)
    with pytest.raises(LeagueMismatchError):
        report.generate_report(
            "drop", OTHER_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
        )


@pytest.mark.parametrize("report_type", ["start_sit", "drop", "waiver_pickups"])
def test_generate_report_header_names_the_league(tmp_path, monkeypatch, report_type):
    raw_dir, persist_dir, signals_dir = _setup(tmp_path, monkeypatch)
    result = report.generate_report(
        report_type, VS30_ID, raw_dir=raw_dir, persist_dir=persist_dir, season=_SEASON, as_of_week=_WEEK, signals_dir=signals_dir
    )
    assert result["league_id"] == VS30_ID
    assert result["league_name"] == "Victorious Secret 3.0"


@pytest.mark.parametrize("report_type", ["start_sit", "drop", "waiver_pickups"])
def test_generate_report_ranking_is_scoring_format_independent_today(tmp_path, monkeypatch, report_type):
    """Documents a real property of the current code rather than a goal:
    report.py's opportunity score never consumed scoring settings (there
    was nothing half-PPR-specific in it to parameterize), so two leagues
    with identical rosters and different scoring produce identical
    entries. If a scoring-aware ranking is ever added (Phase 6's
    points-based proxy is the natural place), this test should start
    failing and be rewritten -- that's the point of pinning it."""
    half_raw, half_chroma, half_sig = _setup(tmp_path, monkeypatch, VS30_ID, "Victorious Secret 3.0", HALF_PPR, subdir="half")
    full_raw, full_chroma, full_sig = _setup(tmp_path, monkeypatch, OTHER_ID, "Full PPR League", FULL_PPR, subdir="full")

    half = report.generate_report(
        report_type, VS30_ID, raw_dir=half_raw, persist_dir=half_chroma, season=_SEASON, as_of_week=_WEEK, signals_dir=half_sig
    )
    full = report.generate_report(
        report_type, OTHER_ID, raw_dir=full_raw, persist_dir=full_chroma, season=_SEASON, as_of_week=_WEEK, signals_dir=full_sig
    )

    assert half["league_id"] != full["league_id"]  # genuinely two leagues...
    assert half["entries"] == full["entries"]  # ...same ranking, because nothing in it is scoring-dependent
    assert half["entries"], "fixture should produce at least one entry so the equality is meaningful"


# ---- evals/build_ground_truth.py reads scoring the same way ----


def test_ground_truth_scoring_matches_what_recommend_is_shown(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_league(raw_dir, VS30_ID, "Victorious Secret 3.0", HALF_PPR)

    via_path = bgt.load_scoring_settings(raw_dir / "league.json")
    via_league_id = bgt.load_scoring_settings(raw_dir / "league.json", league_id=VS30_ID)
    via_loader = load_league(VS30_ID, raw_dir=raw_dir).scoring_settings

    assert via_path == via_league_id == via_loader == HALF_PPR


def test_ground_truth_refuses_to_score_against_another_leagues_settings(tmp_path):
    raw_dir = tmp_path / "sleeper"
    _seed_league(raw_dir, OTHER_ID, "Full PPR League", FULL_PPR)
    with pytest.raises(LeagueMismatchError):
        bgt.load_scoring_settings(raw_dir / "league.json", league_id=VS30_ID)


def test_ground_truth_points_change_with_the_leagues_real_scoring_not_a_constant():
    """The numeric application of scoring settings lives only here; it
    has always been driven by the dict it's given, never a fixed 0.5."""
    row = {"receptions": 4, "receiving_yards": 50}
    assert bgt.compute_points(row, HALF_PPR) == pytest.approx(4 * 0.5 + 50 * 0.1)
    assert bgt.compute_points(row, FULL_PPR) == pytest.approx(4 * 1.0 + 50 * 0.1)
    assert bgt.compute_points(row, STANDARD) == pytest.approx(50 * 0.1)


# ---- architectural guard: the per-league join stays out of signals/rag ----


def test_league_module_is_not_imported_by_signals_or_rag():
    """CLAUDE.md's key principle: the signals table and RAG corpus stay
    league-agnostic. Phase 5.1 added the per-league join in
    src/reasoning/league.py; nothing under src/signals/ or src/rag/ may
    import anything from src/reasoning/ or read scoring_settings at all
    (prose mentions of "scoring settings" in docstrings, explaining that
    they're deliberately NOT used there, are fine -- the literal
    scoring_settings key is not)."""
    root = Path(league_module.__file__).resolve().parents[2]
    offenders = []
    for path in sorted(list((root / "src" / "signals").rglob("*.py")) + list((root / "src" / "rag").rglob("*.py"))):
        text = path.read_text()
        if "src.reasoning" in text or "scoring_settings" in text:
            offenders.append(str(path.relative_to(root)))
    assert offenders == []
