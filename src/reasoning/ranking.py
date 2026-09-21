"""The one deterministic ranking used to compare players -- shared by the
Feed's reports (src/reasoning/report.py) and Chat's `rank_players` tool
(src/reasoning/recommend.py), so a start/sit verdict is computed by
exactly one piece of code no matter which surface asks.

Why this module exists (Chat vs. Feed consistency). Real usage testing
found Chat (`recommend()`) and the Feed's start_sit report
(`generate_report()`) reaching different verdicts on the same real
decision over the same signals: asked "should I start Justin Herbert or
Patrick Mahomes," Chat's own reasoning said it was "defaulting to Mahomes
based on his consistently elite performance history" once the
current-season signal went stale -- exactly the ungrounded-claim pattern
Phase 3.7's addendum guards against for trade advice, but nothing had
guarded start/sit comparisons. The two surfaces were two independent
reasoning paths: report.py ranked by a deterministic opportunity score,
recommend() let the model weigh get_player_signals' prose freely (and
fill in from general knowledge when that prose was stale).

The product decision (recorded in TODO.md): Chat's *verdict* on a
head-to-head / start-sit comparison must come from the same deterministic
ranking the Feed shows, with Chat free to add signals-backed explanation
on top. Mechanically that means the scoring/ranking logic has to be
callable for an arbitrary set of players, not just a whole roster --
and it has to live somewhere both callers can import. It can't stay in
report.py: report.py imports recommend.py (to reuse its tools), so
recommend.py importing report.py back would be a circular import. Hence
this module: it depends only on polars and the signals table location,
never on recommend.py or report.py.

Everything here was moved verbatim from report.py (the weights,
thresholds, table loading, prior-season fallback, score, formatting);
report.py's output is unchanged and pinned by tests/test_report.py.
The only genuinely new logic is `rank_candidates()`, which is the
"sort by score, take the top" step that used to be inlined in
`_start_sit_report` -- now also reporting whether the result is a clear
verdict, a tie, or ungroundable, so a caller that must speak honestly
about ambiguity (Chat) can, while the report keeps its existing
behavior.

**Ties.** The score is a float composite of real signal values, so an
exact tie between two different players is rare in practice, but it's
not impossible (two players with identical stale fallback rows, say),
and the honest answer to "which one?" in that case is "the ranking can't
separate them." `rank_candidates` reports `verdict: "tied"` when the top
two scores are within TIE_TOLERANCE. The start_sit report deliberately
keeps its pre-existing behavior on a tie (stable sort -> the first
candidate in roster order is listed as the starter) so this change
doesn't alter generate_report()'s output; Chat is told to say "tied"
instead. That one edge is documented in TODO.md rather than papered
over.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from src.signals.matchup_signals import PROCESSED_DIR as SIGNALS_DIR

# Re-exported on purpose: report.py and recommend.py both take "where the
# signals tables live" from here, so the two surfaces share one default.
__all__ = [
    "SIGNALS_DIR",
    "SignalTables",
    "load_signals_table",
    "load_prior_season_fallback_table",
    "signal_row",
    "target_share",
    "opportunity_score",
    "stale_fields",
    "fmt_signal_row",
    "weakness_reasons",
    "rank_candidates",
]

# Deliberately simple, unfitted weights -- same spirit as
# matchup_signals.py's own documented 0.1 reweighting constant: a
# reasonable first pass, not a fitted model. Revisit once real usage or
# decision-accuracy evals suggest better weights.
# Fitted weights (2026-09-21, evals/fit_ranking_weights.py --blend 4): a
# no-intercept pairwise logistic model on the feature DIFFERENCES of
# 19,045 same-position pairs from every 2023 regular-season week (both
# players >= 5 pts that week, features strictly as-of the week), tested on
# 19,283 pairs from 2024. Held-out accuracy 61.8% (58.7% in weeks 2-5,
# ~72% where the actual gap was 10+ pts) against 57.6% for the hand-set
# weights this replaced -- and 61.5% for points-per-game ALONE, which is
# the honest headline: points carry nearly all of the signal; the usage
# terms mostly earn their place by making the verdict explainable. The
# points term is a shrunk estimate, (games x ppg + 4 x last season's ppg)
# / (games + 4), so one big week in September does not read like a
# season's average (the unshrunk fit put a 1-game 33.8 over a proven WR).
# The score is w . features, so a 10-point ppg edge (0.81) outweighs a
# 10-percentage-point target-share edge (0.03) or red-zone edge (0.06).
# evals/results/2026-09-20_ranking_fit_blend4.json has the run (and
# ..._ranking_fit.json the unshrunk one).
PPG_WEIGHT = 0.0812               # shrunk points per game under THIS league's scoring -- see blended_ppg()
PPG_BLEND_GAMES = 4.0             # last season counts as this many games' worth of evidence
TARGET_SHARE_WEIGHT = 0.2942
RED_ZONE_SHARE_WEIGHT = 0.6056
EPA_TREND_WEIGHT = -0.0736        # slightly negative on held-out data: a hot trailing window does not predict next week
IMPLIED_TOTAL_WEIGHT = 0.0146     # everyone, not just passers
CPOE_WEIGHT = 0.0198              # passers only (the only rows that carry it)
RUN_FUNNEL_WEIGHT = -0.4007       # opponent defense lean; negative = facing a run-funnel defense predicts fewer points
MIN_TREND_PLAYS = 20

# Thresholds below which a signal counts as a concrete "why this player is
# weak" reason in the drop report. Same status as the weights above --
# simple, documented, not fitted.
LOW_RED_ZONE_SHARE = 0.08
LOW_TARGET_SHARE = 0.10
LOW_SEASON_PLAYS = 10

# Two opportunity scores closer than this are reported as a tie rather
# than an ordering -- float-comparison slack, not a "close enough to call
# it even" product threshold (that would be a fitted judgment this
# project hasn't made; see module docstring).
TIE_TOLERANCE = 1e-9

# The plain-language description of what the score is, returned alongside
# every ranking so a caller explaining a verdict says what it actually
# rests on (and nothing it doesn't).
SCORE_DESCRIPTION = (
    "fitted linear score (evals/fit_ranking_weights.py, trained on 2023, tested on 2024: 62% pairwise, 72% when "
    f"the outcome gap was 10+ pts) = {PPG_WEIGHT:g} x points per game under THIS league's scoring -- this season's "
    f"games blended with last season's average as if it were {PPG_BLEND_GAMES:g} games, so a single big week is "
    f"discounted -- + {TARGET_SHARE_WEIGHT:g} x target share (opponent-adjusted when available) "
    f"+ {RED_ZONE_SHARE_WEIGHT:g} x red zone role share + {IMPLIED_TOTAL_WEIGHT:g} x team implied total "
    f"+ {EPA_TREND_WEIGHT:g} x recent EPA/play trend (only from {MIN_TREND_PLAYS} plays on record) "
    f"+ {CPOE_WEIGHT:g} x CPOE (passers) + {RUN_FUNNEL_WEIGHT:g} x opponent run-funnel lean; higher is better. "
    "Points per game alone tested almost as well as the whole score, so treat the other terms as the explanation, "
    "not the edge. A signal that isn't computed for a player contributes nothing; a player with none of them "
    "can't be ranked. Not position-normalized: a QB's score is not comparable to a skill player's."
)


def blended_ppg(row: dict) -> float | None:
    """This season's points per game shrunk toward last season's:
    (games x ppg + PPG_BLEND_GAMES x prior ppg) / (games + PPG_BLEND_GAMES).
    With no prior season the raw ppg; with no games yet, the prior alone;
    None when there is neither."""
    ppg, games, prior = row.get("ppg"), row.get("games_played") or 0, row.get("ppg_prior_season")
    if ppg is None and prior is None:
        return None
    if prior is None:
        return float(ppg)
    if ppg is None:
        return float(prior)
    return (games * float(ppg) + PPG_BLEND_GAMES * float(prior)) / (games + PPG_BLEND_GAMES)


def trend_is_trustworthy(row: dict) -> bool:
    """Whether row's epa_trend rests on enough plays to count (see
    MIN_TREND_PLAYS). A row that doesn't carry season_plays at all is
    taken at face value."""
    plays = row.get("season_plays")
    return plays is None or plays >= MIN_TREND_PLAYS


def load_signals_table(signals_dir: Path, season: int, as_of_week: int) -> dict[str, dict]:
    """The raw numeric signals row for every player with CURRENT-season
    data as of as_of_week, keyed by player_id (nflverse gsis_id) -- the
    same underlying table src/rag/embed.py's build_signal_chunks() turns
    into the text get_player_signals returns, read directly here because
    ranking needs the numbers, not the prose (see report.py's module
    docstring).

    Unions every locally-computed `signals_{season}_week*.parquet` file
    with week <= as_of_week (never a later week -- that would leak future
    data into an as-of-week-filtered report, violating CLAUDE.md's
    as-of-date-filtering rule), keeping each player's highest available
    week. This is deliberately more lenient than "only the exact
    as_of_week file": it's what decides whether a player counts as having
    ANY current-season signal at all for the prior-season fallback
    threshold (see report.py's "Prior-season signal fallback" section) --
    a player missing from this week's file specifically but present in an
    earlier one still has real current-season data, and shouldn't be
    treated the same as a player with nothing this season."""
    rows: dict[str, dict] = {}
    best_week: dict[str, int] = {}
    for path in sorted(signals_dir.glob(f"signals_{season}_week*.parquet")):
        week = int(path.stem.rsplit("week", 1)[-1])
        if week > as_of_week:
            continue
        for row in pl.read_parquet(path).to_dicts():
            player_id = row["player_id"]
            if player_id not in best_week or week > best_week[player_id]:
                best_week[player_id] = week
                rows[player_id] = row
    return rows


def load_prior_season_fallback_table(signals_dir: Path, season: int) -> tuple[int | None, dict[str, dict]]:
    """The most recent locally-computed `season - 1` signals table (the
    highest as_of_week file found -- as close to "final, full season"
    numbers as what's actually on disk), used only as a stale fallback
    when `season` has no current-season data for a player at all. Returns
    (prior_season, rows_by_player_id); (None, {}) if no prior-season file
    exists locally."""
    prior_season = season - 1
    candidates = sorted(signals_dir.glob(f"signals_{prior_season}_week*.parquet"))
    if not candidates:
        return None, {}

    def _week_num(path: Path) -> int:
        return int(path.stem.rsplit("week", 1)[-1])

    latest = max(candidates, key=_week_num)
    rows = {row["player_id"]: row for row in pl.read_parquet(latest).to_dicts()}
    return prior_season, rows


@dataclass
class SignalTables:
    """The two tables a ranking reads -- current-season rows as of a week,
    and the prior-season fallback -- loaded once per report / per chat
    turn and reused for every candidate (one parquet read, not one per
    player)."""

    season: int
    as_of_week: int
    signals_by_id: dict[str, dict]
    fallback_season: int | None
    fallback_rows: dict[str, dict]
    # Phase 6: the per-league join. Points per game this season (weeks
    # before as_of_week) and over all of last season, computed from the
    # league-agnostic weekly stat lines under THIS league's scoring --
    # empty when no scoring settings or no stat tables were given.
    proxy: dict[str, dict] = field(default_factory=dict)
    proxy_prior: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        signals_dir: Path,
        season: int,
        as_of_week: int,
        scoring_settings: dict | None = None,
        stats_dir: Path | None = None,
    ) -> "SignalTables":
        signals_by_id = load_signals_table(signals_dir, season, as_of_week)
        fallback_season, fallback_rows = load_prior_season_fallback_table(signals_dir, season)
        proxy: dict[str, dict] = {}
        proxy_prior: dict[str, dict] = {}
        # Default: next to the signals tables (data/processed/player_stats in
        # production). Deriving it from signals_dir rather than a module
        # constant is what keeps a test's temp signals dir from reading the
        # real stat tables on the developer's disk.
        stats_dir = stats_dir or (Path(signals_dir).parent / "player_stats")
        if scoring_settings:
            from src.reasoning import points_proxy
            from src.signals import player_stats as ps  # local: keeps ranking importable without nflreadpy loaded eagerly

            proxy = points_proxy.season_points_proxy(ps.load_weekly_stats(season, stats_dir), scoring_settings, as_of_week)
            proxy_prior = points_proxy.season_points_proxy(ps.load_weekly_stats(season - 1, stats_dir), scoring_settings)
        return cls(season, as_of_week, signals_by_id, fallback_season, fallback_rows, proxy, proxy_prior)

    def proxy_fields(self, player_id: str) -> dict:
        """The Phase 6 points proxy for one player, as the fields every
        row/tool output carries: ppg / games_played /
        season_points_so_far_proxy (this season, as-of) and
        ppg_prior_season / prior_games_played (all of last season)."""
        now = self.proxy.get(player_id) or {}
        prior = self.proxy_prior.get(player_id) or {}
        return {
            "ppg": now.get("ppg"),
            "games_played": now.get("games_played"),
            "season_points_so_far_proxy": now.get("season_points_so_far_proxy"),
            "ppg_prior_season": prior.get("ppg"),
            "prior_games_played": prior.get("games_played"),
        }

    def row_for(self, player_id: str) -> dict | None:
        """The player's signal row (current-season, else stale prior-season
        fallback) with the points proxy joined on. A player with points on
        record but no usage row at all (e.g. a QB who only threw, before
        the passing signals exist for the week) gets a proxy-only row --
        rankable on points, labeled as carrying no usage signals."""
        row = signal_row(player_id, self.signals_by_id, self.fallback_season, self.fallback_rows)
        proxy = self.proxy_fields(player_id)
        if row is None:
            if proxy["ppg"] is None and proxy["ppg_prior_season"] is None:
                return None
            row = {"player_id": player_id, "proxy_only": True, "stale": False}
        return {**row, **proxy}


def signal_row(
    player_id: str,
    signals_by_id: dict[str, dict],
    fallback_season: int | None,
    fallback_rows: dict[str, dict],
) -> dict | None:
    """A player's current-season row if they have one at all (see
    load_signals_table's docstring for what counts); otherwise their
    most recent PRIOR season's row, explicitly marked stale so it's never
    mistaken for current data downstream (fmt_signal_row/weakness_reasons
    both check row["stale"]). None if neither exists."""
    row = signals_by_id.get(player_id)
    if row is not None:
        return row
    fallback_row = fallback_rows.get(player_id)
    if fallback_row is None:
        return None
    return {
        **fallback_row,
        # The matchup-specific fields describe LAST season's final week's
        # game, not this week's -- they must not score or print as if
        # they did (they count for everyone since the 2026-09-21 refit).
        "opponent": None,
        "implied_total": None,
        "run_funnel_rate_vs_avg": None,
        "stale": True,
        "source_season": fallback_season,
        "source_as_of_week": fallback_row.get("as_of_week"),
    }


def target_share(row: dict) -> float | None:
    adjusted = row.get("target_share_adjusted")
    return adjusted if adjusted is not None else row.get("target_share")


def opportunity_score(row: dict | None) -> float | None:
    """The fitted linear score (see the weights above). Higher = the better
    start/pickup; used inverted for the drop report's "weakest". None
    when a player has no scored signal at all -- such a player can't be
    ranked against anything, and callers must exclude them rather than
    silently treating missing data as zero."""
    if row is None:
        return None
    terms = [
        (PPG_WEIGHT, blended_ppg(row)),
        (TARGET_SHARE_WEIGHT, target_share(row)),
        (RED_ZONE_SHARE_WEIGHT, row.get("red_zone_share")),
        (IMPLIED_TOTAL_WEIGHT, row.get("implied_total")),
        (EPA_TREND_WEIGHT, row.get("epa_trend") if (row.get("epa_trend") is not None and trend_is_trustworthy(row)) else None),
        (CPOE_WEIGHT, row.get("cpoe")),
        (RUN_FUNNEL_WEIGHT, row.get("run_funnel_rate_vs_avg")),
    ]
    present = [(w, v) for w, v in terms if v is not None]
    if not present:
        return None
    return sum(w * float(v) for w, v in present)


def stale_fields(row: dict | None) -> dict:
    """The explicit staleness marker every output entry carries alongside
    its prose -- never rely on a reader noticing a season number buried
    in signals_summary text (see report.py's "Prior-season signal
    fallback" section)."""
    if row is None or not row.get("stale"):
        return {"stale": False, "source_season": None, "source_as_of_week": None}
    return {
        "stale": True,
        "source_season": row.get("source_season"),
        "source_as_of_week": row.get("source_as_of_week"),
    }


# Early-season: every play so far sits inside the trailing window, so there is
# no baseline for a trend to differ from (matchup_signals.recent_efficiency_trend
# nulls the trend and reports epa_baseline_plays == 0). Said out loud rather
# than omitted, so a reader never infers "no trend mentioned" == "steady".
NO_TREND_YET = "no efficiency trend yet (too early in the season for a trailing-window comparison)"


def _trend_direction(epa_trend: float) -> str:
    return "up" if epa_trend > 0 else "down" if epa_trend < 0 else "flat"


def fmt_signal_row(row: dict | None) -> str:
    """Human-readable citation of the specific numbers a ranking/reason
    was grounded in -- every number here traces back to a real computed
    signal, never a generic-sounding filler sentence."""
    if row is None:
        return "no computed signals available for this player/week"
    prefix = ""
    if row.get("stale"):
        # Say WHY it's stale: a row is missing from the current-season
        # table exactly when nflverse has no play with this player as
        # rusher/receiver/passer before the as-of week -- so either the
        # season hasn't started, or (the confusing case for a user: a
        # rostered RB showing last year's numbers in week 2) they haven't
        # touched the ball yet this season.
        prefix = (
            f"[STALE -- no plays recorded for this player this season before the as-of week, "
            f"showing {row.get('source_season')} season-end reference instead] "
        )
    parts = []
    if row.get("ppg") is not None:
        parts.append(f"{row['ppg']:.1f} pts/game over {row['games_played']} game(s) this season in this league's scoring")
    if row.get("ppg_prior_season") is not None:
        parts.append(f"{row['ppg_prior_season']:.1f} pts/game last season")
    if row.get("proxy_only"):
        parts.append("no usage/matchup signals computed yet")
    if row.get("epa_trend") is not None and not trend_is_trustworthy(row):
        parts.append(
            f"efficiency trend {row['epa_trend']:+.2f} EPA/play on only {row['season_plays']} plays (too few to count)"
        )
    elif row.get("epa_trend") is not None:
        parts.append(f"efficiency trending {_trend_direction(row['epa_trend'])} ({row['epa_trend']:+.2f} EPA/play)")
    elif row.get("epa_baseline_plays") == 0:
        parts.append(NO_TREND_YET)
    if row.get("red_zone_share") is not None:
        parts.append(f"red zone role share {row['red_zone_share'] * 100:.0f}%")
    share = target_share(row)
    if share is not None:
        parts.append(f"target share {share * 100:.0f}%")
    if row.get("cpoe") is not None:
        parts.append(f"completion % over expected {row['cpoe']:+.1f}")
    if row.get("opponent"):
        parts.append(f"facing {row['opponent']}")
    if row.get("implied_total") is not None:
        parts.append(f"team implied total {row['implied_total']:.1f} pts")
    if row.get("run_funnel_rate_vs_avg") is not None:
        lean = "run-funnel" if row["run_funnel_rate_vs_avg"] > 0 else "pass-funnel"
        parts.append(f"opponent defense skews {lean} ({row['run_funnel_rate_vs_avg'] * 100:+.0f}pts vs. avg)")
    return prefix + ("; ".join(parts) if parts else "no individual signal values computed")


def weakness_reasons(row: dict) -> list[str]:
    """Concrete, threshold-based reasons a player looks weak -- never
    just "lowest score," per the requirement to explain *why*. When row
    is a stale prior-season fallback, that's named explicitly as the
    first reason so every downstream reason reads in context (e.g.
    "efficiency trending down" here means down relative to last season,
    not this one)."""
    reasons = []
    if row.get("stale"):
        reasons.append(
            f"based on {row.get('source_season')} season-end data (stale -- no current-season "
            "signal computed yet); treat these as a reference point, not this season's performance"
        )
    if row.get("epa_trend") is not None and row["epa_trend"] < 0:
        reasons.append(f"efficiency trending down ({row['epa_trend']:+.2f} EPA/play, trailing window)")
    if row.get("red_zone_share") is not None and row["red_zone_share"] < LOW_RED_ZONE_SHARE:
        reasons.append(f"minimal red zone role ({row['red_zone_share'] * 100:.0f}% share)")
    share = target_share(row)
    if share is not None and share < LOW_TARGET_SHARE:
        reasons.append(f"low target share ({share * 100:.0f}%)")
    if row.get("season_plays") is not None and row["season_plays"] < LOW_SEASON_PLAYS:
        reasons.append(f"minimal season-long usage ({row['season_plays']} plays involved so far)")
    if not reasons:
        reasons.append(
            "lowest composite opportunity score among the roster's signal-bearing players, "
            "though no single signal here is individually alarming"
        )
    return reasons


def rank_candidates(candidates: list[dict]) -> dict:
    """Rank a set of candidates by opportunity score -- THE start/sit
    verdict, shared by the Feed's start_sit report and Chat's
    rank_players tool.

    Each candidate is {"player_id", "name", "position", "team", "row"}
    where "row" is the player's signal row (current-season or stale
    fallback, from SignalTables.row_for / signal_row) or None.

    Returns:
      verdict: "clear" -- 2+ rankable candidates and a single top score;
               "tied" -- 2+ rankable candidates but the top two scores are
               indistinguishable (within TIE_TOLERANCE);
               "insufficient_data" -- fewer than 2 candidates had any
               usable signal, so there is nothing to ground a comparison
               in (the same condition under which the start_sit report
               skips a position).
      recommended: the top-ranked entry when verdict is "clear", else None
               -- a caller must never pick one on a tie or without data.
      ranked:  every rankable candidate, best first (stable: on equal
               scores, input order is preserved, exactly as the start_sit
               report has always ordered them), each carrying its
               opportunity_score, a signals_summary citing the actual
               numbers, and the explicit stale markers.
      tied_at_top: on a tie, every entry sharing the top score; else [].
      unranked: candidates with no usable signal at all (row None, or a
               row with none of the three scored signals).
      score_description: what the score is, in words.
    """
    scored = []
    unranked = []
    for candidate in candidates:
        row = candidate.get("row")
        score = opportunity_score(row)
        entry = {
            "player_id": candidate.get("player_id"),
            "name": candidate.get("name"),
            "position": candidate.get("position"),
            "team": candidate.get("team"),
        }
        if score is None:
            unranked.append(entry)
            continue
        scored.append(
            {
                **entry,
                "opportunity_score": round(score, 4),
                "signals_summary": fmt_signal_row(row),
                **stale_fields(row),
                "_score": score,
            }
        )
    scored.sort(key=lambda entry: entry["_score"], reverse=True)

    if len(scored) < 2:
        verdict = "insufficient_data"
        tied_at_top: list[dict] = []
    elif abs(scored[0]["_score"] - scored[1]["_score"]) <= TIE_TOLERANCE:
        verdict = "tied"
        top = scored[0]["_score"]
        tied_at_top = [entry for entry in scored if abs(entry["_score"] - top) <= TIE_TOLERANCE]
    else:
        verdict = "clear"
        tied_at_top = []

    ranked = [{k: v for k, v in entry.items() if k != "_score"} for entry in scored]
    tied_at_top = [{k: v for k, v in entry.items() if k != "_score"} for entry in tied_at_top]
    return {
        "verdict": verdict,
        "recommended": ranked[0] if verdict == "clear" else None,
        "ranked": ranked,
        "tied_at_top": tied_at_top,
        "unranked": unranked,
        "score_description": SCORE_DESCRIPTION,
    }
