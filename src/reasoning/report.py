"""Phase 3.5 report generation: generate_report() produces a structured,
whole-roster (or whole-player-pool) report -- a genuinely different
orchestration pattern from recommend.py's single question -> single
answer loop. See PROJECT_SPEC.md's "Phase 3.5: Multi-turn conversation +
report generation" section for the full spec and sequencing rationale.

Three report types, in the order the spec sequences them (buildable now,
by data readiness):
  - start_sit: for every roster position with 2+ signal-bearing players,
    recommend a starter over the alternatives.
  - drop: the weakest current roster contributors, with the specific
    signal(s) that make each one weak.
  - waiver_pickups: rank the unrostered NFL player pool (every league
    roster subtracted from nflreadpy's full player list -- the same
    source src/rag/player_index.py already uses) by opportunity signals.

Trade suggestions are deliberately NOT built here -- PROJECT_SPEC.md is
explicit that this needs signal work that doesn't exist yet (season-long/
rest-of-season player value, cross-roster positional need); a trade
report grounded in this-week's-matchup-scoped signals standing in for
season-long value would be actively misleading. Scoped as a named
follow-up, not attempted.

**Design choice: no Claude API call in this module.** recommend.py's
single-question path uses a Claude tool-use loop because the space of
possible questions is open-ended. A report's job is different -- rank a
known, bounded set of candidates by already-computed numeric signals --
and that ranking has one honest, auditable answer: whichever candidate's
numbers are better. Doing that ranking and citing the specific numbers
directly in code (rather than asking a model to eyeball the numbers and
hope it reasons correctly) is both more reliable and, concretely, the
only way to validate this session's real-2024-data requirement in this
sandbox: no ANTHROPIC_API_KEY is configured here (same blocker as every
prior phase -- see TODO.md), so a Claude-dependent report could not have
been end-to-end validated at all. This does reuse recommend.py's tools
and src/rag/player_index.py's identity resolution -- see below.

**Where this does and doesn't reuse recommend.py's tools:**
- get_my_roster and get_player_signals are called through
  recommend.dispatch_tool() exactly as recommend()'s agent loop would --
  same roster listing, same name resolution
  (src/rag/player_index.resolve_player, exact-then-fuzzy against the
  real player list, never a guess), same human-readable signal text.
  That's genuine reuse for identity + narrative grounding.
- get_team_record / get_current_matchup are also called through
  dispatch_tool() to give the start/sit and drop reports real week
  context (record, this week's opponent) instead of generating a report
  in a vacuum.
- search_league_info (semantic search) is NOT used here -- ranking
  candidates by number is exactly the class of problem structured
  lookup exists for over semantic search (same principle
  src/rag/player_index.py's docstring explains for player identity);
  there's no ranking question here semantic search is better suited to
  answer.
- What ISN'T reused: get_player_signals's chunk text has no numeric
  fields (src/rag/embed.py only stores type/player_id/week/season as
  Chroma metadata -- the actual numbers live only inside the generated
  sentence). Ranking and drop-reason thresholds need the numbers
  themselves, so this module reads the same signals parquet
  (src/signals/matchup_signals.py's build_signals_table() output) that
  those chunks were generated from, directly via polars, rather than
  regex-parsing get_player_signals's prose or duplicating
  matchup_signals.py's computation. For waiver_pickups specifically, the
  unrostered pool can run into the hundreds of candidates -- issuing one
  Chroma lookup per candidate would be slow and redundant with the one
  parquet read that already has every candidate's numbers.

**Known simplification, documented rather than silently assumed:**
start_sit groups a roster by Sleeper's own `position` field (QB/RB/WR/TE
-- the same skill positions src/rag/player_index.py's SKILL_POSITIONS
covers, since those are the only positions with computed matchup
signals). It does not model a league's actual Sleeper `roster_positions`
slot structure (FLEX/superflex/bench counts, IDP, etc.) -- a true
slot-by-slot lineup optimizer is a materially bigger scope than "for
every roster slot with more than one viable option, recommend who to
start" needs to demonstrate. Revisit if FLEX-aware recommendations turn
out to matter once this is used against a real roster.

**Known simplification for "rising" opportunity signals:** the signals
table is one point-in-time row per player per as_of_week (season-to-date
aggregates, not a rolling week-over-week series), so "rising target
share" in waiver_pickups is approximated by the current target_share
(adjusted for opponent pass defense when available) rather than an
actual week-over-week delta -- that delta isn't a signal this project
computes yet. Documented here rather than overclaiming a trend that
isn't actually measured.

**Prior-season signal fallback (Phase 3.6).** Every signal in this
project is trailing/current-season by construction (EPA trend, red zone
share, target share, ...) -- confirmed live: with the 2026 season not
yet started, nflreadpy has no 2026 plays published yet, so every report
run against the real current week came back with empty `entries` and a
"no computed signals" note. Correct, honest behavior for a genuinely
empty signals table, but a bad first-use experience if a friend opens
this in Week 1 and gets nothing. Fix: when a player has NO current-season
signal at all (see `_load_signals_table`'s docstring for exactly what
"current-season" means here), fall back to their most recent PRIOR
season's final numbers instead of nothing -- but every such number is
explicitly marked stale (`"stale": True`, `"source_season"`,
`"source_as_of_week"` on the row, the entry, AND a `[STALE -- ...]`
prefix on `signals_summary`/`reasoning` text) so a report never presents
last year's numbers as if they were this year's. The same fallback (and
the same never-silent labeling) lives in
`src/rag/retrieve.py:query_player_signal_with_fallback()` for
`recommend.py`'s `get_player_signals` chat tool -- see that function's
docstring; this file's version reads the raw parquet table directly for
the same reason the rest of this module does (see "Where this does and
doesn't reuse recommend.py's tools" above), so the two are separate,
parallel implementations against two different data sources, not one
shared function.

**Fallback threshold: N=1** -- a player with ANY current-season signal
row at all (even from an earlier week than as_of_week) is used as-is,
never blended with a stale prior-season number. Only a player with ZERO
current-season data falls back. A larger N (smoothing out one noisy
early-season game by requiring 2+ weeks before trusting current data) is
NOT implemented: that needs a real "distinct weeks active this season"
field, which doesn't exist in `matchup_signals.py`'s output today --
adding one is signals-*computation* work, out of scope for this unit,
which is deliberately confined to the signal-*loading* layer (this
file's `_load_signals_table`/`_load_prior_season_fallback_table` and
`retrieve.py`'s `query_player_signal_with_fallback`). Revisit N if
`matchup_signals.py` ever gains that field.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from src.rag import lookup, player_index
from src.rag.embed import CHROMA_DIR, RAW_DIR
from src.reasoning import recommend
from src.signals.matchup_signals import PROCESSED_DIR as SIGNALS_DIR

REPORT_TYPES = ("start_sit", "drop", "waiver_pickups")

# Deliberately simple, unfitted weights -- same spirit as
# matchup_signals.py's own documented 0.1 reweighting constant: a
# reasonable first pass, not a fitted model. Revisit once real usage or
# decision-accuracy evals suggest better weights.
_EPA_TREND_WEIGHT = 2.0
_RED_ZONE_SHARE_WEIGHT = 3.0
_TARGET_SHARE_WEIGHT = 2.0

# Thresholds below which a signal counts as a concrete "why this player is
# weak" reason in the drop report. Same status as the weights above --
# simple, documented, not fitted.
_LOW_RED_ZONE_SHARE = 0.08
_LOW_TARGET_SHARE = 0.10
_LOW_SEASON_PLAYS = 10


def _load_signals_table(signals_dir: Path, season: int, as_of_week: int) -> dict[str, dict]:
    """The raw numeric signals row for every player with CURRENT-season
    data as of as_of_week, keyed by player_id (nflverse gsis_id) -- the
    same underlying table src/rag/embed.py's build_signal_chunks() turns
    into the text get_player_signals returns, read directly here because
    ranking needs the numbers, not the prose (see module docstring).

    Unions every locally-computed `signals_{season}_week*.parquet` file
    with week <= as_of_week (never a later week -- that would leak future
    data into an as-of-week-filtered report, violating CLAUDE.md's
    as-of-date-filtering rule), keeping each player's highest available
    week. This is deliberately more lenient than "only the exact
    as_of_week file": it's what decides whether a player counts as having
    ANY current-season signal at all for the prior-season fallback
    threshold (see module docstring's "Prior-season signal fallback"
    section) -- a player missing from this week's file specifically but
    present in an earlier one still has real current-season data, and
    shouldn't be treated the same as a player with nothing this season."""
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


def _load_prior_season_fallback_table(signals_dir: Path, season: int) -> tuple[int | None, dict[str, dict]]:
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


def _signal_row(
    player_id: str,
    signals_by_id: dict[str, dict],
    fallback_season: int | None,
    fallback_rows: dict[str, dict],
) -> dict | None:
    """A player's current-season row if they have one at all (see
    _load_signals_table's docstring for what counts); otherwise their
    most recent PRIOR season's row, explicitly marked stale so it's never
    mistaken for current data downstream (_fmt_signal_row/_weakness_reasons
    both check row["stale"]). None if neither exists."""
    row = signals_by_id.get(player_id)
    if row is not None:
        return row
    fallback_row = fallback_rows.get(player_id)
    if fallback_row is None:
        return None
    return {
        **fallback_row,
        "stale": True,
        "source_season": fallback_season,
        "source_as_of_week": fallback_row.get("as_of_week"),
    }


def _target_share(row: dict) -> float | None:
    adjusted = row.get("target_share_adjusted")
    return adjusted if adjusted is not None else row.get("target_share")


def _opportunity_score(row: dict | None) -> float | None:
    """Higher = more opportunity (better start_sit/waiver candidate);
    used inverted (ascending sort) for drop's "weakest" ranking. None
    when a player has no usable signal at all -- such a player can't be
    ranked against anything, and callers must exclude them rather than
    silently treating missing data as zero opportunity."""
    if row is None:
        return None
    score = 0.0
    has_any_signal = False
    if row.get("epa_trend") is not None:
        score += _EPA_TREND_WEIGHT * row["epa_trend"]
        has_any_signal = True
    if row.get("red_zone_share") is not None:
        score += _RED_ZONE_SHARE_WEIGHT * row["red_zone_share"]
        has_any_signal = True
    target_share = _target_share(row)
    if target_share is not None:
        score += _TARGET_SHARE_WEIGHT * target_share
        has_any_signal = True
    return score if has_any_signal else None


def _stale_fields(row: dict | None) -> dict:
    """The explicit staleness marker every output entry carries alongside
    its prose -- never rely on a reader noticing a season number buried
    in signals_summary text (see module docstring's "Prior-season signal
    fallback" section)."""
    if row is None or not row.get("stale"):
        return {"stale": False, "source_season": None, "source_as_of_week": None}
    return {
        "stale": True,
        "source_season": row.get("source_season"),
        "source_as_of_week": row.get("source_as_of_week"),
    }


def _stale_note(candidates: list[dict]) -> str | None:
    """A summary note when any candidate's row (candidates carrying a
    "row" key, e.g. from _resolve_roster_with_signals) is a stale
    prior-season fallback -- surfaced once per report in `notes`, on top
    of (never instead of) the per-entry stale markers, so the gap is
    visible even to a caller only skimming `notes`."""
    stale = [c for c in candidates if c.get("row") and c["row"].get("stale")]
    if not stale:
        return None
    source_season = stale[0]["row"].get("source_season")
    names = ", ".join(c["name"] for c in stale)
    return (
        f"{len(stale)} player(s) had no current-season signal yet and fell back to stale "
        f"{source_season} season-end data: {names}."
    )


def _fmt_signal_row(row: dict | None) -> str:
    """Human-readable citation of the specific numbers a ranking/reason
    was grounded in -- every number here traces back to a real computed
    signal, never a generic-sounding filler sentence."""
    if row is None:
        return "no computed signals available for this player/week"
    prefix = ""
    if row.get("stale"):
        prefix = (
            f"[STALE -- no current-season signal yet, showing {row.get('source_season')} "
            f"season-end reference instead] "
        )
    parts = []
    if row.get("epa_trend") is not None:
        direction = "up" if row["epa_trend"] > 0 else "down"
        parts.append(f"efficiency trending {direction} ({row['epa_trend']:+.2f} EPA/play)")
    if row.get("red_zone_share") is not None:
        parts.append(f"red zone role share {row['red_zone_share'] * 100:.0f}%")
    target_share = _target_share(row)
    if target_share is not None:
        parts.append(f"target share {target_share * 100:.0f}%")
    if row.get("opponent"):
        parts.append(f"facing {row['opponent']}")
    if row.get("implied_total") is not None:
        parts.append(f"team implied total {row['implied_total']:.1f} pts")
    if row.get("run_funnel_rate_vs_avg") is not None:
        lean = "run-funnel" if row["run_funnel_rate_vs_avg"] > 0 else "pass-funnel"
        parts.append(f"opponent defense skews {lean} ({row['run_funnel_rate_vs_avg'] * 100:+.0f}pts vs. avg)")
    return prefix + ("; ".join(parts) if parts else "no individual signal values computed")


def _weakness_reasons(row: dict) -> list[str]:
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
    if row.get("red_zone_share") is not None and row["red_zone_share"] < _LOW_RED_ZONE_SHARE:
        reasons.append(f"minimal red zone role ({row['red_zone_share'] * 100:.0f}% share)")
    target_share = _target_share(row)
    if target_share is not None and target_share < _LOW_TARGET_SHARE:
        reasons.append(f"low target share ({target_share * 100:.0f}%)")
    if row.get("season_plays") is not None and row["season_plays"] < _LOW_SEASON_PLAYS:
        reasons.append(f"minimal season-long usage ({row['season_plays']} plays involved so far)")
    if not reasons:
        reasons.append(
            "lowest composite opportunity score among the roster's signal-bearing players, "
            "though no single signal here is individually alarming"
        )
    return reasons


def _resolve_roster_with_signals(
    ctx: recommend.RecommendContext,
    signals_by_id: dict[str, dict],
    fallback_season: int | None,
    fallback_rows: dict[str, dict],
) -> tuple[list[dict], list[str]]:
    """get_my_roster + get_player_signals (via dispatch_tool -- the real
    tools, not a reimplementation) for every rostered player, joined with
    that player's raw numeric row (current-season, or a stale prior-season
    fallback -- see _signal_row) for ranking. Returns
    (resolved_candidates, unresolved_player_names) -- a player whose name
    can't be identity-resolved is reported, never silently dropped."""
    roster_result = recommend.dispatch_tool("get_my_roster", {}, ctx)
    if "error" in roster_result:
        raise RuntimeError(roster_result["error"])

    resolved: list[dict] = []
    unresolved: list[str] = []
    for player in roster_result["players"]:
        name = player.get("name")
        if not name:
            continue
        signal_result = recommend.dispatch_tool("get_player_signals", {"player_name": name}, ctx)
        if not signal_result.get("resolved"):
            unresolved.append(name)
            continue
        player_id = signal_result["player_id"]
        resolved.append(
            {
                "player_id": player_id,
                "name": signal_result["player_name"],
                "position": player.get("position") or signal_result.get("position"),
                "team": player.get("team") or signal_result.get("team"),
                "row": _signal_row(player_id, signals_by_id, fallback_season, fallback_rows),
            }
        )
    return resolved, unresolved


def _report_header(ctx: recommend.RecommendContext, report_type: str) -> dict:
    team = lookup.current_roster(ctx.raw_dir)
    record = recommend.dispatch_tool("get_team_record", {}, ctx)
    matchup = recommend.dispatch_tool("get_current_matchup", {}, ctx)
    return {
        "report_type": report_type,
        "season": ctx.season,
        "as_of_week": ctx.as_of_week,
        "roster_id": team.get("roster_id"),
        "team_name": team.get("team_name"),
        "record": record if "error" not in record else None,
        "current_matchup": matchup,
    }


def _start_sit_report(
    ctx: recommend.RecommendContext,
    signals_by_id: dict[str, dict],
    fallback_season: int | None,
    fallback_rows: dict[str, dict],
) -> dict:
    resolved, unresolved = _resolve_roster_with_signals(ctx, signals_by_id, fallback_season, fallback_rows)

    by_position: dict[str, list[dict]] = {}
    for candidate in resolved:
        by_position.setdefault(candidate["position"], []).append(candidate)

    entries = []
    notes = []
    if unresolved:
        notes.append(f"Could not identity-resolve {len(unresolved)} rostered player(s): {', '.join(unresolved)}.")
    stale_note = _stale_note(resolved)
    if stale_note:
        notes.append(stale_note)

    for position in sorted(by_position):
        candidates = by_position[position]
        scored = [(c, _opportunity_score(c["row"])) for c in candidates]
        grounded = [(c, score) for c, score in scored if score is not None]
        if len(candidates) < 2:
            continue
        if len(grounded) < 2:
            notes.append(
                f"{position}: {len(candidates)} rostered player(s) but fewer than 2 had computed signals "
                f"for {ctx.season} week {ctx.as_of_week} -- skipped, nothing to ground a comparison in."
            )
            continue
        grounded.sort(key=lambda pair: pair[1], reverse=True)
        starter, alternatives = grounded[0][0], [c for c, _ in grounded[1:]]
        reasoning = f"Start {starter['name']} at {position}: {_fmt_signal_row(starter['row'])}."
        for alt in alternatives:
            reasoning += f" By comparison, {alt['name']}: {_fmt_signal_row(alt['row'])}."
        entries.append(
            {
                "position": position,
                "recommended_starter": {
                    "player_id": starter["player_id"],
                    "name": starter["name"],
                    "team": starter["team"],
                    **_stale_fields(starter["row"]),
                },
                "alternatives_considered": [
                    {
                        "player_id": alt["player_id"],
                        "name": alt["name"],
                        "team": alt["team"],
                        "signals_summary": _fmt_signal_row(alt["row"]),
                        **_stale_fields(alt["row"]),
                    }
                    for alt in alternatives
                ],
                "reasoning": reasoning,
            }
        )

    header = _report_header(ctx, "start_sit")
    header["entries"] = entries
    header["notes"] = notes
    return header


def _drop_report(
    ctx: recommend.RecommendContext,
    signals_by_id: dict[str, dict],
    fallback_season: int | None,
    fallback_rows: dict[str, dict],
    bottom_n: int = 3,
) -> dict:
    resolved, unresolved = _resolve_roster_with_signals(ctx, signals_by_id, fallback_season, fallback_rows)

    notes = []
    if unresolved:
        notes.append(f"Could not identity-resolve {len(unresolved)} rostered player(s): {', '.join(unresolved)}.")
    stale_note = _stale_note(resolved)
    if stale_note:
        notes.append(stale_note)

    scored = [(c, _opportunity_score(c["row"])) for c in resolved]
    grounded = [(c, score) for c, score in scored if score is not None]
    ungrounded = [c for c, score in scored if score is None]
    if ungrounded:
        notes.append(
            f"{len(ungrounded)} rostered player(s) had no computed signals and were excluded from ranking: "
            f"{', '.join(c['name'] for c in ungrounded)}."
        )

    grounded.sort(key=lambda pair: pair[1])
    weakest = grounded[: min(bottom_n, len(grounded))]
    entries = [
        {
            "player_id": c["player_id"],
            "name": c["name"],
            "position": c["position"],
            "team": c["team"],
            "weakness_reasons": _weakness_reasons(c["row"]),
            "signals_summary": _fmt_signal_row(c["row"]),
            **_stale_fields(c["row"]),
        }
        for c, _ in weakest
    ]

    header = _report_header(ctx, "drop")
    header["entries"] = entries
    header["notes"] = notes
    return header


def _rostered_nflverse_ids(raw_dir: Path, player_idx: pl.DataFrame) -> set[str]:
    """Every nflverse player_id rostered by ANY team in the league (not
    just "my" roster) -- resolved via the same structured name resolution
    player_index.resolve_player() already provides, never a guess. An
    ambiguous resolution adds every candidate (safer to over-exclude a
    name that might be rostered than to list an actually-rostered player
    as available); an unresolved name is simply skipped."""
    rostered_ids: set[str] = set()
    for player in lookup.all_rostered_players(raw_dir):
        name = player.get("full_name")
        if not name:
            continue
        result = player_index.resolve_player(name, player_idx)
        rostered_ids.update(match.player_id for match in result.candidates)
    return rostered_ids


def _waiver_pickups_report(
    raw_dir: Path,
    ctx: recommend.RecommendContext,
    signals_by_id: dict[str, dict],
    fallback_season: int | None,
    fallback_rows: dict[str, dict],
    top_n: int = 10,
) -> dict:
    rostered_ids = _rostered_nflverse_ids(raw_dir, ctx.player_idx)

    candidates = []
    for player in ctx.player_idx.to_dicts():
        if player["player_id"] in rostered_ids:
            continue
        row = _signal_row(player["player_id"], signals_by_id, fallback_season, fallback_rows)
        if row is None:
            continue  # no measured usage/signals, current or prior season -- nothing to ground a pickup in
        candidates.append({**player, "row": row})

    scored = [(c, _opportunity_score(c["row"])) for c in candidates]
    scored = [(c, score) for c, score in scored if score is not None]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    top = scored[: min(top_n, len(scored))]

    entries = [
        {
            "player_id": c["player_id"],
            "name": c["player_name"],
            "position": c["position"],
            "team": c["team"],
            "opportunity_score": round(score, 3),
            "reasoning": f"{c['player_name']} ({c['position']}, {c['team']}): {_fmt_signal_row(c['row'])}.",
            **_stale_fields(c["row"]),
        }
        for c, score in top
    ]

    notes = [
        f"Considered {len(candidates)} unrostered player(s) with computed signals out of "
        f"{len(ctx.player_idx)} in the full skill-position player pool "
        f"({len(rostered_ids)} nflverse player_id(s) excluded as rostered somewhere in the league)."
    ]
    stale = [c for c, _ in top if c["row"].get("stale")]
    if stale:
        notes.append(
            f"{len(stale)} of the {len(top)} listed pickup(s) had no current-season signal yet and fell "
            f"back to stale {stale[0]['row'].get('source_season')} season-end data: "
            f"{', '.join(c['player_name'] for c in stale)}."
        )

    return {
        "report_type": "waiver_pickups",
        "season": ctx.season,
        "as_of_week": ctx.as_of_week,
        "entries": entries,
        "notes": notes,
    }


def generate_report(
    report_type: str,
    raw_dir: Path = RAW_DIR,
    persist_dir: Path = CHROMA_DIR,
    season: int | None = None,
    as_of_week: int | None = None,
    signals_dir: Path = SIGNALS_DIR,
    waiver_top_n: int = 10,
    drop_bottom_n: int = 3,
) -> dict:
    """Generate one of the three buildable Phase 3.5 report types
    (start_sit, drop, waiver_pickups -- trade suggestions are explicitly
    out of scope, see module docstring). Reuses recommend.py's existing
    tools for roster/identity/record/matchup lookups and reads the
    already-computed signals table directly for ranking (see module
    docstring for why). Never calls the Claude API -- see module
    docstring.

    start_sit and drop require MY_ROSTER_ID to be configured (same
    requirement as recommend.py's "my"-flavored tools) since both report
    on your own roster; waiver_pickups doesn't need it since it operates
    over the whole league's rosters minus the full player pool.

    A player with no current-season signal at all falls back to their
    most recent prior season's data, explicitly marked stale in every
    output (never silently) -- see module docstring's "Prior-season
    signal fallback" section.
    """
    if report_type not in REPORT_TYPES:
        raise ValueError(f"Unknown report_type {report_type!r} -- must be one of {REPORT_TYPES}.")

    league_path = raw_dir / "league.json"
    if not league_path.exists():
        raise RuntimeError(f"{league_path} doesn't exist -- run `python -m src.ingest.sleeper` first.")

    if season is None or as_of_week is None:
        # Reuses recommend.py's own season/week inference (Sleeper's
        # current-week state) rather than duplicating it here.
        inferred_season, inferred_week = recommend._infer_season_and_week(raw_dir)
        season = season if season is not None else inferred_season
        as_of_week = as_of_week if as_of_week is not None else inferred_week

    ctx = recommend.RecommendContext(
        raw_dir=raw_dir,
        persist_dir=persist_dir,
        season=season,
        as_of_week=as_of_week,
        player_idx=player_index.build_player_index(season),
    )
    signals_by_id = _load_signals_table(signals_dir, season, as_of_week)
    fallback_season, fallback_rows = _load_prior_season_fallback_table(signals_dir, season)

    if report_type == "start_sit":
        return _start_sit_report(ctx, signals_by_id, fallback_season, fallback_rows)
    if report_type == "drop":
        return _drop_report(ctx, signals_by_id, fallback_season, fallback_rows, bottom_n=drop_bottom_n)
    return _waiver_pickups_report(raw_dir, ctx, signals_by_id, fallback_season, fallback_rows, top_n=waiver_top_n)


def _print_report(report: dict) -> None:
    print(json.dumps(report, indent=2, default=str))


def main() -> None:
    import argparse

    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Ask Madden: generate a structured roster/waiver report")
    parser.add_argument("report_type", choices=REPORT_TYPES)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--as-of-week", type=int, default=None)
    args = parser.parse_args()

    report = generate_report(args.report_type, season=args.season, as_of_week=args.as_of_week)
    _print_report(report)


if __name__ == "__main__":
    main()
