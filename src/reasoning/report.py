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
signal at all (see `ranking.load_signals_table`'s docstring for exactly what
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
which is deliberately confined to the signal-*loading* layer (
ranking.py's `load_signals_table`/`load_prior_season_fallback_table` and
`retrieve.py`'s `query_player_signal_with_fallback`). Revisit N if
`matchup_signals.py` ever gains that field.

**Where the ranking actually lives (Chat/Feed verdict alignment).** The
weights, thresholds, signals-table loading, prior-season fallback,
opportunity score and number-citing formatter all moved verbatim to
src/reasoning/ranking.py, and the "sort by score, top one starts" step
that used to be inlined in `_start_sit_report` is now
`ranking.rank_candidates()`. Nothing about this module's output changed
(tests/test_report.py pins it); what changed is that recommend.py's
`rank_players` chat tool now calls the exact same code, so Chat's
head-to-head verdict and this report's start_sit verdict can no longer
diverge on the same signals. See ranking.py's docstring for the real
usage-testing case that motivated it and why the shared code couldn't
simply stay here (circular import).
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.rag import lookup, player_index
from src.rag.embed import CHROMA_DIR, RAW_DIR
from src.reasoning import recommend
from src.reasoning.league import load_league
from src.reasoning.ranking import (
    SIGNALS_DIR,
    SignalTables,
    fmt_signal_row,
    opportunity_score,
    rank_candidates,
    stale_fields,
    weakness_reasons,
)

REPORT_TYPES = ("start_sit", "drop", "waiver_pickups")


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


def _resolve_roster_with_signals(
    ctx: recommend.RecommendContext, tables: SignalTables
) -> tuple[list[dict], list[str]]:
    """get_my_roster + get_player_signals (via dispatch_tool -- the real
    tools, not a reimplementation) for every rostered player, joined with
    that player's raw numeric row (current-season, or a stale prior-season
    fallback -- see ranking.signal_row) for ranking. Returns
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
                "row": tables.row_for(player_id),
            }
        )
    return resolved, unresolved


def _report_header(ctx: recommend.RecommendContext, report_type: str) -> dict:
    team = lookup.current_roster(ctx.raw_dir, roster_id=ctx.roster_id)
    record = recommend.dispatch_tool("get_team_record", {}, ctx)
    matchup = recommend.dispatch_tool("get_current_matchup", {}, ctx)
    return {
        "report_type": report_type,
        "league_id": ctx.league.league_id if ctx.league else None,
        "league_name": ctx.league.name if ctx.league else None,
        "season": ctx.season,
        "as_of_week": ctx.as_of_week,
        "roster_id": team.get("roster_id"),
        "team_name": team.get("team_name"),
        "record": record if "error" not in record else None,
        "current_matchup": matchup,
    }


def _start_sit_report(ctx: recommend.RecommendContext, tables: SignalTables) -> dict:
    resolved, unresolved = _resolve_roster_with_signals(ctx, tables)

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
        if len(candidates) < 2:
            continue
        # The verdict itself -- the same ranking.rank_candidates() call
        # Chat's rank_players tool makes, so the two surfaces can't
        # disagree. On a genuine tie the report keeps its long-standing
        # behavior (ranked[] is a stable sort, so the first candidate in
        # roster order leads) rather than dropping the entry; Chat is
        # told to say "tied" instead. See ranking.py's docstring.
        ranking = rank_candidates(candidates)
        if len(ranking["ranked"]) < 2:
            notes.append(
                f"{position}: {len(candidates)} rostered player(s) but fewer than 2 had computed signals "
                f"for {ctx.season} week {ctx.as_of_week} -- skipped, nothing to ground a comparison in."
            )
            continue
        starter, alternatives = ranking["ranked"][0], ranking["ranked"][1:]
        reasoning = f"Start {starter['name']} at {position}: {starter['signals_summary']}."
        for alt in alternatives:
            reasoning += f" By comparison, {alt['name']}: {alt['signals_summary']}."
        entries.append(
            {
                "position": position,
                "recommended_starter": {
                    "player_id": starter["player_id"],
                    "name": starter["name"],
                    "team": starter["team"],
                    "stale": starter["stale"],
                    "source_season": starter["source_season"],
                    "source_as_of_week": starter["source_as_of_week"],
                },
                "alternatives_considered": [
                    {
                        "player_id": alt["player_id"],
                        "name": alt["name"],
                        "team": alt["team"],
                        "signals_summary": alt["signals_summary"],
                        "stale": alt["stale"],
                        "source_season": alt["source_season"],
                        "source_as_of_week": alt["source_as_of_week"],
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


def _drop_report(ctx: recommend.RecommendContext, tables: SignalTables, bottom_n: int = 3) -> dict:
    resolved, unresolved = _resolve_roster_with_signals(ctx, tables)

    notes = []
    if unresolved:
        notes.append(f"Could not identity-resolve {len(unresolved)} rostered player(s): {', '.join(unresolved)}.")
    stale_note = _stale_note(resolved)
    if stale_note:
        notes.append(stale_note)

    scored = [(c, opportunity_score(c["row"])) for c in resolved]
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
            "weakness_reasons": weakness_reasons(c["row"]),
            "signals_summary": fmt_signal_row(c["row"]),
            **stale_fields(c["row"]),
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
    raw_dir: Path, ctx: recommend.RecommendContext, tables: SignalTables, top_n: int = 10
) -> dict:
    rostered_ids = _rostered_nflverse_ids(raw_dir, ctx.player_idx)

    candidates = []
    for player in ctx.player_idx.to_dicts():
        if player["player_id"] in rostered_ids:
            continue
        row = tables.row_for(player["player_id"])
        if row is None:
            continue  # no measured usage/signals, current or prior season -- nothing to ground a pickup in
        candidates.append({**player, "row": row})

    scored = [(c, opportunity_score(c["row"])) for c in candidates]
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
            "reasoning": f"{c['player_name']} ({c['position']}, {c['team']}): {fmt_signal_row(c['row'])}.",
            **stale_fields(c["row"]),
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
        "league_id": ctx.league.league_id if ctx.league else None,
        "league_name": ctx.league.name if ctx.league else None,
        "season": ctx.season,
        "as_of_week": ctx.as_of_week,
        "entries": entries,
        "notes": notes,
    }


def generate_report(
    report_type: str,
    league_id: str,
    raw_dir: Path = RAW_DIR,
    persist_dir: Path = CHROMA_DIR,
    season: int | None = None,
    as_of_week: int | None = None,
    signals_dir: Path = SIGNALS_DIR,
    waiver_top_n: int = 10,
    drop_bottom_n: int = 3,
    roster_id: str | None = None,
) -> dict:
    """Generate one of the three buildable Phase 3.5 report types
    (start_sit, drop, waiver_pickups -- trade suggestions are explicitly
    out of scope, see module docstring). Reuses recommend.py's existing
    tools for roster/identity/record/matchup lookups and reads the
    already-computed signals table directly for ranking (see module
    docstring for why). Never calls the Claude API -- see module
    docstring.

    league_id: the Sleeper league to report on -- required (Phase 5.1),
    resolved and verified against raw_dir's ingested league.json via
    src/reasoning/league.py's load_league() exactly as recommend() does;
    a raw_dir holding a different league raises LeagueMismatchError
    rather than reporting on the wrong roster. Every report's header
    echoes league_id/league_name. Note that no ranking here is
    scoring-format-dependent (see league.py's docstring): the league's
    scoring_settings are loaded and verified but not consumed by the
    opportunity score, so two leagues with identical rosters and
    different scoring get identical reports today -- a documented
    property, not an oversight (a points-based, scoring-aware ranking is
    Phase 6's proxy work, not parameterization).

    start_sit and drop need to know which roster is yours -- roster_id
    explicitly (Phase 5.2, what the API server passes from the session),
    else the MY_ROSTER_ID environment variable (the single-league
    CLI/.env convention, unchanged; same fallback as recommend.py's
    "my"-flavored tools). waiver_pickups needs neither since it operates
    over the whole league's rosters minus the full player pool.

    A player with no current-season signal at all falls back to their
    most recent prior season's data, explicitly marked stale in every
    output (never silently) -- see module docstring's "Prior-season
    signal fallback" section.

    Loads .env itself so MY_ROSTER_ID (needed for start_sit/drop) is
    available whether this is called via the CLI or imported directly --
    see src/reasoning/recommend.py's recommend() for the same fix and
    why it's needed (this function had the identical gap: env loading
    only happened in this module's own main(), not here). Phase 5.2:
    once per process via recommend.load_dotenv_once(), not per call.
    """
    recommend.load_dotenv_once()

    if report_type not in REPORT_TYPES:
        raise ValueError(f"Unknown report_type {report_type!r} -- must be one of {REPORT_TYPES}.")

    # Phase 5.1: same explicit, verified league resolution as recommend().
    league = load_league(league_id, raw_dir=raw_dir, persist_dir=persist_dir)

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
        league=league,
        roster_id=str(roster_id) if roster_id is not None else None,
        signals_dir=signals_dir,
    )
    tables = SignalTables.load(signals_dir, season, as_of_week)

    if report_type == "start_sit":
        return _start_sit_report(ctx, tables)
    if report_type == "drop":
        return _drop_report(ctx, tables, bottom_n=drop_bottom_n)
    return _waiver_pickups_report(raw_dir, ctx, tables, top_n=waiver_top_n)


def _print_report(report: dict) -> None:
    print(json.dumps(report, indent=2, default=str))


def main() -> None:
    import argparse
    import os

    # See src/reasoning/recommend.py's main() for why the CLI loads .env
    # itself on top of generate_report() doing so: --league-id's default
    # is read at argparse time, before generate_report() runs.
    recommend.load_dotenv_once()
    parser = argparse.ArgumentParser(description="Ask Madden: generate a structured roster/waiver report")
    parser.add_argument("report_type", choices=REPORT_TYPES)
    parser.add_argument(
        "--league-id",
        default=os.environ.get("SLEEPER_LEAGUE_ID"),
        help="the Sleeper league to report on (Phase 5.1: required -- defaults to SLEEPER_LEAGUE_ID from .env)",
    )
    parser.add_argument(
        "--roster-id", default=None, help="which roster is yours (Phase 5.2) -- defaults to MY_ROSTER_ID from .env"
    )
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--as-of-week", type=int, default=None)
    args = parser.parse_args()

    if not args.league_id:
        raise SystemExit(
            "a Sleeper league ID is required: pass --league-id <id>, or set SLEEPER_LEAGUE_ID in .env "
            "(see .env.example)"
        )

    report = generate_report(
        args.report_type, args.league_id, season=args.season, as_of_week=args.as_of_week, roster_id=args.roster_id
    )
    _print_report(report)


if __name__ == "__main__":
    main()
