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

**start_sit follows the league's actual starting slots** (since
2026-09-20 -- the first friend's league started two RBs and the report
recommended one, which is a wrong answer, not a simplification). It
reads the league's Sleeper `roster_positions`: a position with N
dedicated slots gets its top N ranked players as `recommended_starters`
(and the rest as alternatives); FLEX-type slots (FLEX, SUPER_FLEX,
REC_FLEX, WRRB_FLEX -- see src/rag/lookup.py's FLEX_ELIGIBILITY) are
then filled from the players left over after the dedicated slots, ranked
together across positions by the same rank_candidates() call. Only
positions with computed signals take part (QB/RB/WR/TE -- K and DEF have
no matchup signals, so those slots are left to the user). A league whose
roster_positions aren't known falls back to one slot per position, the
pre-2026-09-20 behavior. This is a greedy fill (dedicated slots first,
then flex), not a global optimizer; with one ranking score per player the
two agree unless a player is worth more in a flex slot than a weaker
teammate is in a dedicated one, which the score can't express anyway.

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
from src.rag import teams
from src.reasoning.ranking import is_backup_qb as ranking_is_backup_qb
from src.reasoning.ranking import on_bye as ranking_on_bye
from src.reasoning.ranking import (
    availability_adjusted_score,
    availability_of,
    SIGNALS_DIR,
    SignalTables,
    fmt_signal_row,
    key_stats,
    opportunity_score,
    rank_candidates,
    stale_fields,
    weakness_reasons,
)

REPORT_TYPES = ("start_sit", "drop", "waiver_pickups")

# A waiver pickup needs a role now. Once the season has data, a player
# with fewer current-season plays than this is not ranked as one -- the
# same reasoning as the last-season-only rule below, a week later: on
# 2026-10-05 a back with 2 plays and a 25-point team total led a real list
# over backs with real roles. Ten plays is under one game's worth for a
# starter and more than a goal-line cameo.
MIN_PICKUP_PLAYS = 10
PRE_DRAFT_STATUS = "pre_draft"
DYNASTY_LIKE = {"dynasty", "keeper"}
YOUNG_PLAYER_MAX_YEARS_EXP = 1  # rookies (0) and second-year players (1)
PRE_DRAFT_NOTE = "This league hasn't drafted yet (Sleeper status: pre_draft), so no team has any players."


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
    (resolved_candidates, notes) -- a player whose name can't be
    identity-resolved, or whose position has no signals at all (K, DEF),
    is reported in the notes, never silently dropped."""
    roster_result = recommend.dispatch_tool("get_my_roster", {}, ctx)
    if "error" in roster_result:
        raise RuntimeError(roster_result["error"])

    resolved: list[dict] = []
    unresolved: list[str] = []
    uncovered: list[str] = []
    for player in roster_result["players"]:
        name = player.get("name")
        position = (player.get("position") or "").upper()
        if not name and position != "DEF":
            continue
        if position == "DEF":
            # A team defense: its id IS the team (Sleeper and nflverse
            # agree), so no name resolution; its row is points per game
            # under this league's scoring (2026-10-06).
            abbr = player.get("team") or player.get("player_id")
            resolved.append({
                "player_id": abbr, "name": teams.defense_display_name(abbr), "position": "DEF", "team": abbr,
                "injury_status": None, "depth_chart_order": None, "years_exp": None, "sleeper_id": abbr,
                "row": tables.row_for(abbr),
            })
            continue
        if position not in player_index.INDEX_POSITIONS:
            # An IDP or anything else the signals never cover: nothing to
            # resolve against. Reported as that, not as a resolution
            # failure.
            uncovered.append(f"{name} ({player.get('position') or '?'})")
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
                "injury_status": player.get("injury_status"),
                "depth_chart_order": player.get("depth_chart_order"),  # Sleeper's depth chart -> ranking.is_backup_qb
                "years_exp": player.get("years_exp"),
                "sleeper_id": player.get("player_id"),  # Sleeper's own id -> its headshot CDN in the UI
                "row": tables.row_for(player_id),
            }
        )
    return resolved, _roster_notes(unresolved, uncovered)


def _roster_notes(unresolved: list[str], uncovered: list[str]) -> list[str]:
    notes = []
    if uncovered:
        notes.append(f"No signals exist for this position, so they aren't ranked: {', '.join(uncovered)}.")
    if unresolved:
        notes.append(f"Could not identity-resolve {len(unresolved)} rostered player(s): {', '.join(unresolved)}.")
    return notes


# Sleeper injury designations that mean "cannot play this week". Such a
# player is left out of the start/sit decision and named in the notes,
# rather than ranked as if available (a rookie RB on IR was showing up as
# a SIT candidate with last season's numbers, which reads like a call).
# Questionable/Doubtful stay in -- that is exactly the call the user wants
# help with -- and the entry carries the status so the UI can flag it.
UNAVAILABLE_STATUSES = {"Out", "IR", "PUP", "Sus", "COV", "DNR", "NA"}


def _report_header(ctx: recommend.RecommendContext, report_type: str) -> dict:
    team = lookup.current_roster(ctx.raw_dir, roster_id=ctx.roster_id)
    record = recommend.dispatch_tool("get_team_record", {}, ctx)
    matchup = recommend.dispatch_tool("get_current_matchup", {}, ctx)
    return {
        "report_type": report_type,
        "league_id": ctx.league.league_id if ctx.league else None,
        "league_name": ctx.league.name if ctx.league else None,
        "league_type": ctx.league.league_type if ctx.league else None,
        "season": ctx.season,
        "as_of_week": ctx.as_of_week,
        "roster_id": team.get("roster_id"),
        "team_name": team.get("team_name"),
        "record": record if "error" not in record else None,
        "current_matchup": matchup,
    }


def _slot_counts(roster_positions: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for slot in lookup.starter_slots(roster_positions):
        counts[slot] = counts.get(slot, 0) + 1
    return counts


def _player_ref(ranked: dict, injury_status: str | None = None, sleeper_id: str | None = None) -> dict:
    return {
        "player_id": ranked["player_id"],
        "sleeper_id": sleeper_id,
        "name": ranked["name"],
        "team": ranked["team"],
        "position": ranked.get("position"),
        "injury_status": injury_status,
        # Why this player is a SIT regardless of his numbers (on bye, listed Out/IR) -- None for a ranked alternative.
        "sit_reason": ranked.get("sit_reason"),
        # Sleeper's depth chart and whether it makes him a backup QB -- see ranking.DEPTH_CHART_DESCRIPTION.
        "depth_chart_order": ranked.get("depth_chart_order"),
        "backup_qb": bool(ranked.get("backup_qb")),
        # P(this player outscores the next one in the ranking) and its label -- see ranking.CONFIDENCE_DESCRIPTION.
        "win_probability_vs_next": ranked.get("win_probability_vs_next"),
        "confidence_label": ranked.get("confidence_label"),
        # Injury-report availability (None for a healthy player) -- see ranking.AVAILABILITY_DESCRIPTION.
        "report_status": ranked.get("report_status"),
        "practice_status": ranked.get("practice_status"),
        "play_probability": ranked.get("play_probability"),
        "played_output_factor": ranked.get("played_output_factor"),
        "expected_output": ranked.get("expected_output"),
        "key_stats": ranked.get("key_stats") or {},
        "signals_summary": ranked["signals_summary"],
        "stale": ranked["stale"],
        "source_season": ranked["source_season"],
        "source_as_of_week": ranked["source_as_of_week"],
    }


def _start_sit_entry(
    slot: str, n_slots: int, starters: list[dict], alternatives: list[dict], eligible: tuple[str, ...] | None,
    injuries: dict[str, str | None] | None = None,
    sleeper_ids: dict[str, str | None] | None = None,
) -> dict:
    injuries = injuries or {}
    sleeper_ids = sleeper_ids or {}
    where = slot if n_slots == 1 else f"{slot} ({n_slots} slots)"
    open_slots = max(0, n_slots - len(starters))
    if not starters:
        reasoning = f"No available player for {where} this week."
    elif len(starters) == 1:
        reasoning = f"Start {starters[0]['name']} at {where}: {starters[0]['signals_summary']}."
    else:
        names = ", ".join(s["name"] for s in starters[:-1]) + f" and {starters[-1]['name']}"
        reasoning = f"Start {names} at {where}. " + " ".join(f"{s['name']}: {s['signals_summary']}." for s in starters)
    for alt in alternatives:
        if alt.get("sit_reason"):
            reasoning += f" {alt['name']} sits: {alt['sit_reason']}."
        else:
            reasoning += f" By comparison, {alt['name']}: {alt['signals_summary']}."
    if open_slots:
        reasoning += (f" {open_slots} {slot} slot(s) have no available player this week -- pick one up or stream one "
                      "(see Waiver targets).")
    # A designated starter: say the base rate, and who steps in if he sits
    # (the best alternative not already starting here and not sidelined
    # himself -- a player on bye or Out cannot step in).
    if_out = []
    standby = [a for a in alternatives if not a.get("sit_reason")]
    for s in starters:
        if s.get("play_probability") is None:
            continue
        replacement = standby[0] if standby else None
        if_out.append({"starter": s["name"], "player_id": s["player_id"],
                       "replacement": replacement["name"] if replacement else None,
                       "replacement_player_id": replacement["player_id"] if replacement else None})
        reasoning += " " + availability_sentence(s) + (f" If he sits, {replacement['name']} steps in." if replacement else "")
    entry = {
        "position": slot,
        "slots": n_slots,
        # The top pick, kept under its long-standing name -- it is the
        # verdict Chat's rank_players is held to (see ranking.py). None
        # only when nobody at the position can play this week.
        "recommended_starter": (
            _player_ref(starters[0], injuries.get(starters[0]["player_id"]), sleeper_ids.get(starters[0]["player_id"]))
            if starters else None
        ),
        "recommended_starters": [_player_ref(s, injuries.get(s["player_id"]), sleeper_ids.get(s["player_id"])) for s in starters],
        "alternatives_considered": [_player_ref(a, injuries.get(a["player_id"]), sleeper_ids.get(a["player_id"])) for a in alternatives],
        # Slots at this position with no available player this week (2026-10-06).
        "open_slots": open_slots,
        "reasoning": reasoning,
        "if_out": if_out,
    }
    if eligible is not None:
        entry["eligible_positions"] = list(eligible)
    return entry


def availability_sentence(entry: dict) -> str:
    """'X is Questionable (limited practice): players in that spot have
    played 62% of the time and scored 78% of their usual when they did,
    so the ranking counts him at 48% of himself.'"""
    status = entry.get("report_status") or entry.get("injury_status") or "on the injury report"
    practice = entry.get("practice_status")
    p, k, m = entry.get("play_probability"), entry.get("played_output_factor"), entry.get("expected_output")
    where = f"{status} ({practice})" if practice else f"{status} (no practice report)"
    text = f"{entry['name']} is {where}: players in that spot have played {p:.0%} of the time"
    if k is not None:
        text += f" and scored {k:.0%} of their usual when they did"
    if m is not None:
        text += f", so the ranking counts him at {m:.0%} of himself"
    return text + " -- a base rate for the situation, not a read on his injury."



def _sidelined_refs(benched: list[dict], reasons: dict[str, str]) -> list[dict]:
    """Ranked-style entries for players who can't start this week, each
    with its `sit_reason`, so the Feed shows them as SIT cards with the
    reason instead of dropping them. Their numbers come from the same
    ranking call; a player with no rankable row gets the reason as his
    summary."""
    ranked = {e["player_id"]: e for e in rank_candidates(benched)["ranked"]} if benched else {}
    refs = []
    for c in benched:
        entry = ranked.get(c["player_id"]) or {
            "player_id": c["player_id"], "name": c["name"], "team": c.get("team"), "position": c.get("position"),
            "signals_summary": reasons[c["player_id"]], "key_stats": {}, "stale": False,
            "source_season": None, "source_as_of_week": None,
        }
        refs.append({**entry, "sit_reason": reasons[c["player_id"]]})
    return refs


def _start_sit_report(ctx: recommend.RecommendContext, tables: SignalTables) -> dict:
    resolved, notes = _resolve_roster_with_signals(ctx, tables)

    entries = []
    # Players who cannot be started this week stay VISIBLE: a SIT card with
    # the reason, inside their position's section (2026-10-06, Victorious
    # Secret: Mahomes on bye was dropped before ranking, Herbert alone was
    # "nothing to decide", and the QB section vanished -- the one call the
    # user needed, "bench Mahomes this week", was a line in the notes).
    sidelined: dict[str, str] = {}
    unavailable = [c for c in resolved if c.get("injury_status") in UNAVAILABLE_STATUSES]
    if unavailable:
        notes.append(
            "Not available this week, left out of the lineup: "
            + ", ".join(f"{c['name']} ({c['injury_status']})" for c in unavailable) + "."
        )
        for c in unavailable:
            sidelined[c["player_id"]] = f"listed {c['injury_status']} by Sleeper"
    byes = ctx.bye_weeks()
    on_bye = [c for c in resolved if c["player_id"] not in sidelined
              and ranking_on_bye(c["row"], ctx.season, ctx.as_of_week, c.get("team"), byes)]
    if on_bye:
        notes.append(
            f"On bye in week {ctx.as_of_week}, left out of the lineup: "
            + ", ".join(f"{c['name']} ({c.get('team') or '?'})" for c in on_bye) + "."
        )
        for c in on_bye:
            sidelined[c["player_id"]] = f"on bye in week {ctx.as_of_week}"
    sidelined_by_position: dict[str, list[dict]] = {}
    for c in resolved:
        if c["player_id"] in sidelined:
            sidelined_by_position.setdefault(c["position"], []).append(c)
    resolved = [c for c in resolved if c["player_id"] not in sidelined]
    # This week's table may not exist yet (Sleeper flips its week the moment
    # Monday night ends; the table follows after the next refresh): say so,
    # because the matchup columns are then last week's.
    table_weeks = [r.get("as_of_week") for r in tables.signals_by_id.values() if r.get("season") == ctx.season and r.get("as_of_week")]
    if table_weeks and max(table_weeks) < ctx.as_of_week:
        notes.append(
            f"Week {ctx.as_of_week}'s signals table has not been computed yet (latest is week {max(table_weeks)}); "
            f"the matchup columns (opponent, team total) are week {max(table_weeks)}'s until the next refresh."
        )
    injuries = {c["player_id"]: c.get("injury_status") for c in resolved}
    sleeper_ids = {c["player_id"]: c.get("sleeper_id") for c in resolved}
    backups = [c for c in resolved if ranking_is_backup_qb(c.get("position"), c.get("depth_chart_order"))]
    if backups:
        notes.append(
            "Listed as a backup on Sleeper's depth chart, so ranked behind every starting quarterback whatever the "
            "numbers say: " + ", ".join(f"{c['name']} (QB{int(c['depth_chart_order'])})" for c in backups) + "."
        )
    designated = [
        (c, availability_of(c["row"], c.get("injury_status"), c.get("position")))
        for c in resolved
    ]
    designated = [(c, a) for c, a in designated if a.get("play_probability") is not None]
    if designated:
        notes.append(
            "On the injury report this week, counted at a share of their usual output (historical rate of playing "
            "x output when playing): "
            + "; ".join(
                f"{c['name']} {a['report_status'] or c.get('injury_status')}"
                + (f", {a['practice_status']}" if a.get("practice_status") else "")
                + f" -- plays {a['play_probability']:.0%}, counted at {a['expected_output']:.0%}"
                for c, a in designated
            )
            + "."
        )

    by_position: dict[str, list[dict]] = {}
    for candidate in resolved:
        by_position.setdefault(candidate["position"], []).append(candidate)

    # Same rule as the drop report: the stale note covers players who can
    # actually be ranked; one with last season's row but no scored signal
    # is "not enough usage", not "fell back to stale data".
    rankable = [c for c in resolved if opportunity_score(c["row"]) is not None]
    stale_note = _stale_note(rankable)
    if stale_note:
        notes.append(stale_note)
    unrankable = [c for c in resolved if c not in rankable]
    if unrankable:
        notes.append(
            "Not enough usage on record to rank (no target share, red-zone share, efficiency trend or passing "
            "numbers): " + ", ".join(_usage_label(c) for c in unrankable) + "."
        )

    roster_positions = list(ctx.league.roster_positions) if ctx.league else []
    slots = _slot_counts(roster_positions)
    structure_known = bool(slots)
    flex_slots = [(slot, n) for slot, n in slots.items() if slot in lookup.FLEX_ELIGIBILITY]
    flex_eligible_positions = {pos for slot, _ in flex_slots for pos in lookup.FLEX_ELIGIBILITY[slot]}
    by_id = {c["player_id"]: c for c in resolved}
    flex_pool: list[dict] = []  # candidates (not ranked refs) still available for a flex slot

    for position in sorted(set(by_position) | set(sidelined_by_position)):
        candidates = by_position.get(position, [])
        benched = sidelined_by_position.get(position, [])
        n_slots = slots.get(position, 0) if structure_known else 1
        if n_slots == 0:
            # No dedicated slot for this position in this league -- its
            # players only start via a flex slot, if one takes them.
            if position in flex_eligible_positions:
                flex_pool.extend(candidates)
            continue
        if not benched and len(candidates) <= n_slots:
            continue  # nothing to decide: everyone with this position starts (or there is nobody to compare)
        # The verdict itself -- the same ranking.rank_candidates() call
        # Chat's rank_players tool makes, so the two surfaces can't
        # disagree. On a genuine tie the report keeps its long-standing
        # behavior (ranked[] is a stable sort, so the first candidate in
        # roster order leads) rather than dropping the entry; Chat is
        # told to say "tied" instead. See ranking.py's docstring.
        ranking = rank_candidates(candidates) if candidates else {"ranked": [], "unranked": []}
        if not benched and len(ranking["ranked"]) < 2:
            notes.append(
                f"{position}: {len(candidates)} rostered player(s) but fewer than 2 had computed signals "
                f"for {ctx.season} week {ctx.as_of_week} -- skipped, nothing to ground a comparison in."
            )
            continue
        starters, alternatives = ranking["ranked"][:n_slots], ranking["ranked"][n_slots:]
        if position in flex_eligible_positions:
            flex_pool.extend(by_id[alt["player_id"]] for alt in alternatives)
        alternatives = alternatives + _sidelined_refs(benched, sidelined)
        entries.append(_start_sit_entry(position, n_slots, starters, alternatives, None, injuries, sleeper_ids))

    for slot, n_slots in flex_slots:
        eligible = lookup.FLEX_ELIGIBILITY[slot]
        pool = [c for c in flex_pool if c["position"] in eligible]
        if len(pool) <= n_slots:
            continue
        ranking = rank_candidates(pool)
        if len(ranking["ranked"]) < 2:
            continue
        starters, alternatives = ranking["ranked"][:n_slots], ranking["ranked"][n_slots:]
        chosen = {s["player_id"] for s in starters}
        flex_pool = [c for c in flex_pool if c["player_id"] not in chosen]
        entries.append(_start_sit_entry(slot, n_slots, starters, alternatives, eligible, injuries, sleeper_ids))

    header = _report_header(ctx, "start_sit")
    header["entries"] = entries
    header["notes"] = notes
    return header


def _drop_report(ctx: recommend.RecommendContext, tables: SignalTables, bottom_n: int = 3) -> dict:
    resolved, notes = _resolve_roster_with_signals(ctx, tables)
    # Dynasty / keeper leagues: a rookie or second-year player is held for
    # the seasons ahead, and nothing here measures that. Their low usage
    # today is not a drop reason, so they are held out and named (a
    # dynasty owner was told to drop the rookie WR he had just drafted,
    # 2026-09-21). Redraft leagues are unchanged.
    league_type = ctx.league.league_type if ctx.league else None
    if league_type in DYNASTY_LIKE:
        young = [c for c in resolved if c.get("years_exp") is not None and c["years_exp"] <= YOUNG_PLAYER_MAX_YEARS_EXP]
        if young:
            notes.append(
                f"{league_type.capitalize()} league: rookies and second-year players are long-term assets the "
                f"usage score can't value, so they aren't drop candidates here: "
                + ", ".join(f"{c['name']} ({'rookie' if c['years_exp'] == 0 else '2nd year'}, {c['position']})" for c in young) + "."
            )
            resolved = [c for c in resolved if c not in young]

    # Kickers and defenses are ranked on points per game alone, which is
    # not comparable to a skill player's score -- they would always be the
    # "weakest". They are streamed, not dropped; the waiver list ranks them.
    units = [c for c in resolved if c.get("position") in ("K", "DEF")]
    if units:
        notes.append("Kickers and defenses aren't drop candidates here (see Waiver targets to stream one): "
                     + ", ".join(c["name"] for c in units) + ".")
        resolved = [c for c in resolved if c not in units]
    scored = [(c, opportunity_score(c["row"])) for c in resolved]
    grounded = [(c, score) for c, score in scored if score is not None]
    ungrounded = [c for c, score in scored if score is None]
    # The stale note only covers players who could actually be ranked --
    # a player who fell back to last season AND turned out to have no
    # scored signal is one story ("not enough usage to rank"), not two.
    stale_note = _stale_note([c for c, score in grounded])
    if stale_note:
        notes.append(stale_note)
    if ungrounded:
        notes.append(
            "Not enough usage on record to rank (no target share, red-zone share, efficiency trend or passing "
            "numbers): " + ", ".join(_usage_label(c) for c in ungrounded) + "."
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
            "key_stats": key_stats(c["row"]),
            "years_exp": c.get("years_exp"),
            "injury_status": c.get("injury_status"),
            "sleeper_id": c.get("sleeper_id"),
            **stale_fields(c["row"]),
        }
        for c, _ in weakest
    ]

    header = _report_header(ctx, "drop")
    header["entries"] = entries
    header["notes"] = notes
    return header


def _usage_label(candidate: dict) -> str:
    row = candidate.get("row")
    if not row:
        return f"{candidate['name']} (no plays on record)"
    plays = row.get("season_plays")
    season = row.get("source_season") if row.get("stale") else None
    return f"{candidate['name']} ({plays} play(s){f' in {season}' if season else ''})"


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

    # Once the current season has a table with anyone in it, a player whose
    # only numbers are last season's has, by construction, no plays this
    # season -- and a waiver pickup with no role now is not a pickup, no
    # matter how their 2025 ended (the first real report's top three were
    # exactly that: 4-8 plays each in 2025, zero in 2026). Before the
    # season's first table exists, last season is all there is and is used,
    # labeled stale as everywhere else.
    season_has_data = bool(tables.signals_by_id)
    # Sleeper's side of each unrostered player: injury designation and depth
    # chart (2026-10-05 -- a real list led with a back on IR with a torn ACL
    # and a QB2; the pool had only ever looked at nflverse usage numbers).
    sleeper_info = lookup.sleeper_player_info_by_name(raw_dir)
    candidates = []
    last_season_only = 0
    too_few_plays = 0
    no_team: list[dict] = []
    unavailable: list[tuple[dict, str]] = []
    backup_qbs: list[tuple[dict, int | None]] = []
    for player in ctx.player_idx.to_dicts():
        if player["player_id"] in rostered_ids:
            continue
        row = tables.row_for(player["player_id"])
        if row is None:
            continue  # no measured usage/signals, current or prior season -- nothing to ground a pickup in
        if season_has_data and row.get("stale"):
            last_season_only += 1
            continue
        if season_has_data and not row.get("unit") and (row.get("season_plays") or 0) < MIN_PICKUP_PLAYS:
            too_few_plays += 1  # a 2-play body with a big implied total led a real list (2026-10-05)
            continue
        info = sleeper_info.get((player["player_name"], player["position"]))
        matched = info is not None  # Sleeper's data has this (name, position); otherwise none of its facts apply
        info = info or {}
        if matched and not info.get("team"):
            no_team.append(player)  # a free agent in Sleeper's data is not a pickup, whatever he did before he was cut
            continue
        if info.get("injury_status") in UNAVAILABLE_STATUSES:
            unavailable.append((player, info["injury_status"]))
            continue
        # A quarterback is a pickup only as his team's listed starter: QB2+
        # is a backup, and no listing at all (every team lists a QB1) means
        # a practice-squad arm. A name Sleeper's data doesn't carry at all
        # is left alone -- there is nothing to say he isn't the starter.
        if player["position"] == "QB" and matched and info.get("depth_chart_order") != 1:
            backup_qbs.append((player, info.get("depth_chart_order")))
            continue
        candidates.append({**player, "row": row, "injury_status": info.get("injury_status"),
                           "depth_chart_order": info.get("depth_chart_order")})

    # Team defenses (2026-10-06): every team not rostered as a DEF in this
    # league, ranked on points per game under its scoring.
    rostered_defs = {p.get("team") or p["player_id"] for p in lookup.all_rostered_players(raw_dir)
                     if (p.get("position") or "").upper() == "DEF"}
    for abbr in teams.NFL_TEAMS:
        if abbr in rostered_defs:
            continue
        row = tables.row_for(abbr)
        if row is None:
            continue
        if season_has_data and row.get("stale"):
            last_season_only += 1
            continue
        candidates.append({"player_id": abbr, "player_name": teams.defense_display_name(abbr), "position": "DEF",
                           "team": abbr, "row": row, "injury_status": None, "depth_chart_order": None})

    scored = []
    for c in candidates:
        score = opportunity_score(c["row"])
        if score is None:
            continue
        # A Questionable/Doubtful pickup is counted at his expected output,
        # exactly as the start/sit ranking counts him.
        avail = availability_of(c["row"], c.get("injury_status"), c.get("position"))
        c["availability"] = avail
        if avail.get("expected_output") is not None:
            score = availability_adjusted_score(score, c["row"], avail["expected_output"])
        scored.append((c, score))
    # Ranked WITHIN position, then interleaved (the best RB, the best WR,
    # the best TE, the best QB, the second-best RB, ...). The score is not
    # position-normalized -- its own description says so -- and once
    # points per game entered it (2026-09-21) a raw cross-position sort
    # returned five backup QBs as the top five pickups in a league that
    # starts one. A pickup list is a per-position question anyway.
    by_position: dict[str, list[tuple[dict, float]]] = {}
    for c, score in sorted(scored, key=lambda pair: pair[1], reverse=True):
        by_position.setdefault(c["position"], []).append((c, score))
    order = [pos for pos in ("RB", "WR", "TE", "QB", "K", "DEF") if pos in by_position] + sorted(
        pos for pos in by_position if pos not in ("RB", "WR", "TE", "QB", "K", "DEF")
    )
    top: list[tuple[dict, float, int]] = []
    depth = 0
    while len(top) < min(top_n, len(scored)):
        added = False
        for pos in order:
            if depth < len(by_position[pos]) and len(top) < top_n:
                c, score = by_position[pos][depth]
                top.append((c, score, depth + 1))
                added = True
        if not added:
            break
        depth += 1

    sleeper_by_name = lookup.sleeper_ids_by_name(raw_dir)
    entries = [
        {
            "player_id": c["player_id"],
            "sleeper_id": sleeper_by_name.get((c["player_name"], c["position"])) or (c["team"] if c["position"] == "DEF" else None),
            "name": c["player_name"],
            "position": c["position"],
            "team": c["team"],
            "position_rank": position_rank,
            "opportunity_score": round(score, 3),
            "injury_status": c.get("injury_status"),
            "depth_chart_order": c.get("depth_chart_order"),
            # No game this week (a pickup for next week, not this one's lineup).
            "bye_this_week": ranking_on_bye(c["row"], ctx.season, ctx.as_of_week),
            "key_stats": {**key_stats(c["row"]), "play_probability": c["availability"].get("play_probability"),
                          "practice_status": c["availability"].get("practice_status"),
                          "expected_output": c["availability"].get("expected_output"),
                          "depth_chart_order": c.get("depth_chart_order")},
            "report_status": c["availability"].get("report_status"),
            "practice_status": c["availability"].get("practice_status"),
            "play_probability": c["availability"].get("play_probability"),
            "played_output_factor": c["availability"].get("played_output_factor"),
            "expected_output": c["availability"].get("expected_output"),
            "reasoning": f"{c['player_name']} ({c['position']}, {c['team']}): {fmt_signal_row(c['row'])}."
            + (f" {availability_sentence({**c['availability'], 'name': c['player_name']})}" if c["availability"].get("play_probability") is not None else ""),
            **stale_fields(c["row"]),
        }
        for c, score, position_rank in top
    ]

    notes = [
        "Ranked within each position (position_rank), then interleaved RB/WR/TE/QB/K/DEF -- the score is not "
        "comparable across positions. Kickers and defenses are ranked on points per game under this league's "
        "scoring alone (no usage signals exist for them); the matchup columns are shown for context only.",
        f"Considered {len(candidates)} unrostered player(s) with computed signals out of "
        f"{len(ctx.player_idx)} in the full skill-position player pool "
        f"({len(rostered_ids)} nflverse player_id(s) excluded as rostered somewhere in the league)."
    ]
    if last_season_only:
        notes.append(
            f"{last_season_only} unrostered player(s) with no {ctx.season} plays yet were not ranked as pickups -- "
            f"a role at the end of last season says nothing about a role now."
        )
    if unavailable:
        shown = ", ".join(f"{p['player_name']} ({status})" for p, status in unavailable[:6])
        more = f", and {len(unavailable) - 6} more" if len(unavailable) > 6 else ""
        notes.append(
            f"Left out, listed by Sleeper as not available to play (Out / IR / PUP / suspended): {shown}{more}."
        )
    byes_listed = [c for c, _, _ in top if ranking_on_bye(c["row"], ctx.season, ctx.as_of_week)]
    if byes_listed:
        notes.append(
            f"On bye in week {ctx.as_of_week} (a pickup for the weeks after, not for this lineup): "
            + ", ".join(f"{c['player_name']} ({c['team']})" for c in byes_listed) + "."
        )
    if too_few_plays:
        notes.append(
            f"{too_few_plays} unrostered player(s) with fewer than {MIN_PICKUP_PLAYS} plays this season were not "
            "ranked as pickups -- too small a role now to call a pickup."
        )
    if no_team:
        notes.append(
            "Left out, not on an NFL roster in Sleeper's data: "
            + ", ".join(p["player_name"] for p in no_team[:6]) + (f", and {len(no_team) - 6} more" if len(no_team) > 6 else "") + "."
        )
    if backup_qbs:
        listed = [(p, d) for p, d in backup_qbs if d is not None]
        unlisted = [p for p, d in backup_qbs if d is None]
        if listed:
            notes.append(
                "Left out, a backup on Sleeper's depth chart (a QB2 plays only if the starter sits): "
                + ", ".join(f"{p['player_name']} (QB{int(depth)})" for p, depth in listed[:6])
                + (f", and {len(listed) - 6} more" if len(listed) > 6 else "") + "."
            )
        if unlisted:
            notes.append(
                f"{len(unlisted)} quarterback(s) with no depth-chart listing in Sleeper's data were not ranked as "
                "pickups -- every team lists a starter, so an unlisted QB is not one."
            )
    stale = [c for c, _, _ in top if c["row"].get("stale")]
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
    player_stats_dir: Path | None = None,
    count_games_played_this_week: bool = False,
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

    # count_games_played_this_week (2026-10-05, the live product's setting --
    # see ranking.SignalTables.points_through_week): points per game count
    # every game already played, including this week's finished games. Off
    # by default so a backtest stays strictly as-of. Set on the context too,
    # so a tool a report calls (get_player_signals) reads the same horizon.
    points_through_week = as_of_week + 1 if count_games_played_this_week else None
    ctx = recommend.RecommendContext(
        raw_dir=raw_dir,
        persist_dir=persist_dir,
        season=season,
        as_of_week=as_of_week,
        player_idx=player_index.build_player_index(season),
        league=league,
        roster_id=str(roster_id) if roster_id is not None else None,
        signals_dir=signals_dir,
        points_through_week=points_through_week,
    )
    tables = SignalTables.load(
        signals_dir, season, as_of_week, scoring_settings=league.scoring_settings, stats_dir=player_stats_dir,
        points_through_week=points_through_week,
    )

    # A league that hasn't drafted has no rosters: nothing to start, sit or
    # drop, and "unrostered" is the entire NFL. Found live 2026-09-21 -- a
    # pre_draft league's feed told its owner to target Jahmyr Gibbs on
    # waivers. Say so instead.
    if league.status == PRE_DRAFT_STATUS:
        if report_type in ("start_sit", "drop"):
            header = _report_header(ctx, report_type)
            header["entries"] = []
            header["notes"] = [PRE_DRAFT_NOTE]
            return header
        report = _waiver_pickups_report(raw_dir, ctx, tables, top_n=waiver_top_n)
        report["notes"].insert(0, PRE_DRAFT_NOTE + " Every player is unrostered, so this list is the full pool ranked "
                               "by the score -- a draft board, not a waiver list.")
        return report

    if report_type == "start_sit":
        result = _start_sit_report(ctx, tables)
    elif report_type == "drop":
        result = _drop_report(ctx, tables, bottom_n=drop_bottom_n)
    else:
        result = _waiver_pickups_report(raw_dir, ctx, tables, top_n=waiver_top_n)
    result["points_through_week"] = points_through_week
    if count_games_played_this_week:
        result["notes"].append(points_horizon_note(as_of_week))
    return result


def points_horizon_note(as_of_week: int) -> str:
    """What the points-per-game numbers cover when the live product counts
    this week's finished games (the matchup columns are still this
    week's)."""
    return (
        f"Points per game count every game played so far this season, including any week-{as_of_week} games already "
        f"final; the matchup columns (opponent, team total) are for week {as_of_week}."
    )


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
