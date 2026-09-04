# CLAUDE.md — Ask Madden

Context for Claude Code sessions on this repo. Read this first, then
`ask-madden-project-spec.md` for full detail on any phase.

## What this project is
An AI fantasy football assistant. Retrieval-augmented generation over
league/player data, plus a computed matchup-signals layer, feeding a
Claude tool-use agent that gives recommendations with real reasoning —
not just a projection number.

Started scoped to one league (Victorious Secret 3.0, Sleeper league ID
1389341490030862336, 12 teams, half-PPR). **Current scope includes a
final phase (Phase 5) that generalizes this into a small real product**:
multi-league, hosted, anyone can paste in their own Sleeper league ID.
This is a portfolio project — the product doesn't need to be polished
or monetized, it needs to genuinely work for more than one league.

## Phase status (update this section as phases complete)
- Phase 1 (RAG foundation): complete — Sleeper ingest, ChromaDB embed
  pipeline, structured lookups, CLI interface, eval harness all done.
  `ground_truth.jsonl` has real 2024 nflverse data computed with the
  league's actual scoring settings; `run_eval.py` runs as-of-week-filtered
  retrieval scoring (Sleeper-fact-driven). Note: `ground_truth.jsonl` is
  nflverse-driven but not yet consumed by anything — reserved for Phase 3
  decision-accuracy grading. Qualitative hand-curated dilemma seed set
  still deferred, tracked separately.
- Phase 2 (signals layer): implemented — nflverse pbp/schedules and NGS
  ingest, odds/game-script derivation, `matchup_signals.py`'s core
  signals (all as-of-week filtered, validated against real 2024 data),
  and signal chunks wired into the RAG corpus. Two documented gaps: line
  movement (needs a live odds API this project doesn't have) and
  matchup-fit score (still Phase 4, needs coverage classification). See
  TODO.md's Phase 2 section for the full checklist and design deviations
  (game script sourced from nflverse schedules rather than a separate
  odds API; CROE approximated via NGS's YAC-over-expectation/separation
  since NGS doesn't publish a literal catch-rate-over-expected stat).
- Phase 3 (reasoning/recommend.py): implemented — `src/reasoning/recommend.py`'s
  Claude tool-use agent decides between structured roster lookup, structured
  name-resolved signal lookup, and semantic search, ending with a terminal
  `submit_recommendation` tool call so the result is a parseable structure.
  Fixed a real named-player retrieval bug along the way (confirmed live:
  "Christian McCaffrey" returned his brother Luke's signal chunk via pure
  embedding search) by adding `src/rag/player_index.py` (structured
  exact/fuzzy name → player_id resolution against the real player list) and
  `retrieve.query_player_signal()` (exact metadata-filtered lookup, never
  similarity ranking) — the same "structured lookup before semantic search"
  pattern already used for "my"-flavored questions, generalized to any named
  player. Eval harness now grades decision accuracy separately from
  retrieval accuracy (`evals/build_decision_questions.py` +
  `run_decision_eval.py`, dilemmas generated from `ground_truth.jsonl`, never
  hand-authored). Known gap: this session couldn't run the live Claude API or
  fetch the live Sleeper roster (no `ANTHROPIC_API_KEY` / Sleeper blocked in
  this sandbox) — validated everything else (signal computation, name
  resolution, structured retrieval, dilemma generation) against real 2024
  nflverse data instead. See TODO.md's Phase 3 section for full detail.
  A follow-up session fixed a real crash in `recommend()`: a question
  nothing could answer ("what's my team's record and who do i play this
  week?") burned through `max_turns` and raised an unhandled exception.
  Fixed both the immediate crash (`recommend()` now returns a graceful
  "not enough information" result instead of ever raising on
  non-convergence) and the actual gap (`get_team_record`/
  `get_current_matchup` tools, backed by new `src/rag/lookup.py`
  functions — Sleeper's own roster `settings` already carries
  wins/losses/ties, no new ingest needed).
- Phase 3.5 (multi-turn conversation + report generation): implemented,
  except trade suggestions (deliberately deferred). Multi-turn
  conversation — `recommend()` takes/returns an optional `messages`
  history, `src/reasoning/recommend.py --interactive` is a CLI REPL
  exercising it. Single-question callers (the eval harness) are
  unaffected since they never pass `messages`. Report generation is
  `src/reasoning/report.py`'s `generate_report()`, reusing
  `recommend.py`'s tools (`get_my_roster`/`get_player_signals`/
  `get_team_record`/`get_current_matchup`) rather than duplicating their
  logic; CLI entry point `recommend.py --report {start_sit,drop,
  waiver_pickups}`. Three of the spec's four report types are built —
  start/sit, drop, waiver pickups, all reasoning grounded in the actual
  numeric signals table (never a Claude API call; see `report.py`'s
  docstring for why). Trade suggestions is **not built**, deliberately —
  it needs signal work that doesn't exist yet (season-long player value,
  cross-roster positional need); PROJECT_SPEC.md is explicit that a
  trade report grounded in this-week's-matchup-scoped signals standing in
  for season-long value would be actively misleading. See TODO.md's
  Phase 3.5 section for full detail, including the specific real-2024-data
  validation run and two documented scoping simplifications (start_sit
  groups by position rather than a league's full Sleeper roster-slot
  structure; waiver_pickups' "rising" target share is a point-in-time
  value, not an actual week-over-week delta the project doesn't compute
  yet). **Known gap, flagged not built:** `src/scheduler/refresh.py`
  doesn't exist — every report and `recommend()` call is only as current
  as the last manual ingest/signals/embed run. See TODO.md's Phase 3.5
  section for detail; deliberately left for its own scoped session
  (scheduling/infra, not reasoning-layer work).
- Phase 3.6 (prior-season signal fallback): implemented. Discovered
  during Phase 3.5's real-data validation, not planned ahead of time:
  every signal is trailing/current-season by construction, so a
  not-yet-started season (confirmed live against the real, unstarted
  2026 season) means `generate_report()`/`recommend()` correctly return
  an empty/near-empty result — honest, but a bad first-use experience.
  Fix: when a player has no current-season signal at all, fall back to
  their most recent prior season's final numbers, explicitly labeled
  stale everywhere (`stale`/`source_season`/`source_as_of_week` on the
  raw row, every report entry, AND a `[STALE -- ...]` prefix on any
  prose) — never silently presented as current. Two separate, parallel
  implementations, matching the existing report/chat split:
  `src/rag/retrieve.py`'s `query_player_signal_with_fallback()` (Chroma,
  for `recommend.py`'s `get_player_signals` chat tool) and
  `src/reasoning/report.py`'s `_load_prior_season_fallback_table()` /
  `_signal_row()` (the raw parquet table, for report ranking).
  Fallback threshold is N=1 (any current-season signal at all, even one
  earlier week, wins over a stale fallback) — a larger N would need a
  new "distinct weeks active" field `matchup_signals.py` doesn't compute
  today, out of scope for this unit (the signal-*loading* layer, not
  signal computation). Validated against the real, newly-computed-this-
  session `data/processed/signals/signals_2025_week19.parquet` (full
  2025 regular season, computed live from real nflverse data) standing
  in for "last season's final numbers" against the real, actually-empty
  2026 season — confirmed real players (Saquon Barkley, Justin
  Jefferson) with real 2025 numbers surfacing correctly, explicitly
  labeled stale, in both `generate_report("drop", ...)` and
  `recommend.py`'s `get_player_signals` tool. See TODO.md's Phase 3.6
  section for full detail.
- Phase 3.7 (compound questions + structured data gaps, chat path only):
  implemented. Discovered during Phase 3.6's own real-model validation
  (`recommend.py` run for real against a real roster) — two gaps, both
  specific to the chat path (`report.py`'s reports already handle both
  correctly via `notes`). Gap 1: a compound question ("what's my weakest
  position, and who should I trade with to strengthen it") burned all
  tool-use turns and failed the *whole* question, even though the first
  half was fully answerable and only the trade-partner half was
  genuinely out of scope. Gap 2: a player with zero signal data at all
  (distinct from Phase 3.6's stale-but-present case) wasn't consistently
  flagged — confirmed live: asking about weak TE options surfaced two
  real rookies (Harold Fannin, Kenyon Sadiq) with generic "unproven
  rookie" reasoning instead of citing (or saying it couldn't cite) any
  actual signal. Fix: a new `data_gaps` field on `submit_recommendation`
  / `RecommendResult` (`reason: "no_signal_data"` or
  `"out_of_scope_capability"`, structured for a future Phase 5 UI to
  render distinctly, never a replacement for saying it in prose), a new
  explicit `has_signals: true/false` on `get_player_signals`'s output,
  and system-prompt instructions telling the model to (a) answer the
  answerable part of a compound question rather than failing it whole
  and (b) record a `no_signal_data` gap instead of filling in from its
  own background knowledge. CLI output (`--interactive` and
  single-question) prints `data_gaps` when non-empty. Validated
  `has_signals` against the real Fannin/Sadiq case: with the real,
  already-committed 2024+2025 signals tables properly re-embedded,
  Fannin correctly gets Phase 3.6's stale fallback (real 2025 numbers,
  `has_signals: true, stale: true`) and Sadiq correctly gets
  `has_signals: false` (genuinely zero 2025 involvement) — strongly
  suggesting the original live symptom was a stale local Chroma index
  (missing the 2025 chunks, a `src/scheduler/refresh.py`-shaped gap,
  deliberately not touched here) rather than a defect in the fallback
  logic itself; either way, `has_signals: false` is exactly the signal
  this fix now requires the model to act on explicitly. See TODO.md's
  Phase 3.7 section for full detail, including the live-model validation
  gap this sandbox still can't close (no `ANTHROPIC_API_KEY`).
- Phase 4 (coverage classification stretch): optional, not started
- Phase 5 (productization — final deliverable): not started

Always check TODO.md for the up-to-date task list within the active phase.

## Key architectural principle — do not violate
The signals table and RAG corpus are **league-agnostic** — computed
from NFL-wide sources (nflverse, NGS, odds), not tied to any one
league. Only roster ownership, matchup schedule, and scoring settings
are league-specific. When building Phase 2/3 code, do not hardcode
assumptions that only hold for Victorious Secret 3.0 (e.g. half-PPR
scoring) into the signals or RAG layers — those belong in the
per-league join at query time (`recommend.py`, later `src/api/`), not
in `matchup_signals.py` or `rag/`. This separation is what makes Phase
5 cheap; breaking it defeats the point.

## Non-negotiable methodology rules
- **As-of-date filtering is mandatory** in evals and in any signal
  computation used for backtesting — never let a signal see data from
  after the eval week's kickoff. This is the single easiest thing to
  get subtly wrong; check it explicitly whenever eval numbers look
  suspiciously good.
- **`ground_truth.jsonl` is never hand-authored** — always generated
  programmatically from nflverse box scores.
- **Score retrieval accuracy and decision accuracy separately** in
  evals. Don't collapse them into one number.
- Keep the three eval types distinct and don't let the qualitative
  seed set get quietly dropped just because the systematic set is
  easier to automate.

## Repo conventions
- Python, ChromaDB for the vector store.
- `pip install X --break-system-packages` if working in the sandboxed
  environment; normal installs if on Rohan's actual machine.
- Prefer explicit phase gates: don't start Phase 2 work until Phase 1
  validation checklist (see TODO.md / project spec) is actually run,
  not just assumed done.
- Rohan is relatively new to terminal/VS Code workflows — prefer
  concrete, copy-pasteable commands over abstract instructions, and
  confirm before running anything destructive (drops, force-pushes,
  overwrites of ground_truth.jsonl, etc.).

## When the project spec changes
If `ask-madden-project-spec.md` has been edited outside a Claude Code
session (e.g. planning done in Claude chat), don't assume this file or
TODO.md are already in sync with it. Diff intent against current repo
state and flag any mismatch before starting new work.
