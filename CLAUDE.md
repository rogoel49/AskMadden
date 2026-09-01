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
