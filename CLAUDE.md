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
- Phase 2 (signals layer): not started
- Phase 3 (reasoning/recommend.py): not started
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
