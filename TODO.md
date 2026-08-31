# TODO

See `PROJECT_SPEC.md` for the full phased plan and signals table. Scope
now includes Phase 5 (productization: multi-league, hosted) — see that
section below and `CLAUDE.md`'s "Key architectural principle" before
writing any Phase 2 signals/RAG code, since Phase 5 depends on that code
staying league-agnostic.

We are still on Phase 1 validation — do not start Phase 2 or Phase 5
work until that's confirmed done.

## Phase 1: Foundation (RAG basics)
- [x] Sleeper ingest: league, rosters, matchups, player pool
- [x] Store as structured JSON (`data/raw/sleeper/`)
- [x] Chunk + embed into ChromaDB (`data/chroma/`)
- [x] CLI loop: question → retrieve → answer (`src/cli.py`)
- [x] Pull exact box scores via nflverse for chosen eval weeks (`src/ingest/nflverse.py`)
- [x] Auto-generate ground_truth.jsonl from nflverse weekly stats (`evals/build_ground_truth.py`)
- [x] Build evals/run_eval.py backtest harness (as-of-date filtering) — retrieval accuracy only; decision accuracy needs Phase 3's recommend.py
- [ ] **Deferred, tracked, not started:** qualitative eval seed set — hand-researched, verified (not invented) real pregame dilemmas (e.g. Week 5 2025 Dobbins/Harvey flex split, Addison vs. Jeudy), per `PROJECT_SPEC.md`'s eval methodology. `evals/build_eval_questions.py` currently produces neither this nor the spec's nflverse box-score "systematic set" — see its docstring. Needs actual research to source verified dilemmas; do not fabricate. Revisit once Phase 3's decision-accuracy grading exists to make these gradeable.

## Phase 2: Signals layer
**Architectural rule (see CLAUDE.md): signals and the RAG corpus must
stay league-agnostic — computed from NFL-wide sources, not tied to
Victorious Secret 3.0. Don't hardcode this league's scoring settings
(e.g. half-PPR) into `matchup_signals.py` or `rag/`; that join belongs
in `recommend.py` (Phase 3) / `src/api/` (Phase 5) at query time. This
is what keeps Phase 5 cheap.**
- [ ] nflverse ingest: play-by-play, EPA/WPA, personnel/formation
- [ ] NGS ingest: CROE, aDOT, RYOE
- [ ] Odds ingest: game script / implied totals
- [ ] Compute matchup signals
- [ ] Store signals alongside RAG corpus, retrievable by player/matchup
- [ ] realtime.py: injury/inactive, weather, line movement

## Phase 3: Reasoning layer
- [ ] recommend.py: retrieved facts + signals → Claude tool-use call → recommendation + explanation
- [ ] Expand eval set to grade recommendation quality, not just retrieval
- [ ] README write-up: architecture diagram, eval numbers, example Q&A

## Phase 4: Stretch (optional — not a blocker for Phase 5)
- [ ] Derived coverage classification (Big Data Bowl tracking data)
- [ ] Discord bot wrapper
- [ ] Weekly auto-generated lineup recommendations

## Phase 5: Productization (final deliverable)
Turns this from a single-league tool into a small real product: paste
in any Sleeper league ID, get the same signals-backed recommendations.
No password/OAuth, no payments — a portfolio deliverable, not a
business. See `PROJECT_SPEC.md`'s Phase 5 section for full detail and
success criteria. **Not started — do not begin until Phase 1 validation
and Phases 2-3 are actually done, not just assumed done.**
- [ ] Pull Sleeper scoring settings per league; parameterize signals/recommend accordingly
- [ ] Build storage layer: league_id/team_id → user mapping (SQLite)
- [ ] Build API layer (`src/api/`) wrapping recommend.py, scoped per league_id
- [ ] Build minimal web frontend (`web/`): register league → view roster → ask/recommend
- [ ] Add per-user/day query caps to bound Claude API spend
- [ ] Deploy to free-tier host (Railway/Render/Fly.io)
- [ ] Get 2-3 friends in different leagues to actually use it
- [ ] README: document the "started as one league, generalized to a product" story explicitly
