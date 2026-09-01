# TODO

See `PROJECT_SPEC.md` for the full phased plan and signals table. Scope
now includes Phase 5 (productization: multi-league, hosted) — see that
section below and `CLAUDE.md`'s "Key architectural principle" before
writing any Phase 2 signals/RAG code, since Phase 5 depends on that code
staying league-agnostic.

Phase 1 validation for this session: full `pytest` suite (33/33, now 55/55
with Phase 2's new tests) passes. Live end-to-end validation against fresh
Sleeper API data could not be run in this sandbox -- its network policy
blocks `api.sleeper.app` (confirmed via the proxy status endpoint: a
policy denial, not a code error). nflverse's data host is reachable from
here and was used to validate Phase 2's signals against real 2024 season
data (see below). Re-run `python -m src.ingest.sleeper` /
`python -m src.rag.embed` / `python -m src.cli` on a machine with normal
network access to confirm the live Sleeper path end-to-end.

Phase 2 (signals layer) is now implemented -- see the checklist below.

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
is what keeps Phase 5 cheap.** Verified: none of the new modules below
take a league_id, roster, or scoring_settings parameter anywhere.
- [x] nflverse ingest: play-by-play, EPA/WPA (`src/ingest/nflverse.py`
      `fetch_pbp`/`save_pbp`). Personnel/formation columns come through
      as part of the raw pbp pull but have no computed signal yet —
      PROJECT_SPEC.md's signals table doesn't call for one either;
      revisit if a future signal needs it.
- [x] NGS ingest: aDOT, RYOE (`src/ingest/ngs.py`). CROE as literally
      named ("catch rate over expected") isn't a real published NGS
      stat — used `avg_yac_above_expectation` + `avg_separation` as the
      proxy instead and labeled it as such in code (see `ngs.py`'s
      docstring and `matchup_signals.py`'s `croe_proxy_*` fields).
- [x] Odds ingest (`src/ingest/odds.py`): **design deviation from the
      spec's "odds API (free tier)" wording** — derives game script /
      implied totals from nflverse's own `schedules` dataset
      (`spread_line`/`total_line`), which is free, has no API key or
      rate limit, and nflverse was already a dependency. No separate
      odds API integrated.
- [x] Compute matchup signals (`src/signals/matchup_signals.py`):
      defense run-funnel rate, red zone role share, recent efficiency
      trend, opponent-adjusted target share, game script/implied total,
      aDOT, RYOE, CROE-proxy — all as-of-week filtered (history strictly
      before the target week; see the module docstring). Matchup-fit
      score is still Phase 4 (needs Big Data Bowl coverage
      classification) — not attempted here. Validated against real 2024
      nflverse data (e.g. Saquon Barkley week-6 red zone share ≈48%,
      Justin Jefferson's bye week correctly nulls out opponent/implied
      total) — see git history for the ad hoc validation script.
- [x] Store signals alongside RAG corpus, retrievable by player/matchup
      (`src/rag/embed.py`: `build_signal_chunks`/`load_signal_chunks`,
      one chunk per player per as-of-week per the locked-in chunk
      granularity rule below; wired into `embed()`'s `main()` when
      `data/processed/signals/` has any computed tables).
- [x] realtime.py: injury/inactive (`fetch_injuries`/
      `current_injury_status`, via nflreadpy's injury report) and
      weather (`fetch_weather`, from schedules' temp/wind/roof). Line
      movement is **not implemented** — nflverse schedules only carry
      the closing line, not a time series; real movement tracking needs
      a live odds API with historical snapshots, a dependency this
      project doesn't have yet. Documented in `realtime.py`'s docstring
      rather than faked.

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
