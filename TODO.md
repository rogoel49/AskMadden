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

Phase 2 (signals layer) is implemented -- see its checklist below.

Phase 3 validation for this session: full `pytest` suite is 95/95 (84
before this phase's new tests). Real-data validation was possible for
everything except the live Claude API call and the live Sleeper roster
pull -- same two environment blockers as Phase 1/2 (this sandbox's
network policy allows nflverse's data host but blocks `api.sleeper.app`,
and no `ANTHROPIC_API_KEY` is configured here). What *was* validated
against 100% real data: computed real 2024 signals for as-of-week 5
(417 players) and embedded them into a real local Chroma collection;
reproduced the exact reported bug live (asking `retrieve.query()` about
"Christian McCaffrey" returned Luke McCaffrey's chunk as the top hit)
and confirmed `get_player_signals` resolves both McCaffreys and Caleb
Williams correctly via the real `nflreadpy.load_players()` reference
list (1436 real skill-position players); pulled real, grounded signal
text for real players (Saquon Barkley, James Cook, Rhamondre
Stevenson) as a stand-in for "real start/sit questions" since the
actual Victorious Secret 3.0 roster couldn't be fetched here; and
generated real decision dilemmas from the real (committed)
`ground_truth.jsonl`. Re-run `python -m src.reasoning.recommend "..."`
and `python -m evals.run_decision_eval` on a machine with both
`ANTHROPIC_API_KEY` and live Sleeper access to validate the full
live agent loop against Victorious Secret 3.0's actual roster.

This session (crash fix + multi-turn conversation): full `pytest` suite
is 113/113 (104 before this session). Same two sandbox blockers as
every prior session -- no `ANTHROPIC_API_KEY`, Sleeper API blocked.
`get_team_record`/`get_current_matchup` needed no new real-data
validation (they're pure structured reads of already-ingested/already-
tested JSON shapes); the multi-turn conversation wiring was validated
mechanically with a scripted fake client (proves message-history
threading is correct) but not against the real model -- see Phase 3.5's
checklist below for the specific re-run command.

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
      **Known magic number to revisit:** `opponent_adjusted_target_share`'s
      0.1 reweighting constant (how much an opponent's pass-defense z-score
      moves a player's target share) is a deliberately simple, unfitted
      guess — not validated against outcomes. Revisit once Phase 3's
      decision-accuracy evals exist to check whether it's actually
      predictive; don't let it quietly calcify as load-bearing.
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
**Named-player bug fixed as part of this phase (was blocking correctness,
not deferred):** `retrieve.py`'s `query()` is pure semantic embedding
search, which can't reliably tell apart two real NFL players who share a
surname -- confirmed live: asking about "Christian McCaffrey" returned
his brother Luke's signal chunk as the top hit. Fixed by generalizing
`cli.py`'s existing "my"-question routing pattern (structured lookup
before semantic search) to any named-player mention:
`src/rag/player_index.py` resolves a name to a specific nflverse
`player_id` via exact-then-fuzzy structured matching against the real
player list (never embedding proximity), and
`retrieve.query_player_signal()` fetches that exact player's chunk via
a metadata filter, never a similarity ranking. `recommend.py`'s
`get_player_signals` tool always goes through this path for a named
player and never calls `search_league_info` (the semantic fallback) for
that. An ambiguous resolution (e.g. the bare surname "McCaffrey", or two
real players who share a full name) is returned as an explicit
candidate list, never silently guessed.
- [x] `src/rag/player_index.py`: `build_player_index()` (nflverse's full
      player reference table, filtered to modern gsis_id format,
      skill positions, and players active in the last two seasons --
      that recency filter matters, nflverse's `status` field alone
      doesn't reliably mean "currently rostered") and `resolve_player()`
      (exact full-name match, then *token-level* fuzzy matching --
      whole-string fuzzy ratio has a real length bias that reproduces
      the same bug it's meant to fix, caught by this phase's own tests;
      see the module for detail). `FUZZY_CUTOFF` tuned against real data
      to 0.8 after 0.75 let "McCaffrey" alone fuzzy-match an unrelated
      "Nate McCrary" -- a simple heuristic, not a fitted model, revisit
      if real usage surfaces more false positives/negatives.
- [x] `src/rag/retrieve.py`: `query_player_signal()`, an exact metadata
      filter (`type` + `player_id` [+ `season`/`week`]), never a
      similarity query.
- [x] `src/reasoning/recommend.py`: retrieved facts + signals → Claude
      tool-use agent → recommendation + explanation. Tools:
      `get_my_roster`/`find_owner` (structured Sleeper-space lookups,
      `src/rag/lookup.py`), `get_player_signals` (structured
      name-resolved nflverse-space lookup, the fix above),
      `search_league_info` (semantic fallback for non-player-named
      general league questions), and a terminal `submit_recommendation`
      tool the agent must call to conclude -- makes the result a
      parseable structure (`recommendation`/`reasoning`/`player_id`)
      instead of free text callers have to guess at. The model decides
      which tool(s) to call; no hardcoded routing like `cli.py`'s "my"
      string match. Per-league join (this league's roster + real
      `scoring_settings`) happens here, in the system prompt built from
      `data/raw/sleeper/league.json` -- verified `matchup_signals.py`
      and `rag/` still take no league_id/roster/scoring parameter
      anywhere. **Scope note:** `recommend()` operates on whatever
      league is already ingested (like `lookup.py`/`embed.py`) -- it
      does not itself take a `league_id`; multi-league parameterization
      stays Phase 5's job (`src/api/`), not pulled forward here.
      Injectable `client` param for testability without hitting the
      real API.
- [x] Expand eval harness to grade recommendation quality (decision
      accuracy), scored separately from retrieval accuracy per
      CLAUDE.md's non-negotiable rule -- `run_eval.py` is untouched and
      still reports `decision_accuracy: None`, since it doesn't measure
      it. `evals/build_decision_questions.py` generates PROJECT_SPEC.md's
      "systematic set": pairwise start/sit dilemmas from
      `ground_truth.jsonl` (never hand-authored -- every dilemma and its
      `expected_winner` trace back to a measured nflverse stat line),
      pairing same-week/same-position players who both cleared a
      `min_points` floor (symmetric in who wins, so selection can't bias
      the eval toward whichever side happened to score more), capped and
      shuffled with a fixed seed to bound eval cost (one live Claude call
      per dilemma). `evals/run_decision_eval.py` calls `recommend()` for
      each dilemma as-of-week-filtered to that dilemma's own week and
      checks whether the recommended `player_id` (falling back to a name
      check in the recommendation text) matches who actually scored more.
      Only ran against real ground truth's committed week 5, 2024 data in
      this sandbox -- see the validation note above.
- [ ] Once decision-accuracy evals have actually been run at volume,
      check whether `opponent_adjusted_target_share`'s 0.1 reweighting
      constant (`src/signals/matchup_signals.py`) is predictive; refit or
      drop it rather than leaving it as an unvalidated guess. Still not
      done -- this session generated the harness but couldn't run it live
      (no `ANTHROPIC_API_KEY` in this sandbox).
- [ ] Qualitative hand-curated dilemma seed set (deferred since Phase 1)
      is now gradeable (decision-accuracy grading exists) but still not
      built -- needs actual research to source verified real dilemmas, not
      fabricated ones. Still tracked, still not started.
- [ ] README write-up: architecture diagram, eval numbers, example Q&A

**Bug fixed this session (not deferred): `recommend()` crashed on a
question nothing could answer.** Repro: `python -m src.reasoning.recommend
"what's my team's record and who do i play this week?"` burned through
`max_turns` (nothing computed a win/loss record or resolved the current
opponent) and raised an unhandled `RuntimeError`, which crashed the CLI
process ungracefully on exit. Two real gaps, both closed:
- [x] `src/rag/lookup.py`: `team_record()`/`my_team_record()` (and
      `team_record_for_owner()` for asking about another team) --
      **no new ingest work needed**, Sleeper's own `/rosters` endpoint
      already computes wins/losses/ties and `teams.json` already
      carries it in each team's `settings` field; this just reads it
      structurally instead of leaving it uncomputed.
      `current_matchup()`/`my_current_matchup()`/
      `current_matchup_for_owner()` resolve this week's opponent from
      the already-ingested `matchups_week_{week}.json` -- returns
      `None` (never a guess) if that week hasn't been ingested locally.
      `recommend.py` gained matching `get_team_record`/
      `get_current_matchup` tools.
- [x] `recommend()` no longer raises when the agent can't converge
      within `max_turns` for *any* reason -- returns a
      `{"recommendation": "I don't have enough information to answer
      that.", "error": "max_turns_exceeded", ...}` result instead. This
      is the general safety net; the two new tools above are the actual
      fix for this specific question (a model with those tools should
      converge well before hitting the limit).
- [ ] Not done: `sleeper.py`'s ingest still only ever fetches one
      week's matchups per run (whichever week is current or requested).
      `team_record()` doesn't need this (Sleeper's roster `settings`
      already carries the season record), but a from-scratch matchup-
      history recomputation would. Not needed for this fix; flagging in
      case a future feature wants full weekly matchup history locally.

## Phase 3.5: Multi-turn conversation + report generation
See `PROJECT_SPEC.md`'s Phase 3.5 section for the full writeup
(sequencing rationale for the four report types, why report generation
is its own function rather than folded into the chat path).
- [x] Multi-turn conversation: `recommend()` takes an optional prior
      `messages` list and returns the updated history
      (`RecommendResult.messages`) so a clarifying question can be
      answered in the same conversation. The `submit_recommendation`
      tool_use is resolved with a synthetic `tool_result` before
      returning, so the returned history is always valid to continue
      from (the Messages API rejects a new user turn while a prior
      `tool_use` is unresolved). `evals/run_eval.py` and
      `evals/run_decision_eval.py` never pass `messages` -- confirmed
      by test that each of their calls is independent, with the fake
      client's own call log as proof (see
      `test_eval_style_calls_never_carry_state_between_independent_questions`
      in `tests/test_recommend.py`).
- [x] CLI REPL: `python -m src.reasoning.recommend --interactive`
      loops, threading `messages` through each turn. The single-question
      invocation (`recommend.py "question"`) is unchanged and still the
      default (`question` is now an optional positional arg, required
      unless `--interactive` is given).
- [ ] **Not built, scoped only:** `generate_report()` and its three
      buildable report types (start/sit, drop, waiver-wire pickups) --
      the task for this session was to add Phase 3.5 to the spec, not
      implement report generation. All three are real, buildable now
      with existing signals per the spec; none attempted yet.
- [ ] **Not built, deliberately deferred:** trade suggestions. Needs
      signal work that doesn't exist yet -- season-long/rest-of-season
      player value (every current signal is single-week-matchup-scoped)
      and positional-need assessment across rosters (nothing today
      compares roster composition between teams). Do not build the
      report before that signal work exists.
- [ ] Live validation gap (same two sandbox blockers as every phase so
      far): multi-turn conversation was validated mechanically with a
      scripted fake client in this sandbox (proves the message-history
      threading and control flow are correct -- see
      `test_recommend_second_call_sends_the_full_prior_history_to_the_client`)
      but never against the real model, since no `ANTHROPIC_API_KEY` is
      configured here. Re-run `python -m src.reasoning.recommend
      --interactive` for real, asking something ambiguous enough to
      trigger an actual clarifying question (e.g. "who should I start
      at flex" without naming anyone), to confirm the real model's own
      conversational behavior, not just the plumbing.

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
