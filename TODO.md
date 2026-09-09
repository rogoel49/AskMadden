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

This session (report generation): full `pytest` suite is 123/123 (113
before this session). Built `src/reasoning/report.py`'s `generate_report()`
and all three buildable report types (start_sit, drop, waiver_pickups) --
trade suggestions deliberately not attempted, see below. Real-data
validation: ran all three report types in this sandbox against the real,
committed `data/processed/signals/signals_2024_week5.parquet` and the
real live nflverse player reference table (`nflreadpy` -- reachable here,
same as every prior phase), with a stand-in league roster built from real
2024 players who actually appear in that signals table (Saquon Barkley,
Rhamondre Stevenson, Justin Jefferson, George Kittle, Josh Allen, Jeff
Wilson on "my" team; James Cook and Ja'Marr Chase on a second team, to
prove waiver_pickups excludes rostered players league-wide, not just
"my" roster). Confirmed real, grounded output: start_sit correctly
recommended Stevenson over Barkley/Wilson at RB citing real red-zone-share
numbers (57%/48%/2%); drop correctly ranked Jeff Wilson weakest (real
-0.34 EPA/play trend, real 2% target share) ahead of Josh Allen and
Justin Jefferson; waiver_pickups correctly excluded all 8 real rostered
player_ids and ranked real unrostered players (Kylen Granson, Christian
Watson, Kyren Williams, etc.) by real computed signals, out of 397
unrostered signal-bearing candidates in the 1436-player pool. Same two
sandbox blockers as every prior phase (no `ANTHROPIC_API_KEY`, live
Sleeper API blocked) did NOT block this validation, because
`generate_report()` deliberately never calls the Claude API (see
`report.py`'s module docstring for why) and the stand-in roster stands in
only for the real Sleeper roster pull, not for the signals data itself
(which is real and already committed). Re-run
`python -m src.reasoning.recommend --report start_sit` (etc.) against the
real Victorious Secret 3.0 roster on a machine with Sleeper access to
confirm against the actual league, not the stand-in.
- [x] `src/reasoning/report.py`: `generate_report(report_type, raw_dir,
      persist_dir, season, as_of_week, ...)`. Reuses `recommend.py`'s
      `dispatch_tool()` for `get_my_roster`/`get_player_signals`/
      `get_team_record`/`get_current_matchup` (identity resolution and
      roster/record/matchup lookups -- never reimplemented), plus a new
      `src/rag/lookup.py:all_rostered_players()` (every team's roster
      league-wide, the same "structured read over the raw Sleeper pull"
      pattern the rest of that file already uses) for waiver_pickups'
      set difference. Deliberately does NOT call the Claude API --
      ranking a bounded set of candidates by already-computed numbers has
      one auditable answer, computing it directly in code is more
      reliable than asking a model to eyeball numbers, and it's the only
      way this session's real-data validation requirement could be met
      at all without `ANTHROPIC_API_KEY`. Also deliberately does NOT go
      through `get_player_signals`'s chunk *text* for ranking math (that
      text has no machine-readable numeric fields -- see `report.py`'s
      docstring) -- reads the same underlying signals parquet
      `matchup_signals.py` produces directly via polars instead.
  - [x] start_sit: groups a roster by position (QB/RB/WR/TE -- the same
        skill positions signals exist for) and, for every group with 2+
        signal-bearing players, recommends a starter with alternatives
        and reasoning citing the specific signal values compared.
        **Documented simplification:** groups by position, not by a
        league's actual Sleeper `roster_positions` slot structure
        (FLEX/superflex/bench counts) -- a full slot-by-slot lineup
        optimizer is materially more scope than demonstrating "recommend
        who to start when there's more than one viable option" needs.
  - [x] drop: ranks roster contributors by the same composite signal
        score (recent efficiency trend, red zone role share, target
        share -- deliberately simple, unfitted weights, same
        documented-not-fitted status as `matchup_signals.py`'s own 0.1
        reweighting constant) and returns the weakest, each with concrete
        threshold-based reasons (e.g. "efficiency trending down", "low
        target share", "minimal season-long usage"), never just a bare
        score.
  - [x] waiver_pickups: full skill-position NFL player pool
        (`player_index.build_player_index()`, the same source
        `get_player_signals` already uses) minus every roster league-wide
        (`lookup.all_rostered_players()`, a set difference over
        already-ingested data, no new ingest), ranked by the same
        opportunity-score composite. **Documented simplification:**
        "rising" target share is approximated by the current
        (opponent-adjusted, where available) target share value, not an
        actual week-over-week delta -- the signals table is one
        point-in-time row per player/week, not a rolling series, so a
        real trend isn't computed yet.
  - [x] CLI: `python -m src.reasoning.recommend --report
        {start_sit,drop,waiver_pickups}` -- separate from `--interactive`
        and the single-question mode, prints the structured report as
        JSON.
- [ ] **Not built, deliberately deferred (unchanged from before this
      session):** trade suggestions. Still needs signal work that doesn't
      exist yet -- season-long/rest-of-season player value and
      positional-need assessment across rosters. Did not attempt even a
      rough version -- PROJECT_SPEC.md is explicit that a trade report
      grounded in this-week's-matchup-scoped signals standing in for
      season-long value would be actively misleading.
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
- [ ] **Known gap, flagged not built:** `src/scheduler/refresh.py` doesn't
      exist yet -- `src/scheduler/` is an empty `__init__.py` only. Every
      report from `generate_report()` and every `recommend()` call is
      only as current as whenever someone last manually ran
      `matchup_signals.py` + `embed.py` by hand (and `sleeper.py` for the
      roster side). This is fine for this session's real-data validation
      (a fixed, already-computed week-5 signals table is exactly what's
      being validated against) but is a real gap before any of this is
      useful against a live, in-progress season -- signals would go
      stale the moment a week passes without someone remembering to
      re-run the pipeline. Deliberately NOT attempted this session:
      building a scheduler is an infra/scheduling problem, a different
      kind of work from this session's reasoning-layer scope, and
      deserves its own scoped session (deciding a cadence, an
      idempotent/incremental refresh strategy for `embed.py`'s current
      full-rebuild-on-every-run design, and where it runs in Phase 5's
      hosted deployment) rather than being bolted onto this one.

## Phase 3.6: Prior-season signal fallback
See `PROJECT_SPEC.md`'s Phase 3.6 section for the full writeup. Not
planned ahead of time -- discovered live during Phase 3.5's real-data
validation: every signal this project computes (EPA trend, red zone
share, target share, ...) is trailing/current-season by construction, so
running a report or `recommend()` against a season with zero games played
yet (confirmed live against the real, unstarted 2026 season --
`nfl_state.json`'s real `season_start_date` is 2026-09-09, and
`nflreadpy` has no 2026 play-by-play/NGS published yet, so
`stats_player_week_2026.parquet` 404s and the loaders hard-reject seasons
past 2025) comes back with an empty/near-empty result and an honest "no
computed signals" note. Correct given a genuinely empty table, but a bad
first-use experience for a friend setting a Week 1 lineup right when a
first impression matters most.

This session: full `pytest` suite is 133/133 (123 before this session; 10
new tests: 4 in `tests/test_retrieve_player_signal.py`, 4 in
`tests/test_report.py`, 2 in `tests/test_recommend.py`). Real-data
validation: computed a real, full 2025-regular-season signals table live
this session (`python -m src.signals.matchup_signals --season 2025
--as-of-week 19`, real nflverse data -- reachable in this sandbox, same as
every prior phase) and committed it as
`data/processed/signals/signals_2025_week19.parquet` (33KB, same "small
computed sample, can commit" convention as the existing
`signals_2024_week5.parquet`). Ran both the report path and the chat-tool
path against the real, actually-empty 2026 season with this real 2025
table as the only thing on record to fall back to: `generate_report("drop",
season=2026, as_of_week=1, ...)` correctly returned real players (Saquon
Barkley, Justin Jefferson) with their real 2025 season-end numbers (e.g.
Barkley's real +0.17 EPA/play trend, 36% red zone share), every one
explicitly marked `"stale": true, "source_season": 2025` and prefixed
`[STALE -- ...]` in `signals_summary`/`weakness_reasons`/the report-level
`notes`; `recommend.py`'s `get_player_signals` tool for "Saquon Barkley"
under the same 2026/week-1 context returned the same real 2025 numbers via
the Chroma-backed path, same explicit stale labeling. Confirmed the
opposite too: a player with real current-season data present never picks
up the stale fallback even when prior-season data also exists on record
(`test_drop_never_falls_back_when_current_season_data_exists`), and a
player with neither current- nor prior-season data anywhere degrades to
"excluded, noted" rather than crashing
(`test_drop_handles_a_player_with_no_signal_data_at_all_gracefully`).
- [x] `src/rag/retrieve.py`: `query_player_signal_with_fallback()` --
      exact current-week match first; if none, the most recent same-
      season chunk at or before as_of_week (never a later week, so this
      never leaks future data into an as-of-week-filtered lookup -- a
      real latent bug caught by
      `test_fallback_never_leaks_a_later_current_season_week` during this
      session's own testing, fixed by adding `query_player_signal()`'s
      new `max_week`/`max_season` bounded-search parameters); if still
      none, the most recent chunk from strictly before the requested
      season, explicitly flagged `stale`. `recommend.py`'s
      `_tool_get_player_signals` uses this instead of the old exact-only
      `query_player_signal()`, adds `stale`/`source_season`/
      `source_as_of_week` to its returned dict, and prefixes the
      `signals` text itself with `[STALE -- ...]` so the model can't miss
      it even without reading the structured fields. The system prompt
      (`_build_system_prompt`) also now tells the model explicitly to
      name the season when `stale: true` rather than presenting the
      numbers as current.
- [x] `src/reasoning/report.py`: `_load_signals_table()` now unions every
      locally-computed `signals_{season}_week*.parquet` file with week <=
      as_of_week (not just the exact-week file -- a player missing from
      this week's file but present in an earlier one still counts as
      having current-season data, mirroring the chat path's same-season
      check); `_load_prior_season_fallback_table()` loads the highest
      available `season - 1` file; `_signal_row()` combines the two with
      the same never-silent stale labeling. Every report entry across all
      three types (start_sit's `recommended_starter`/
      `alternatives_considered`, drop's entries, waiver_pickups' entries)
      now carries explicit `stale`/`source_season`/`source_as_of_week`
      fields, not just prose -- plus a per-report `notes` summary listing
      which players fell back.
- [x] Fallback threshold documented as N=1 (any current-season signal at
      all beats a stale fallback, even one week's worth) in both
      `report.py`'s and `retrieve.py`'s docstrings, with the specific
      reason a larger N isn't implemented (would need a new "distinct
      weeks active this season" field in `matchup_signals.py`'s output --
      signals-*computation* work, out of scope for this unit, which stays
      confined to the signal-*loading* layer per the task's own scoping).
- [x] Test coverage for the three required cases: no data at all (current
      or prior) degrades to excluded/None, never a crash; only
      prior-season data gets the stale-labeled fallback; current-season
      data present never sees the stale fallback even when prior-season
      data also exists.
- [ ] Not touched, deliberately (explicitly out of scope for this unit):
      trade suggestions (still Phase 3.5's own deferred item), the
      `src/scheduler/refresh.py` gap below (still not built), and
      `report.py`'s existing signal-weight constants (`_EPA_TREND_WEIGHT`
      etc. -- unchanged).
- [x] Live validation gap -- **closed**: this sandbox still has no
      `ANTHROPIC_API_KEY`, but Rohan ran this for real on his own machine
      against the real Sleeper roster and real 2025 signals data and
      confirmed it end-to-end, including the model correctly
      self-identifying stale data and naming the source season without
      being asked. Real validation surfaced two further real gaps in the
      chat path, though -- see Phase 3.7 below.

## Phase 3.7: Compound questions + structured data gaps in the chat path
See `PROJECT_SPEC.md`'s Phase 3.7 section for the full writeup. Not
planned ahead of time -- discovered during Rohan's own real-model
validation of Phase 3.6 (`recommend.py` run for real, `ANTHROPIC_API_KEY`
configured locally, against the real Sleeper roster). Two real gaps, both
in `recommend.py`'s chat path specifically -- `report.py`'s reports
already handle both correctly (a zero-signal player is excluded from
ranking and named in `notes`), so this is scoped as chat-path-only, not a
`report.py` change.

This session: full `pytest` suite is 136/136 (133 before this session; 3
new tests in `tests/test_recommend.py`). No new real-data ingest needed --
this is chat-orchestration/prompt work, not a new signal. Real-data
validation for `has_signals` specifically: rebuilt a real local Chroma
collection from the real, already-committed 2024+2025 signals tables and
ran `get_player_signals` for the real two rookies from Rohan's own report
(Harold Fannin, gsis `00-0040663`; Kenyon Sadiq, gsis `00-0041032`) under
season=2026/week=1 (the real, actually-empty current season). Confirmed:
Fannin actually HAS real 2025 season data (115 real plays) and correctly
gets Phase 3.6's stale fallback (`has_signals: true, stale: true,
source_season: 2025`, real cited numbers); Sadiq genuinely has zero 2025
pbp involvement and correctly gets `has_signals: false`. This strongly
suggests Rohan's original live symptom (both players getting generic,
uncited reasoning) was a stale local Chroma index at the time he ran it
(missing the 2025 chunks -- a `src/scheduler/refresh.py`-shaped gap,
deliberately not touched in this unit) rather than a defect in the
fallback logic itself. Either way, `has_signals: false` is exactly the
structured signal this fix now requires the model to act on explicitly
instead of silently reasoning from its own background knowledge --
correct behavior for both "genuinely no data" and "stale index" causes.
- [x] Gap 1 fix: system prompt (`_build_system_prompt`) now explicitly
      instructs the model to recognize a compound question with an
      answerable part and an out-of-scope part, answer the answerable
      part, and record the unanswerable part as a `data_gaps` entry
      (`reason: "out_of_scope_capability"`) instead of letting it fail
      the whole question. This is prompt-only -- no new tool, no new
      signal; the underlying capability gap (season-long value,
      cross-roster positional need) is still Phase 3.5's own tracked,
      deferred trade-suggestions item, unchanged.
- [x] Gap 2 fix: `_tool_get_player_signals` (`src/reasoning/recommend.py`)
      now returns an explicit `has_signals: true`/`false` field --
      `false` when nothing at all has been computed for a player (not
      even Phase 3.6's stale fallback), distinct from `stale: true`
      (data exists, just old). The system prompt tells the model that
      `has_signals: false` means "record a `no_signal_data` data_gaps
      entry and say so in your reasoning," not "fill the gap with your
      own knowledge of the player."
- [x] `submit_recommendation`'s tool schema gains a `data_gaps` array
      (`player_name?`, `reason` enum
      [`no_signal_data`,`out_of_scope_capability`], `detail`).
      `RecommendResult` gains a matching `data_gaps: list[dict]` field
      (default `[]`, via `field(default_factory=list)`) and `to_dict()`
      includes it. Decided and documented empty-case shape: always a
      present list, `[]` when nothing's wrong -- never omitted, never
      `None` -- matching `tool_calls`/`messages`' existing always-a-list
      convention, so a caller never needs a None-guard and a future
      Phase 5 UI can render it directly.
- [x] CLI: `_print_result()` (shared by both `--interactive` and the
      single-question path, so both pick this up with one change) prints
      a "Data gaps:" section listing each entry's reason/player/detail
      when `data_gaps` is non-empty; prints nothing extra otherwise.
- [x] Test coverage (`tests/test_recommend.py`, scripted fake-client
      pattern, same as the rest of the file):
      `test_recommend_answers_the_answerable_half_of_a_compound_question`
      (the weakest-position half comes back as a real recommendation, the
      trade half becomes one `out_of_scope_capability` data_gaps entry,
      `error` stays `None` -- never `max_turns_exceeded`),
      `test_recommend_surfaces_no_signal_data_gap_instead_of_generic_reasoning`
      (a `has_signals: false` player produces a matching `no_signal_data`
      data_gaps entry), and an added assertion on the existing
      fully-grounded-answer test confirming `data_gaps == []` there --
      no regression, a normal answer isn't cluttered with an empty-but-
      present field turning into noise. Also two `dispatch_tool`-level
      tests confirming `has_signals` is `True`/`False` in the right
      cases.
- [ ] Not touched, deliberately (explicitly out of scope for this unit):
      trade suggestions themselves, `src/scheduler/refresh.py`, and any
      change to `report.py` -- including the open question (documented,
      not resolved) of whether `report.py`'s existing `notes`-string
      approach should eventually migrate to this same structured
      `data_gaps` shape for consistency.
- [x] Live validation, part 1 (Gap 2) -- **closed**: Rohan ran this for
      real against a real roster and confirmed Gap 2's fix works
      correctly -- `Jeremiyah Love` and `Kenyon Sadiq` both correctly
      flagged `has_signals: false`, no fabricated reasoning about them.
- [ ] Live validation, part 2 (Gap 1) -- **found a deeper problem,
      addressed below, still needs re-validation**: asking the real
      compound trade question ("what's my weakest position, and who
      should I trade with in the league to strengthen it") got back
      `data_gaps: []` but `reasoning` full of specific, fabricated trade
      strategy ("package one QB (Herbert) or a mid-tier RB... to upgrade
      at TE") with zero supporting tool calls -- every call in
      `tool_calls` was `get_my_roster`/`get_player_signals` on the user's
      own roster, no `find_owner`, no cross-team lookup anywhere. Worse
      than the original bug: the model found a way to *sound* responsive
      to the out-of-scope half without actually engaging it, which slid
      past the original "record a gap for the part you couldn't do"
      instruction -- offering plausible asset-packaging advice apparently
      counted, to the model, as "doing" the trade-partner half. See the
      addendum below for the fix; re-run the exact same question once
      `ANTHROPIC_API_KEY` is available to confirm it actually holds this
      time (this sandbox still can't -- same blocker as every phase).

### Addendum: strengthen the anti-fabrication rule + explicit capability boundary
Found in this phase's own real-model validation of Gap 1 (above), not a
new phase -- Phase 3.7 itself wasn't actually done until this held up
under a real compound question, so this is a correction to the same unit
of work, not a follow-up phase.
- [x] System prompt (`_build_system_prompt`): new explicit, stricter rule
      -- any specific claim in `recommendation`/`reasoning` (a trade to
      make, assets to package, a player to target from another team,
      anything about another team's roster/needs/value) must be backed
      by an actual tool call made that turn; if none exists, say so
      explicitly and record an `out_of_scope_capability` data_gaps entry
      instead of improvising. Stricter than the original "record a gap"
      instruction, which the model found a loophole in.
- [x] Explicit capability boundary, per the user's suggestion this needed
      more than prompt wording alone: `find_owner`'s own tool description
      and the system prompt both now state plainly that `find_owner` only
      answers "who owns this one named player" and is NOT a
      roster-comparison or trade-fit tool, and that no tool here inspects
      another team's full roster or compares needs/value across teams --
      so the model has an explicit boundary instead of inferring one.
      `submit_recommendation`'s `data_gaps` schema description reinforces
      the same rule (defense in depth -- schema descriptions are also
      part of what the model reads).
- [x] Test coverage, honestly scoped (see
      `tests/test_recommend.py`'s new tests' own docstrings for the full
      caveat): `test_recommend_does_not_reject_ungrounded_trade_advice_at_the_code_level`
      documents, explicitly, that `recommend()` has NO code-level guard
      against a model fabricating ungrounded advice -- it's a scripted
      fake-client response proving the orchestration loop passes through
      whatever the model says, not a safeguard. This is a genuine
      limitation, not something a fake client can meaningfully fix: the
      actual failure is model behavior (choosing to fabricate plausible
      prose instead of admitting a gap), which no code branch can
      exercise. `test_system_prompt_explicitly_bars_ungrounded_trade_advice`
      is a regression guard confirming the new rule's specific language
      is present in the prompt -- it protects against a future edit
      silently weakening or deleting the rule, and nothing more; it does
      not prove a real model follows it.
- [x] Full `pytest` suite: 138/138 (136 before this addendum; 2 new
      tests, both in `tests/test_recommend.py`).
- [x] Real-model re-validation -- **closed**: Rohan ran the real compound
      trade question against the real model and confirmed the
      strengthened rule holds -- no more fabricated trade strategy. PR
      #15 merged.
- [ ] Not touched, still out of scope: trade suggestions themselves,
      `src/scheduler/refresh.py`, `report.py`.

## Phase 3.8: Roster-composition visibility across the league
See `PROJECT_SPEC.md`'s Phase 3.8 section for the full writeup. Closes a
specific, closeable gap Phase 3.7 left behind: "who should I trade with"
correctly declined to fabricate an answer and correctly explained why --
but "no tool inspects another team's roster" was a real, closeable gap,
not a permanent one. The data already exists and is already ingested
(Sleeper's roster endpoint returns every team's roster;
`lookup.py`'s `all_rostered_players()` already proved the data is
accessible league-wide) -- nothing exposed per-team roster *composition*
to the reasoning agent. This phase closes exactly that, and only that --
composition, never valuation (deferred to Phase 6, after Phase 5 -- see
that section for why).

This session: full `pytest` suite is 145/145 (138 before this session; 7
new tests: 3 in `tests/test_lookup.py`, 3 dispatch-level +
1 orchestration in `tests/test_recommend.py`).
- [x] `src/rag/lookup.py`: `all_team_rosters()` (every team's roster,
      grouped by position, with a per-position count -- the league-wide,
      per-team-grouped counterpart to `my_players()`) and
      `team_roster_for_owner()` (the same, filtered to one team, same
      case-insensitive-exact-match convention as
      `team_record_for_owner()`/`current_matchup_for_owner()`).
- [x] New `get_league_rosters` tool in `src/reasoning/recommend.py`,
      wired into `TOOLS`/`_DISPATCH` the same way every other tool is.
      Takes an optional `owner_display_name`: omit for every team in the
      league at once (the useful shape for surveying the whole league for
      surplus/need at a position), give one for a single team. Unknown
      owner reports a structured error, never a guess.
- [x] System prompt updated carefully, not just additively -- the
      compound-question paragraph now says the trade-partner half is
      *partially* answerable (composition, via `get_league_rosters`)
      while the valuation half still isn't, and the anti-fabrication
      paragraph now explicitly permits a composition claim backed by a
      real `get_league_rosters` call while still blocking any claim about
      trade value, fairness, or what to offer. Rewriting rather than
      appending mattered here: the old wording ("no tool here inspects
      another team's roster") would have flatly contradicted the new
      tool's existence if left as-is.
- [x] `find_owner`'s own tool description corrected for the same reason --
      it previously said outright that no tool inspects another team's
      full roster, which was true when written and is not true anymore.
      Left uncorrected, the model would be told something false about its
      own capabilities.
- [x] Test coverage: `all_team_rosters()`/`team_roster_for_owner()`
      (found, and unknown-owner-returns-None) in `tests/test_lookup.py`;
      `get_league_rosters` dispatch (all teams, one named team, unknown
      owner reports a structured error) and one orchestration test in
      `tests/test_recommend.py`
      (`test_recommend_uses_get_league_rosters_for_composition_but_still_declines_valuation`)
      -- framed honestly in its own docstring as orchestration-wiring
      proof (recommend() correctly threads a scripted ideal response
      through), not proof a real model chooses to call the new tool or
      phrases the caveat this way.
- [ ] Live validation gap (same sandbox blocker as every phase so far):
      no `ANTHROPIC_API_KEY` here, and no real Sleeper league data either
      (both blocked in this sandbox, same as every prior phase) -- the
      new tool and prompt wording were only validated mechanically. Real
      Sleeper roster/team data isn't needed to have proven the code
      logic correct (that's what `tests/test_lookup.py`'s fixtures did),
      but only a real model, asked the real question ("what's my weakest
      position, and which teams in my league might be willing to trade
      at that position?"), can confirm it actually calls
      `get_league_rosters`, cites real teams/rosters, and still declines
      trade fairness/specific offers -- still outstanding.
- [ ] Not touched, deliberately out of scope: trade valuation itself
      (deferred to Phase 6), `src/scheduler/refresh.py`, `report.py`.

## Fixed: recommend()/generate_report() silently depended on the CLI's main() to load .env
Not part of any phase's planned scope -- a real gap found by investigation
(`python -c "from src.reasoning.recommend import recommend;
recommend('...')"` failed with an Anthropic auth error even though
`python -m src.reasoning.recommend "..."` worked with the same `.env`
file). Confirmed, not assumed: `load_dotenv()` was only ever called
inside each module's own CLI `main()` (`src/reasoning/recommend.py`,
`src/reasoning/report.py`) -- `recommend()` and `generate_report()`
themselves never called it, so calling either directly (skipping the CLI
entry point entirely) never populated `ANTHROPIC_API_KEY`/`MY_ROSTER_ID`
from `.env` at all. Reproduced the exact mechanism directly: with
`ANTHROPIC_API_KEY` absent from the environment, `anthropic.Anthropic()`
constructs without error, but the first real `.messages.create()` call
fails with `TypeError: Could not resolve authentication method...` --
exactly the reported "Anthropic auth error" symptom, and it happens
before any network call.

This matters beyond a one-off `-c` repro: Phase 5.2's API layer will
import and call `recommend()`/`generate_report()` directly, the same way
the failing one-liner did -- left unfixed, the same failure would have
shown up in Phase 5.2 in production.
- [x] `recommend()` and `generate_report()` now call `load_dotenv()`
      themselves, as their own first statement -- `python-dotenv` was
      already an established project dependency (`requirements.txt`,
      already used the same way in `cli.py`/`sleeper.py`/both modules'
      own `main()`), so this matches the existing pattern rather than
      introducing a new one. `load_dotenv()`'s default behavior (never
      overrides a variable already present in the environment) is
      exactly the "production-correct" semantics needed: real env vars
      (Phase 5.2's actual deployment target) always win, and it's a
      cheap no-op when no `.env` file exists at all (the normal
      production case) -- so this doesn't just paper over local dev, it
      is correct in both cases.
- [x] Removed the now-redundant `load_dotenv()` calls from both modules'
      `main()` (every path through `main()` reaches `recommend()` or
      `generate_report()`, which now load it themselves) rather than
      leaving a duplicate call sitting in two places.
- [x] Test coverage: `tests/test_recommend.py`'s
      `test_recommend_loads_dotenv_itself_not_only_via_cli_main` (spies on
      `load_dotenv` to confirm `recommend()` actually calls it -- verified
      to fail against the pre-fix code) and
      `test_recommend_works_when_called_directly_with_env_already_set`
      (the production-correct path: real env vars already set, no `.env`
      file involved, called directly rather than via the CLI);
      `tests/test_report.py`'s equivalent
      `test_generate_report_loads_dotenv_itself_not_only_via_cli_main`.
      Full `pytest` suite: 148/148 (145 before this fix; 3 new tests).

## Phase 4: Stretch (optional — not a blocker for Phase 5)
- [ ] Derived coverage classification (Big Data Bowl tracking data)
- [ ] Discord bot wrapper
- [ ] Weekly auto-generated lineup recommendations

## Phase 5: Productization (final deliverable)

Turns this from a single-league tool into a small real product: anyone
can connect their own Sleeper account, pick a league, and get the same
signals-backed recommendations. No password/OAuth, no payments — a
portfolio deliverable, not a business. See PROJECT_SPEC.md's Phase 5
section for full detail, rationale, and success criteria.

5.1, 5.2 and 5.3 are implemented (see their sections below); 5.4-5.6
are not started.
Phase 4 is explicitly optional and not a blocker (see
above) — the actual gate was Phases 1-3.8, which are done. Phase 3.7's
anti-fabrication addendum and Phase 3.8's roster-composition tool
(get_league_rosters) are both real-model validated and closed as of
this update — see their respective sections above. Phase 6 (the
points-based trade-value proxy) is deferred past Phase 5 by design, not
a prerequisite — see Phase 6's own section for the rationale.

A UI design for this phase already exists, built outside a Claude Code
session: design/askmadden-ui-mockup.html — a static (no real data)
HTML/CSS/JS prototype with three top-level views: a landing page (hero
pitch, feature cards, an eval-numbers band whose values are explicitly
labeled placeholders), the login/league-picker flow, and one
responsive app shell for Feed/Chat/Roster/Moves — a phone frame below
900px viewport width, a sidebar-nav desktop layout above it, same DOM,
CSS media queries only, not two separate builds. Phase 5.3 below wires
it to real data; it is not a from-scratch design task, and it is not a
responsive-layout task either — the breakpoint CSS is already in the
file.

Sequencing constraint, same shape as every prior phase's dependency
chain: 5.1 and 5.2 (backend) must exist before 5.3 (frontend) is
anything but a static demo calling nothing real.

### 5.1 — Scoring + league parameterization
Implemented. Full `pytest` suite is 176/176 (148 before this phase's
28 new tests, `tests/test_league.py`). Same two sandbox blockers as
every prior session -- no `ANTHROPIC_API_KEY`, `api.sleeper.app` blocked
by the sandbox's network policy (confirmed again via the proxy status
endpoint) -- so see "Validation" below for exactly what was and wasn't
run against real data.

**What the investigation actually found (the useful part).** The
Phase 5 plan text (this file and PROJECT_SPEC.md) said
`matchup_signals.py` and `recommend.py` "currently assume half-PPR".
Traced before changing anything, that turned out to be a **single-
league blind spot, not a magic number** -- there was no hardcoded 0.5
(or any other scoring weight) anywhere to replace:
- `recommend.py` has read the league's real Sleeper `scoring_settings`
  from the ingested `league.json` since Phase 3 and put them in the
  system prompt verbatim (`rec=0.5, pass_td=4, ...`). That's its ONLY
  use of scoring: a text summary the model is told to let "inform which
  stats matter". It was correct for Victorious Secret 3.0 only because
  the one ingested league happened to be half-PPR -- the code already
  did the right thing, it had just never been shown a second league.
- `report.py` never read scoring settings at all. Its ranking
  (`_opportunity_score`) is a weighted sum of `epa_trend` /
  `red_zone_share` / `target_share` -- usage/efficiency signals that
  are the same number in any scoring format. So reports have no
  scoring-dependent output today, full stop: two leagues with identical
  rosters and different scoring get identical reports. Now pinned by a
  test (`test_generate_report_ranking_is_scoring_format_independent_today`)
  so that if a scoring-aware ranking is ever added (Phase 6's
  points-based proxy is the natural place), the test fails and gets
  rewritten deliberately rather than the property changing silently.
- `matchup_signals.py` (and everything under `src/rag/`) has no
  scoring concept whatsoever -- verified by grep and now by a guard test
  (`test_league_module_is_not_imported_by_signals_or_rag`). The spec
  sentence naming it was simply wrong; corrected in PROJECT_SPEC.md.
- `evals/build_ground_truth.py` is the one place scoring settings are
  applied *numerically* (its `STAT_TO_SCORING_KEY` mapping), and it
  reads them from the same `data/raw/sleeper/league.json`, same
  `data.scoring_settings` key, as `recommend.py` -- consistent, just
  path-keyed rather than league-keyed, and (like everything else) with
  no record of *which* league a row was scored under.
- "Which league" was genuinely implicit everywhere: `recommend()` /
  `generate_report()` took `raw_dir`/`persist_dir` and trusted whatever
  `src.ingest.sleeper` last wrote there. No `league_id` was read, passed,
  or checked anywhere in the reasoning layer -- Sleeper's `league.json`
  carries one, nothing looked at it. Ingest league B, ask about league
  A, get league B's roster and scoring with no error.

**What changed.**
- [x] New `src/reasoning/league.py` (the per-league join layer, not
      `rag/`): `load_league(league_id, raw_dir, persist_dir)` reads the
      ingested `league.json`, returns a `LeagueConfig` carrying the
      league's real `scoring_settings` verbatim, and raises
      `LeagueMismatchError` (naming both IDs and a copy-pasteable
      re-ingest command) if the directory holds a *different* league or
      a `league.json` with no `league_id` at all. The point: a
      wrong-league answer is now a loud error, never a quiet one.
- [x] `recommend()` and `generate_report()` accept `league_id` as a
      required second positional parameter and go through
      `load_league()` -- `recommend()` refuses before spending a model
      call. `raw_dir`/`persist_dir` still say *where* that league's data
      lives (the existing flat `data/raw/sleeper/` / `data/chroma/`
      convention, unchanged so Rohan's single-league setup keeps working);
      per-league storage layout is 5.2's job, not this parameter's.
      Every result now echoes the league: `league_id` on
      `recommend()`'s dict, `league_id` + `league_name` on every report
      header (additive keys only -- see the regression note).
- [x] Scoring settings pulled from that league's real Sleeper data --
      which, per above, they already were; the real change is that the
      read now happens in one place (`LeagueConfig.scoring_settings`)
      keyed by a verified `league_id`, and `build_ground_truth.py`'s
      loader routes through the same `league.py` reader (optionally
      verifying `league_id`) so ground truth and the agent can't be
      scored against two different leagues' settings.
- [x] CLIs: `python -m src.reasoning.recommend` (all three modes),
      `python -m src.reasoning.report`, `python -m evals.run_decision_eval`,
      and `python -m evals.build_ground_truth` take `--league-id`,
      defaulting to `SLEEPER_LEAGUE_ID` from `.env` (the same variable
      `src.ingest.sleeper` already used) and exiting with a clear
      message if neither is set. The CLIs call `load_dotenv()` themselves
      again for that default (PR #18 had moved it into `recommend()`/
      `generate_report()`; both places now, harmlessly).
- [x] Regression check, Victorious Secret 3.0 unchanged before/after --
      see "Validation".
- [x] Verified: no change touches `matchup_signals.py` or anything under
      `src/rag/` (`git diff --stat` -- the only `src/` files touched are
      `src/reasoning/{recommend,report,league}.py`), and the new guard
      test fails if anything in `src/signals/` or `src/rag/` ever imports
      from `src/reasoning/` or reads `scoring_settings`.

**Still implicit -- flagged, deliberately not fixed here.**
`MY_ROSTER_ID` (which roster in the league is "mine") is still read
from the environment inside `src/rag/lookup.py`'s `current_roster()`. A
`roster_id` is per-league, so it belongs alongside `league_id` as an
explicit input in a multi-league product -- but that change lives in
`src/rag/lookup.py`, which this phase was scoped not to touch. 5.2's
"league_id + your team within it" session model is where it gets
parameterized (an explicit `roster_id` parameter threaded through
`lookup.py` and `RecommendContext`). Two related 5.2 notes: (1) the
Chroma collection in `persist_dir` mixes league-specific chunks
(`league:settings`, `team:*`, `matchup:*`, `transaction:*`) with the
league-agnostic signal chunks in one collection, so `persist_dir` is
effectively per-league today and signals get re-embedded per league --
fine for one league, worth splitting when 5.2 stores more than one; (2)
`ground_truth.jsonl` rows say `scoring_format: "league_actual"` but not
which league, which becomes ambiguous the moment two leagues' evals
exist -- row schema left alone here (regenerating needs Sleeper access
this sandbox doesn't have), flagged for 5.2.

**Validation.** No real Victorious Secret 3.0 pull exists in this
sandbox (`data/raw/` is gitignored and Sleeper is blocked), so "before/
after against the actual ingested data" was run the closest honest way
available: a VS3.0-*shaped* fixture (real league ID, name, 12 teams,
Sleeper-shaped half-PPR `scoring_settings`, a roster of real players)
joined to the REAL committed `signals_2024_week5.parquet` and the REAL
nflverse player index, run under `origin/main` (a worktree) and under
this branch, for all three report types plus a `recommend()` call with
a scripted fake client exercising all nine tools (real tool execution,
only the model is faked). Result: **zero differences apart from the
additive `league_id`/`league_name` keys** -- every report entry, every
tool result, and the system prompt (byte-identical) match. A second,
full-PPR fixture league confirmed the other direction: its system
prompt shows `rec=1.0` and a `bonus_rec_te=0.5` key half-PPR lacks (and
a no-PPR fixture in the tests shows no `rec=` at all -- nothing invents
a reception value), while its reports are identical to the half-PPR
league's, exactly as the finding above predicts. One non-additive diff
did show up in that second league's `search_league_info` results and was
chased down: same code, same query, two independently-built Chroma
indexes from identical chunks returned different top-5 lists (HNSW
approximate-search recall -- a closer chunk was missing from one
index's results). That's `src/rag/retrieve.py` behavior this phase
didn't touch, not a regression; noting it because it will bite any
future exact-output regression check that includes semantic search.
**Gaps for Rohan's machine, same pattern as every prior phase:** (1)
re-run the real thing -- `python -m src.reasoning.recommend --report
drop` and a real question, with `SLEEPER_LEAGUE_ID` in `.env`, against
the actual ingested league, and confirm the output reads the same as
before; (2) no second *real* league exists in this sandbox's fixtures
(none was fabricated), so "a genuinely different scoring format produces
genuinely different, correct output" is only demonstrated at the
system-prompt level with fixtures -- ingest a friend's PPR/standard
league (`python -m src.ingest.sleeper --league-id <theirs>` into a
separate `raw_dir`, then `recommend(..., league_id=<theirs>,
raw_dir=...)`) to validate it for real, and note that only the prompt
will differ until something downstream actually consumes scoring
numerically.

### 5.2 — API + storage layer
Implemented. Full `pytest` suite is 246/246 (180 before this phase --
that count includes the .env test-isolation fix, PR #21, which this
phase is stacked on; 66 new tests across `tests/test_api_auth.py`,
`test_api_storage.py`, `test_api_leagues.py`, `test_api_main.py`, and
`test_roster_id.py`). Same sandbox blockers as every prior phase (no
`ANTHROPIC_API_KEY`, `api.sleeper.app` blocked) -- see "What's mocked
vs. what's real" below, which is the important part of this entry.

**What the step-1 investigation found.**
- *MY_ROSTER_ID's flow* was exactly the shape 5.1 flagged: read from
  `os.environ` inside `src/rag/lookup.py`'s `current_roster()`, which
  every "my"-flavored function (`my_players`, `my_players_by_position`,
  `my_team_record`, `my_current_matchup`) went through, called from
  `recommend.py`'s `get_my_roster`/`get_team_record`/
  `get_current_matchup` tool handlers, `report.py`'s `_report_header`,
  and `cli.py`. Nothing else in the project touched it. One process-wide
  variable answering "whose roster" is fine for one person's CLI and
  wrong for a server answering for several people at once.
- *`load_dotenv()` placement (PR #18)* was correct but per-request:
  every `recommend()`/`generate_report()` call walked the filesystem
  for a `.env` and re-parsed it. Harmless for a one-shot CLI process
  (exactly one call), pure waste for a server -- and since
  `load_dotenv()` never overrides a variable that's already set, repeat
  calls could never even change anything. Not a correctness bug, an
  efficiency one, invisible until there's a server.
- *`data_gaps` / stale markers' real shapes* differ between the two
  entry points, which matters for passing them through honestly:
  `recommend()` returns `data_gaps` at the top level (a list of
  `{reason, detail, player_name?}` straight from the model's
  `submit_recommendation` call) but its stale/`has_signals` markers are
  NOT top-level -- they live on each `get_player_signals` tool result
  inside `tool_calls` (`has_signals`, `stale`, `source_season`,
  `source_as_of_week`). `generate_report()` has the opposite shape:
  every entry carries `stale`/`source_season`/`source_as_of_week`, the
  report carries `notes`, and there is no `data_gaps` concept at all.

**What changed.**
- [x] `roster_id` is an explicit parameter, threaded the same way 5.1
      threaded `league_id`: `lookup.current_roster()` and the four
      `my_*` functions take `roster_id=None` (explicit wins; `None`
      falls back to `MY_ROSTER_ID` exactly as before, so the
      single-league CLI is unchanged -- pinned by tests and by the
      before/after regression below); `RecommendContext` carries it;
      `recommend()`/`generate_report()` accept `roster_id=None` and
      echo it on their results; CLIs take `--roster-id` (default: the
      env var). New `lookup.team_roster_for_roster_id()` for the API's
      roster endpoint. **This is the one `src/rag/` file touched, by
      explicit instruction; the guard test is unmodified and passes.**
- [x] `load_dotenv()` -> `recommend.load_dotenv_once()`: once per
      process (memoized), called from `recommend()`,
      `generate_report()`, both CLIs, and the API's startup hook. The
      PR #18 guarantee (a direct import loads `.env` on first use) is
      preserved and tested; five calls in one process now parse `.env`
      once (tested). `tests/conftest.py` resets the memo per test.
- [x] `src/api/auth.py`: username -> `GET /v1/user/<username>` ->
      user_id -> `GET /v1/user/<user_id>/leagues/nfl/<season>` (season
      defaults to Sleeper's `state/nfl`, the same source ingest uses) ->
      each league's `scoring_settings` verbatim + the user's own
      `roster_id` in it (via the existing `fetch_rosters`, matching
      `owner_id` or `co_owners`). Sleeper's documented "null body for
      an unknown username" and a 404 both become `UnknownSleeperUser`.
- [x] `src/api/storage.py`: SQLite, four tables (`users`, `leagues`,
      `sessions`, `query_counts`), `CREATE TABLE IF NOT EXISTS` on
      open, no migration framework. A session can only be pointed at a
      league Sleeper listed for its user.
- [x] `src/api/leagues.py` (not in the plan, but necessary): where a
      league's ingested data lives. The developer's own
      `SLEEPER_LEAGUE_ID` league keeps the flat `data/raw/sleeper/` +
      `data/chroma/` (so CLI and server share it); every other league
      gets `data/raw/leagues/<league_id>/{sleeper,chroma}/` (under
      `data/raw/` so the existing gitignore covers it), ingested on
      first use with the *existing* `sleeper.run()` + `embed.embed()`
      pointed at those dirs -- no new ingest code. The signals table
      stays shared and league-agnostic; it's embedded into each
      league's collection alongside that league's own chunks, exactly
      what `embed.main()` already does for one league (the
      per-league re-embedding of shared signal chunks is redundant at
      friend scale and is the `src/rag/` split 5.1 already flagged).
- [x] `src/api/main.py` (FastAPI, `uvicorn src.api.main:app`):
      `POST /api/leagues` (login), `GET /api/leagues/{username}`,
      `POST /api/sessions` (pick league; ingests on first use),
      `GET /api/sessions/{id}`, `GET /api/roster`,
      `GET /api/reports/{type}`, `POST /api/chat`. Chat exposes
      `recommend()`'s existing `messages` parameter for multi-turn --
      the API converts the Anthropic SDK content blocks in `messages`
      to plain dicts on the way out and passes them straight back in
      (the SDK accepts dict blocks), nothing re-implemented. The
      session's `roster_id` is passed explicitly into every
      `recommend()`/`generate_report()` call; the server never
      consults `MY_ROSTER_ID` (tested: with the env var pointing at
      another team, the session's roster still wins). Upstream
      failures (Sleeper unreachable at login or first-use ingest) are
      502 with a clear detail, not a bare 500 -- found by booting the
      real server in the sandbox and watching the blocked Sleeper call
      come back as a stack trace.
- [x] `data_gaps` and stale markers pass through as distinct fields.
      `/api/chat` returns `data_gaps` verbatim AND `signals_consulted`:
      every resolved `get_player_signals` tool result the agent saw,
      each carrying `has_signals`/`stale`/`source_season`/
      `source_as_of_week` exactly as the tool returned them -- a
      projection of real tool output (the only place those markers
      exist in `recommend()`'s return value), not a reconstruction, and
      never a single summary flag. `/api/reports/*` returns
      `generate_report()`'s dict verbatim (per-entry stale fields,
      `notes`). Tested for all three cases: grounded (`stale: false`),
      stale fallback (`stale: true, source_season: 2024`), and
      `has_signals: false`.
- [x] Per-user/day query cap: `ASKMADDEN_DAILY_QUERY_CAP` (default 25),
      counted per username per UTC day in one check-and-increment SQL
      statement (no double-spend under concurrency), 429 when hit,
      refused requests not counted. Only `/api/chat` counts -- reports
      never call Claude and are free.

**What's mocked vs. what's real (read this before trusting anything).**
Real in the tests: FastAPI routing/validation via `TestClient`, SQLite
in a temp file, Phase 5.1's league verification, `recommend()`'s actual
tool-use loop and its real tools, `generate_report()`'s real ranking,
all against real-shaped fixtures (real nflverse identities, real 2024
week-5 signal values). Mocked, at exactly these boundaries and nowhere
else: (1) Sleeper -- `auth.resolve_user_leagues` in the API tests and
`sleeper._get` in the auth tests, with responses shaped per Sleeper's
documented API (the fields read are the stable documented ones:
`user_id`, `league_id`, `name`, `season`, `roster_id`, `owner_id`,
`co_owners`); (2) league ingest -- `leagues.ingest_league` seeds
fixture data + a real Chroma index instead of hitting the network;
(3) the Claude model -- a scripted client that decides which tools to
call, the tools themselves run for real; (4) nflverse's player list,
the same monkeypatch every existing test uses. Also done for real:
booted the actual server with uvicorn and probed it (OpenAPI lists all
seven routes; unknown session -> 404; login -> 502 because Sleeper is
blocked here, which is the correct honest answer from this sandbox).
**Needs a real run on Rohan's machine, same pattern as every phase:**
```
uvicorn src.api.main:app --reload
curl -X POST localhost:8000/api/leagues -H 'content-type: application/json' -d '{"username":"<your sleeper username>"}'
curl -X POST localhost:8000/api/sessions -H 'content-type: application/json' -d '{"username":"<you>","league_id":"1389341490030862336"}'
curl "localhost:8000/api/roster?session_id=<from above>"
curl "localhost:8000/api/reports/drop?session_id=<id>"
curl -X POST localhost:8000/api/chat -H 'content-type: application/json' -d '{"session_id":"<id>","question":"who should I start at RB?"}'
```
Specifically unverified until then: that Sleeper's live responses
match the documented shapes this was written against (no fabricated
shape was used, but "documented" isn't "observed"); the first-use
ingest of a *second* league into `data/raw/leagues/<id>/` end to end
(Sleeper + nflverse + embed, all network); and a real Claude turn
through `/api/chat` including a multi-turn follow-up with the returned
`messages`. Known cost of the per-league layout: `sleeper.run()`'s
5MB player-pool cache lives in each league's own directory (its cache
path isn't parameterizable without touching `src/ingest/`, out of
scope), so the first ingest of each league re-fetches it once -- fine
for three leagues, worth sharing if this ever grows.

**Regression (direct calls unchanged after PR #20).** Same harness as
5.1: VS3.0-shaped fixture + real 2024 week-5 signals + real player
index, `origin/main` (post-#20) vs this branch, all three reports plus
a nine-tool `recommend()` run, called the old way (no `roster_id`,
`MY_ROSTER_ID` in the environment). Against the same Chroma index:
zero differences except the additive `roster_id: null` on
`recommend()`'s result; system prompt byte-identical. (Against
independently-built indexes the only extra diffs were again
`search_league_info`'s Chroma approximate-search variance, same as 5.1
found -- not this change.)

### 5.3 — Wire the mockup to real data
design/askmadden-ui-mockup.html has no framework dependency, so this
is a wiring pass. The responsive breakpoint (phone frame below 900px,
sidebar desktop layout above, one DOM + media queries) is already
built into the mockup — wiring real data into it does not require
building responsive layout from scratch, only replacing the hardcoded
`leagues` array, rec-card/waiver-row/roster-row markup, and the chat
transcript with real API renders:

Implemented. A wiring pass on design/askmadden-ui-mockup.html (its
`<script>` block and the hardcoded data markup only -- the CSS, the
landing view, and the responsive shell are untouched) plus one new file,
web/dev_server.py. No file under src/ changed. Full `pytest` suite is
246/246 (unchanged: nothing here is covered by pytest -- the validation
is browser-level, see below).

**What the investigation found before wiring.**
- The mockup's mock data was: a two-entry `leagues` array (id, avatar,
  name, meta string, team, initials) that `selectLeague()` parsed the
  header from; three hardcoded rec-cards ("Start today"), two waiver
  rows, a hardcoded "Trade center" card, a one-exchange chat
  transcript, roster rows with fabricated points/trend arrows and a
  Starters/Bench toggle, and two more waiver rows under Moves.
- The real API (read from src/api/main.py, not the docs): login
  returns `{user, leagues:[{league_id, name, season, roster_id}]}`
  **sorted by name** (storage's query does `ORDER BY name`); sessions
  return the stored row + `league_name` + `scoring_settings`; roster
  returns `{team_name, owner_display_name, players:[{player_id, name,
  position, team}], counts_by_position}` -- **no starters/bench split
  and no points**, so those mockup columns were removed rather than
  faked; reports come back verbatim with per-entry
  `stale`/`source_season`/`source_as_of_week`; chat returns
  `data_gaps` + per-player `signals_consulted` + `messages` +
  `queries_used_today`/`daily_query_cap`.
- **CORS: none.** src/api/main.py has no middleware and no static
  mount, so the mockup opened as a file:// page cannot call
  localhost:8000. Since src/api/ was off limits this phase, the fix is
  web/dev_server.py: a wrapper app (outside src/api/) that mounts
  design/ as static under /ui and the untouched API app under / on ONE
  origin -- `python -m web.dev_server`, open http://127.0.0.1:8000/ --
  so the browser's fetches are same-origin and need no CORS. Phase 5.6
  ("FastAPI mounts the one responsive frontend as a static route") is
  where this folds into src/api/main.py; adding CORS there is only
  needed if the UI is ever served from a different origin.

**What was wired.**
- [x] Login: real `POST /api/leagues`; an unknown username renders the
      API's 404 detail in a visible error block (`#login-error`), a
      502 says Sleeper is unreachable, a dead server says so too.
      Never fails silently.
- [x] Picker + switch-league sheet: one `renderLeagueList()` over the
      real list (name, season, league ID, your roster #); selecting
      calls `POST /api/sessions`, keeps `session_id` in JS state only
      (no localStorage), and sets the header from the real
      `league_name` + a scoring label derived from
      `scoring_settings.rec` (0.5 -> "Half-PPR", 1 -> "PPR", absent/0
      -> "Standard") + roster #. The "paste a league ID" box calls the
      same endpoint and surfaces the API's 404 honestly -- the API only
      opens leagues Sleeper lists for your username, and the UI says so.
- [x] Feed: three real reports, mapped by the mockup's own labels --
      "Start today" -> `/api/reports/start_sit` (starter card = START,
      each alternative = SIT, `reasoning`/`signals_summary` as the
      card text), a new "Drop candidates" section -> `drop`
      (`weakness_reasons`), "Waiver targets" -> `waiver_pickups`
      (`opportunity_score`, no "Add" button -- there is no write-back
      API and Phase 5's spec is read-only). The amber stale chip is now
      driven per entry by the real `stale`/`source_season` fields.
      Report `notes` render as a notes card.
- [x] Chat: `POST /api/chat` with `session_id` + question; renders
      `recommendation` + `reasoning`; threads the returned `messages`
      back into the next call (real multi-turn, nothing reimplemented);
      shows `queries_used_today of daily_query_cap`; a 429 renders the
      cap message as an error bubble. **Three distinct chips**: amber
      `stale` (from `signals_consulted[].stale`, with source season +
      player), grey `nodata` (from `data_gaps[].reason ==
      no_signal_data` and from `signals_consulted[].has_signals ==
      false`, deduped by player), coral `scope` (from
      `out_of_scope_capability`). Suggestion chips send real questions.
- [x] Roster: `GET /api/roster`, grouped by position with counts and
      team name; no fabricated points/trend/starters.
- [x] Moves -> Waivers: the same real waiver report, full list. Moves
      -> Trades: **chose the "ask chat on the user's behalf" option**,
      on request (a button), not automatic. Rationale: there is no
      composition endpoint and this session must not add one;
      `get_league_rosters` is only reachable through the chat agent;
      and running it automatically on tab open would silently spend one
      of the user's capped daily Claude queries every visit. The button
      sends a fixed question ("What's my weakest position, and which
      teams in the league have surplus there? Composition only -- don't
      propose specific trades.") through the same chat renderer (same
      chips, own message thread), under a "COMPOSITION · NOT A
      VALUATION" label with Phase 6 named as not built. The Feed's
      "Trade center" card now points here instead of showing a
      hardcoded answer. Nothing anywhere is wired to a trade value or
      fairness score.
- [x] Landing page and anything PWA-related: untouched.

**Validation -- exactly what was run.** The sandbox has no Anthropic
key and Sleeper is blocked, so the boundary mocks are the same ones
tests/test_api_main.py uses: Sleeper login mocked at
`auth.resolve_user_leagues`, league ingest mocked to seed the two-team
fixture + a real Chroma index (plus one unrostered real player, Justin
Jefferson with his real 2024 week-5 numbers, so the waiver path has an
entry), nflverse's player list mocked, and a scripted Claude that
always calls real tools (`get_my_roster`, `get_player_signals`, and
`get_league_rosters` when the question mentions "weakest") before
submitting an answer carrying one `no_signal_data` and one
`out_of_scope_capability` gap. Everything else was real: the actual
web/dev_server.py wrapper serving the actual mockup file, the actual
API app, SQLite, Phase 5.1 league verification, `recommend()`'s tool
loop, report ranking. Then a real headless Chromium (Playwright)
drove the page like a user, twice -- at 390px (phone frame, bottom tab
bar) and 1280px (sidebar layout) -- with 58 assertions, all passing:
root redirect -> landing; unknown username -> visible error; login ->
two real leagues with real roster #s; select -> session -> header
shows real name/"Half-PPR"/roster #; the correct layout for the
viewport; start/sit renders Barkley START + Cook SIT from real report
entries; drop renders weakness reasons; waivers render Jefferson
(unrostered) and not Chase (rostered by the other team); no stale
chip when signals are current; roster grouped by position with real
counts and no fabricated columns; chat renders recommendation +
reasoning + a grey no-signal-data chip naming the player + a coral
out-of-scope chip; the cap status line; **the second question's first
model call carried 6 prior messages (server-side log), i.e. real
multi-turn threading**; Moves/Waivers real; Trades is on-request and
labeled; the trades question reached the model verbatim and rendered
through the chat path; switch-league sheet marks the active league,
switching updates the header and the roster shows the OTHER league's
players; the 6th query succeeds and the 7th is refused with the API's
429 detail visible in the UI; no uncaught JS errors; and the browser
actually hit all seven routes. (Two console errors are expected in the
sandbox: Google Fonts is blocked by the network policy, and there's no
favicon.) The harness and browser script live in this session's
scratchpad, not the repo -- they're validation, not product.

**Bugfix pass after Rohan's first live run (four observed bugs).**
Scope was bug fixes only: no src/reasoning/ or src/rag/ changes, no
new features. Full `pytest` suite 248/248 (246 + 2 new). Every fix
below was reproduced first, then fixed, then re-checked in the same
way it was reproduced.
- **Bug 1 -- Feed's start_sit and drop 500'd, waiver_pickups loaded.**
  Reproduced by running the real dev server (real nflverse player
  index, Sleeper/ingest mocked at the API boundary as always) with the
  league already on disk, restarting the process cold, and firing the
  three Feed requests concurrently the way the browser does: on the
  third cold restart, start_sit and drop both returned 500 and
  waiver_pickups 200 -- the exact symptom. The server's own traceback
  was NOT the player-index fetch (that was the suspect; with the real
  nflverse fetch three concurrent requests were fine, cold and warm)
  but chromadb: `AttributeError: 'RustBindingsAPI' object has no
  attribute 'bindings'` -> `ValueError: Could not connect to tenant
  default_tenant`. Cause, confirmed in chromadb 1.5.9's source:
  `SharedSystemClient._create_system_if_not_exists` caches the new
  per-path System in a class dict BEFORE calling `start()`, with no
  lock, so when two request threads open `PersistentClient` on the same
  league directory at the same moment (src/rag/retrieve.py opens one
  per query), the second thread gets a cached-but-unstarted client.
  That explains every clue: start_sit and drop resolve every rostered
  player's signals through Chroma, waiver_pickups reads only the
  parquet table; a single curl never races; and it only bites in a
  process that has never opened that path (deployed case: league
  ingested earlier by the CLI, server started later -- which is also
  why PR #23's own browser validation never saw it: its harness
  ingested in-process and had already opened the client). Fix (in
  src/api/leagues.py, the allowed layer): `warm_chroma(persist_dir)`
  opens the shared client once per path under a `threading.Lock`,
  called from `ensure_league_data()`, which every league-scoped request
  already goes through -- so the first open is serialized and every
  later concurrent open reuses the started System. Not "fetch
  sequentially in the Feed": the Feed still fires all three at once.
  Re-checked: six more cold restarts with concurrent requests, 0 500s,
  0 tracebacks. Regression test `tests/test_api_concurrency.py` builds
  the league's Chroma index in a subprocess (so the test process is
  genuinely cold), widens chromadb's start window with a short sleep so
  the race is deterministic rather than 1-in-3, fires the three reports
  concurrently through TestClient, and asserts all 200 + exactly one
  chromadb start; with `warm_chroma` neutralized it fails (verified).
  **Flagged, not fixed (src/rag/):** retrieve.py creating a new
  `PersistentClient` per query is the deeper cause -- one shared client
  per path there would remove the hazard at the source.
- **Bug 2 -- picker and app view visible at once.** Cause: CSS
  specificity, not the JS. `goView()` toggles `.show` correctly and
  `#view-app` never had it before a league was selected, but the rule
  `#view-app{ display:flex }` (an ID selector, specificity 1-0-0)
  outranks `.view{ display:none }` (0-1-0) regardless of `.show`, so
  the app shell rendered on every view -- including under the landing
  page, which is why the phone frame and "Loading start/sit…" card
  bled below the login/picker card. Present in the mockup file as
  delivered in Part 1, not introduced by #23's wiring. Fix:
  `#view-app{ display:none }` + `#view-app.show{ display:flex }`.
  Verified in headless Chromium at both breakpoints: the set of
  visible `.view` elements is exactly `[view-landing]` on load,
  `[view-picker]` after login, `[view-app]` after selecting a league;
  against origin/main's file the same check showed `[view-landing,
  view-app]` on load.
- **Bug 3 -- desktop rendered as a boxed card.** Cause: the ≥900px
  media query kept the phone-mockup chrome (`max-width:1180px`,
  `height:740px`, `border-radius:20px`, a box-shadow and border, and
  36px/24px padding around it). Fix, in that media query only:
  `#view-app{padding:0}`, `.app-shell{width:100%; max-width:none;
  height:100vh; border-radius:0; box-shadow:none; border:none}`,
  `.screen{border-radius:0}`. The sidebar/content layout underneath is
  untouched and the mobile phone frame is untouched. Verified at 1280px:
  the shell's bounding box starts at x=0 with width == viewport width
  and height == viewport height, computed radius 0px, no shadow, no
  border; at 390px the phone-frame radius and shadow are still present.
- **Bug 4 -- dead bell and avatar buttons.** Chose a split: the bell is
  REMOVED (there is no notifications system anywhere in the project,
  so an icon for it is a promise the product can't keep), and the team
  avatar is KEPT as a static indicator -- it shows your team's initials
  from the real roster response, which is information the header
  otherwise lacks -- with `cursor:default`, no handler, and a
  `title="Your team: <name>"` tooltip so it reads as a label, not a
  button. Verified in the browser: zero `.icon-btn` elements in the
  header; the avatar's computed cursor is `default`, it has no onclick,
  and its title starts with "Your team".

**Needs a real run on Rohan's machine:**
```
python -m web.dev_server      # then open http://127.0.0.1:8000/
```
1. The login flow through the actual browser UI against live Sleeper
   (real username -> real league list -> first-use ingest, which will
   take a while and is the first time this UI's "Loading league data"
   state is seen for real).
2. A real multi-turn chat with the real model -- the chips are driven
   by the real `data_gaps`/`signals_consulted`, so this is also the
   first real look at whether the model's gap entries read well as
   chips.
3. Moves -> Trades with the real model: whether it actually calls
   `get_league_rosters` for the fixed question and names surplus teams
   (Phase 3.8 validated this for the CLI; through the UI it's unverified).
4. Google Fonts load (blocked here, so the sandbox rendered fallbacks).
**Known, flagged, not fixed (all outside this session's scope):** the
report endpoints rebuild the nflverse player index on every call
(`player_index.build_player_index()` inside `generate_report()`), which
is a network fetch per report on real data -- three reports per Feed
load. Fine at friend scale, but it's the first thing to cache when 5.6
deploys, and it lives in src/reasoning/. And CORS/static serving belong
in src/api/main.py for 5.6, per above.

### 5.4 — PWA installability
- [ ] manifest.json (icons, theme-color, display: standalone)
- [ ] Minimal service worker (cache-first static assets is enough)
- [ ] Verify "Add to Home Screen" on iOS Safari and Android Chrome —
      this is the actual mechanism for getting this on a phone, no
      App Store submission

### 5.5 — Landing page (front door)
Reframed from "static marketing site, separate from the app": the
landing page is now the `#view-landing` view inside the same
responsive design/askmadden-ui-mockup.html — the entry point users
see first, with a CTA into the login view. Still its own small,
independent piece of content work (copy, feature cards, the eval
band), but one view in one file, not a separate build target, and
it shares the app's design tokens by construction.
- [ ] Landing copy/feature cards finalized (the mockup's current text
      is a first draft)
- [ ] Eval-numbers band (`#eval-numbers`): its three metrics are
      placeholders ON PURPOSE, labeled as such in the mockup, never
      invented. Retrieval accuracy and decision accuracy are gated on
      `evals/run_eval.py` / `evals/run_decision_eval.py` running at
      volume (no Phase 4 dependency). Matchup-fit accuracy is
      specifically gated on Phase 4's coverage classification landing
      — per the signals table's own "modeled proxy" note for that
      signal, there is no matchup-fit number to report until then.
      Populate each only when its real number exists.
- [ ] No auth, no API calls from the landing view itself

### 5.6 — Deployment
- [ ] FastAPI mounts the one responsive frontend (landing + app in
      one file) as a static route — one deployment, one URL, no CORS
- [ ] Deploy to free-tier host (Railway/Render/Fly.io)
- [ ] Get 2-3 friends in different leagues to actually use it
- [ ] README: document the "started as one league, generalized to a
      product" story, with real eval numbers from run_decision_eval.py

## Phase 6: A crude, explicitly-labeled trade-value proxy
Not started. Deferred past Phase 5, not dropped: Phase 3.8's real-model
validation confirmed a complete, honestly-bounded product (composition +
signals + an explicit, correctly-refused valuation gap) is demo-ready
now -- confirmed live that a compound trade question correctly
identifies real composition (weak position, surplus-position trade
partners named concretely) while explicitly and correctly declining
valuation/fairness, even under a more insistent phrasing ("give me a
solid trade proposal I can propose right now"). Trade valuation is a
real value-add, not a blocker -- this is sequencing, not scope-cutting.

See `PROJECT_SPEC.md`'s Phase 6 section for the planned scope (a
`season_points_so_far_proxy` field on `get_player_signals`, computed from
real nflverse weekly stats + this league's actual scoring settings,
as-of-date filtered -- explicitly not a real trade-value model, labeled
as a proxy everywhere it appears, same pattern as `report.py`'s waiver
`opportunity_score`).
- [ ] Not started

## Backlog -- future phases, not yet started

Parked items that came out of real usage testing after PR #24 (the
Phase 5.3 bugfix pass). None of these is active work and none blocks
the remaining Phase 5 items (5.4-5.6); they are recorded here so they
aren't lost, not because they're next.

### Phase 7: Coaching-scheme fit signal
Not started. Sequenced after Phase 6 (trade-value proxy): both are new,
speculative signal work, and Phase 5's remaining frontend/deployment
items plus Phase 6 stay the priority. See `PROJECT_SPEC.md`'s Phase 7
section for the full write-up. The gap it closes: when current-season
signals are stale or not yet computed (the exact situation at the start
of a season), the reasoning agent falls back on unlabeled general
knowledge ("consistently elite performance history") instead of a real
signal -- the same class of ungrounded claim Phase 3.7's addendum
guards against in the trade context. This is signals-layer work (a
structured, deterministic per-player lookup through `get_player_signals`,
inheriting the existing `stale`/`source_season` labeling), not RAG.

Tier 1 -- coaching-change flag (cheap, near-zero fabrication risk,
shippable well before Tier 2). Hand-maintained reference metadata is
fine here: it's a roster-of-record fact like a schedule, not eval
ground truth, so the "never hand-authored" rule (which guards
`ground_truth.jsonl`) doesn't apply.
- [ ] Build the reference table (team, season, OC name)
- [ ] Join onto `matchup_signals.py`'s existing per-team output
- [ ] Add to `get_player_signals`' returned fields
- [ ] Update system prompt: when this flag is true, the model may
      note it as a fact, but any claim about *how* the scheme affects
      a specific player still needs Tier 2's real signal to back it --
      until Tier 2 exists, the model should not speculate about fit,
      only note the change occurred

Tier 2 -- real scheme-fit signal (comparable scope to Phase 4, its own
eval-gated validation before it's trusted in a real recommendation).
- [ ] Extend coordinator table with multi-season history per OC
- [ ] Compute scheme profiles from nflverse play-by-play per stint
      (shotgun rate, play-action rate, personnel groupings, no-huddle
      rate, aDOT tendency)
- [ ] Compute player career splits against the same scheme dimensions
- [ ] Compute the fit score, explicitly labeled as a modeled proxy
      (same treatment as the matchup-fit score, never presented with
      the confidence of a measured signal like target share)
- [ ] As-of-date filtering applies here too: no signal may use data
      from after the eval week's kickoff, same rule as everything else
- [ ] Eval check before this is trusted in a real recommendation: does
      including this signal actually move decision accuracy, scored
      the same way every other signal-quality question gets answered
      in this project -- don't add complexity speculatively
- [ ] Update `get_player_signals` and the system prompt once validated

### Feed card redesign + position filters
Not scoped yet -- needs its own design session before implementation.
Real usage testing found the Feed's rec-cards render as inconsistent-
length walls of text: a single card's text includes full comparisons
to 3-4 other players concatenated together, rather than just that
card's own player. Likely shape of the fix: truncate each card to its
own player with an expandable "why" section, and add position filter
chips (QB/RB/WR/etc.) above the Start Today section so a long list can
be narrowed. Frontend only (`design/askmadden-ui-mockup.html`); the
report payloads themselves already carry per-player entries.
- [ ] Design session: card content boundary + expandable "why" +
      position filter chips
- [ ] Implement once designed

### Chat vs. Feed can recommend differently on the same signals
Needs a product decision before any code changes -- flagged so it isn't
lost, not resolved. Real usage testing found Chat (`recommend()`) and
the Feed's start_sit report (`generate_report()`) reached different
verdicts for the same real decision (one QB question), because they are
two independent reasoning paths over the same signals: `report.py` uses
a deterministic ranking/scoring formula, while `recommend()` is Claude
reasoning freely and can fall back on general knowledge when
current-season signals are stale (the same gap Phase 7's Tier 1 is
aimed at). This is a real product-consistency question, not a bug:
should the two paths be guaranteed to agree (e.g. Chat defers to
`report.py`'s ranking and only adds explanation), or is disagreement
acceptable and expected because they serve different purposes?
- [ ] Product decision: guaranteed agreement vs. accepted divergence
- [ ] Only then: any code change
