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
- [ ] Live validation gap (same two sandbox blockers as every phase so
      far): no `ANTHROPIC_API_KEY` here, so the system prompt's new
      stale-handling instruction was never validated against the real
      model, only mechanically (the fake-client tests confirm
      `_tool_get_player_signals`'s output shape, not that a real model
      actually reads and honors the new prompt language). Re-run
      `python -m src.reasoning.recommend "should I start Saquon Barkley"`
      for real once the 2026 season is far enough along to have real
      current-season signals *and* once early enough (or artificially, by
      pointing `--as-of-week` at week 1) to still exercise the fallback
      path, to confirm the real model actually says "this is 2025 data"
      rather than silently treating it as current.

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
