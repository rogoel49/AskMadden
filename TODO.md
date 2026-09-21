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
- [x] **Known gap, flagged not built -- now CLOSED by Phase 5.7, see its
      entry below.** At the time: `src/scheduler/refresh.py` didn't exist
      (`src/scheduler/` was an empty `__init__.py`), so every report and
      every `recommend()` call was only as current as whenever someone
      last ran `matchup_signals.py` + `embed.py` + `sleeper.py` by hand.
      Fine for this session's real-data validation (a fixed,
      already-computed week-5 signals table was exactly what was being
      validated against), a real gap against a live season. Deliberately
      deferred to its own scoped session -- deciding a cadence, an
      idempotent refresh strategy for `embed.py`'s full-rebuild design,
      and where it runs in Phase 5's hosted deployment -- which is what
      Phase 5.7 did. (All three questions got answered there: 6 hours
      from nflverse's measured build cadence; `embed.py`'s full rebuild
      turned out to be exactly the right shape for repeated runs and
      needed no change; a daemon thread for dev with `--once` for 5.6's
      cron/worker. It also turned up a real bug the scheduler would
      otherwise have triggered -- see 5.7's investigation notes.)

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
      `src/scheduler/refresh.py` gap below (not built at the time;
      built in Phase 5.7), and
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
deliberately not touched in this unit; built in Phase 5.7, which also
makes exactly this kind of stale index self-correcting) rather than a
defect in the
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

## Fixed: Chat and the Feed's start_sit report could reach different verdicts on the same signals
Not part of any phase's planned scope -- came out of real usage testing
after PR #24 and sat in the Backlog as a product question until the
decision was made (Chat's verdict on a head-to-head / start-sit
comparison must come from the same deterministic ranking the Feed shows;
Chat may add explanation and context on top; not up for re-litigation).
This session implemented that decision. Scope was `src/reasoning/`
only -- no `src/api/`, `design/` or `web/` changes.

**What the investigation actually found (Step 1, before changing
anything).** The observed case: asked "should I start Justin Herbert or
Patrick Mahomes," Chat's reasoning said it was "defaulting to Mahomes
based on his consistently elite performance history" once the
current-season signal went stale. Reproduced the divergence with the
real committed numbers: on the real 2025 season-end rows
(`data/processed/signals/signals_2025_week19.parquet`), `report.py`'s
opportunity score gives Herbert ~0.46 (EPA trend +0.19, red zone share
3%) vs. Mahomes ~0.06 (red zone share 2%, target share ~0%, and NO
computed EPA trend at all) -- so the Feed's start_sit report says
Herbert, explicitly stale, and Chat said Mahomes from reputation. Two
independent paths, confirmed, not assumed:
- `report.py`'s verdict was `_opportunity_score()` (a fixed-weight
  composite: 2 x EPA trend + 3 x red zone share + 2 x target share) plus
  a descending stable sort inlined in `_start_sit_report()`, picking
  `grounded[0]`. The score function was standalone, but the verdict
  (load tables -> attach current-or-stale row -> score -> sort -> top,
  with the "fewer than 2 grounded -> skip" rule) was only reachable by
  running a whole-roster report. Nothing could ask it about an
  arbitrary pair.
- `recommend()` had no ranking tool and no prompt rule about comparison
  questions at all. It answered a start/sit question from
  `get_player_signals`' prose (which carries the numbers only inside a
  sentence, plus a `[STALE -- ...]` prefix) and its own judgment. The
  Phase 3.7 addendum's "every specific claim must be backed by a tool
  call" rule was written for trade advice and never mentioned start/sit;
  in practice a stale signal was exactly the opening for "performance
  history" to fill the gap.
- A wrinkle that shaped the fix: `report.py` imports `recommend.py` (to
  reuse its tools), so `recommend.py` could not simply import
  `report.py`'s scoring back -- circular import. The shared logic needed
  its own module.

**What changed (Step 2).**
- [x] New `src/reasoning/ranking.py`: the weights, thresholds,
      signals-table loading, prior-season fallback, `opportunity_score`,
      `fmt_signal_row`, `weakness_reasons` and stale markers moved there
      *verbatim* from `report.py` (no reimplementation -- one copy),
      plus a `SignalTables` holder (load once, reuse per candidate) and
      the one genuinely new function, `rank_candidates()`: the
      sort-and-take-the-top step that was inlined in
      `_start_sit_report`, now also reporting `verdict` as `"clear"` /
      `"tied"` (top two scores within 1e-9) / `"insufficient_data"`
      (fewer than 2 rankable -- the same condition under which the Feed
      skips a position), with `recommended` set only on `"clear"`.
- [x] `report.py` refactored onto it. `_start_sit_report` now calls
      `rank_candidates()` for its verdict -- literally the same call
      Chat makes -- and `generate_report()`'s output is unchanged
      (every existing `tests/test_report.py` test passes untouched).
      One deliberate non-change: on an exact tie the report keeps its
      pre-existing behavior (stable sort, first candidate in roster
      order is listed as the starter) rather than dropping the entry or
      adding a note, because the instruction was not to change
      `generate_report()`'s logic. Chat, by contrast, is told to say
      "tied". That's the one edge where the two can still differ, and
      it's a float-exact tie between different real players -- rare,
      but documented rather than papered over. If it matters, the
      report could surface `verdict == "tied"` in `notes` in a
      follow-up; that's a report-output change, so it wasn't made here.
- [x] New `rank_players` tool in `recommend.py` (`TOOLS` + `_DISPATCH`,
      same wiring as every other tool). Takes `player_names` (2+),
      resolves each name with the same `player_index.resolve_player()`
      path `get_player_signals` uses (ambiguous/unknown names come back
      in `unranked` with `reason: "ambiguous"` + candidates /
      `"unresolved"`, never guessed), attaches the same current-or-stale
      row the report would use, calls `rank_candidates()`, and returns
      `verdict` / `recommended` / `ranked` (each entry with
      `opportunity_score`, a `signals_summary` citing the actual numbers,
      and the explicit `stale`/`source_season`/`source_as_of_week`
      markers) / `tied_at_top` / `unranked` (`reason: "no_signal_data",
      has_signals: false` for a resolved player with nothing computed).
      Also `same_position`: the Feed's start_sit only compares within a
      position and the score isn't position-normalized, so an RB-vs-WR
      flex comparison is flagged as a raw cross-position comparison (the
      same thing the Feed's waiver report does across positions), not
      passed off as a within-position start/sit verdict.
- [x] `RecommendContext` gained `signals_dir` (default: the same
      `matchup_signals.PROCESSED_DIR` `generate_report()` reads -- Chat
      and the Feed rank from the same files by construction) and a lazy
      `signal_tables()` (parquet read once per context, only if a
      comparison is actually asked). `recommend()` gained a matching
      `signals_dir` parameter, defaulted, so `src/api/main.py` needed no
      change.
- [x] System prompt: a new paragraph, placed before the "always end with
      submit_recommendation" paragraph. For any comparison between named
      players, call `rank_players` once with every player, make the
      recommendation and `player_id` its `recommended` player, and
      explain with the `signals_summary` numbers; never reach a verdict
      from general knowledge (the real failure phrase, "consistently
      elite performance history," is named in the prompt as the thing
      not to do); a verdict on `stale: true` entries is still the
      verdict (the Feed shows the same one) -- say which season it rests
      on, don't override it; on `"tied"` say the ranking can't separate
      them and pick nobody; on `"insufficient_data"` say the comparison
      can't be grounded, add a `no_signal_data` data_gaps entry per
      player with no signals, and don't pick the one who happened to
      have data. Same standard as Phase 3.7's trade rule.
- [x] Code-level guard, because Phase 3.7's own addendum showed a prompt
      alone is a documented non-safeguard: `recommend()`'s loop now
      checks every `submit_recommendation` against the `rank_players`
      results from the same call (`_verdict_contradiction()`). If the
      submitted `player_id` is one of the compared players but is NOT
      what the ranking supports -- a different player than
      `recommended`, or any pick at all on `"tied"` /
      `"insufficient_data"` -- the submit is not passed through: it's
      recorded in `tool_calls` and handed back to the model as an
      `is_error` tool_result saying exactly what contradicts what, and
      the loop continues so the model resubmits. A model that keeps
      contradicting the ranking ends in the existing graceful
      `max_turns_exceeded` result, never in a wrong verdict presented as
      grounded. Honest limit: the guard keys on `player_id`. A
      recommendation that names the wrong player in prose with no
      `player_id` set passes through -- that half is prompt-only, same
      as Phase 3.7 (pinned by a test that says so).
- [x] Not touched, deliberately: `generate_report()`'s logic/output
      (source of truth, per the decision), `src/api/`, `design/`,
      `web/`, the score's weights (aligning to the Feed's ranking was
      the decision; whether the ranking is *good* is a separate
      question -- see the flag below).

**Validation (Step 3).** Full `pytest` suite is 266/266 (248 before
this session; 18 new tests in `tests/test_verdict_alignment.py`, which
reuses `tests/test_report.py`'s real 2024 week-5 fixtures).
- [x] (a) Clear-cut: `rank_players` on Saquon Barkley vs. James Cook
      (real 2024 wk5 rows) returns `"clear"`/Barkley, and
      `generate_report("start_sit")` on a roster of the same two returns
      the same `player_id`, the same ordering, and the identical
      `signals_summary` text; a fake-client `recommend()` run that
      submits Barkley passes straight through with `player_id` equal to
      the ranking's; one that submits Cook is bounced (asserting the
      recorded rejection, the `is_error` tool_result on the right
      `tool_use_id`, and the corrected resubmit going through); a model
      that never stops contradicting ends in `max_turns_exceeded` with no
      Cook verdict.
- [x] (b) Ambiguous: two players with identical rows -> `"tied"`,
      `recommended: None`, both in `tied_at_top`; a pick on that tie is
      bounced and an honest "it's a tie" answer without `player_id` is
      accepted. One side with no signals -> `"insufficient_data"`,
      Cook in `unranked` as `no_signal_data`, and the Feed skips the
      position under the same condition; picking the only player with
      data is bounced. Ambiguous ("McCaffrey") and unknown names come
      back unranked with candidates / unresolved.
- [x] (c) Real-signals path: with current-season rows for both AND a
      prior-season file on disk with the numbers reversed, the ranking
      uses current data (`stale: false` everywhere, no `[STALE` text),
      matches the Feed, and a later-week file doesn't leak into a week-5
      ranking (as-of-week filtering, same rule as the report).
- [x] The real case, as far as this sandbox can take it: the real
      Herbert/Mahomes 2025 season-end rows under an empty 2026 season.
      `rank_players` and `generate_report("start_sit")` both say Herbert,
      both explicitly `stale: true, source_season: 2025`, identical
      cited numbers; a scripted replay of the original "defaulting to
      Mahomes based on his consistently elite performance history"
      submit (with Mahomes' `player_id`) is bounced and the Herbert
      resubmit goes through.
- [ ] **Needs real-model re-validation on Rohan's machine** (same
      sandbox blockers as every prior phase: no `ANTHROPIC_API_KEY`, no
      local Sleeper/Chroma data). Run the exact question through Chat
      and the Feed side by side:
      `python -m src.reasoning.recommend "should I start Justin Herbert or Patrick Mahomes?"`
      and `python -m src.reasoning.recommend --report start_sit`, and
      confirm (1) Chat's `tool_calls` include a `rank_players` call,
      (2) its stated verdict is that call's `recommended` player and
      matches the report's `recommended_starter` at QB, (3) the
      reasoning names 2025 as the source season if the signal is still
      stale and does not lean on reputation, and (4) no rejected
      `submit_recommendation` shows up in `tool_calls` (if one does, the
      guard worked but the prompt didn't -- worth knowing). Also worth
      one try each: a tied pair and a rookie-vs-veteran pair to see the
      model actually say "tied"/"can't ground" rather than pick.

**Flags for follow-up, found along the way (not done here, out of
scope).**
- `src/api/main.py`'s `signals_consulted` (what the UI's stale chip
  reads) only harvests `get_player_signals` results. A Chat comparison
  answered via `rank_players` alone carries its stale markers inside
  the `rank_players` result, which `signals_consulted` doesn't look at
  -- so the UI may not show the stale chip on a comparison answer
  unless the model also called `get_player_signals`. One-line-ish
  `src/api/` change to also harvest `rank_players`' `ranked` entries;
  not made here because `src/api/` was out of scope.
- The ranking is now the single verdict for both surfaces, which makes
  its quality matter more, and for QBs it is honestly thin: the score
  is built from EPA trend, red zone share and target share, of which
  only EPA trend says much about a QB (red zone *share* is ~2-3% for
  any QB, target share ~0). The real Herbert/Mahomes verdict turns
  almost entirely on Mahomes' 2025 row having no computed `epa_trend`.
  That's a signals-quality question for `matchup_signals.py` /
  the weights (candidates: a QB-appropriate composite, or Phase 6's
  points-based proxy), not an alignment question, and it was
  explicitly not this session's decision to re-litigate.

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

## Fixed: the day after real games, every feed card was [STALE -- 2025] because the as-of week read Sleeper's lagging `display_week`
Found by Rohan on the first real-usage look after week 1 of the 2026
season (Tuesday 2026-09-16): every start/sit card on the Feed said
"[STALE -- no current-season signal yet, showing 2025 season-end
reference instead]" even though week 1 had been played and the 5.7
refresh had run overnight. First live report of the fixed phone layout
(PR #29) too -- Rohan confirmed the phone-in-phone rendering is gone on
his real iPhone.

**What was actually wrong (measured, not inferred).**
- The refresh was fine. `refresh_status.json` showed a clean cycle at
  06:55 UTC: nflverse said 1 completed week, target as-of-week 2, and it
  wrote `signals_2026_week2.parquet` (313 rows). Stafford, Goff, Pollard
  and Gibbs were all in it with real week-1 numbers.
- The reasoning path never asked for that table. `recommend.
  _infer_season_and_week()` (used by both `generate_report()` and
  `recommend()`) read Sleeper's `nfl_state.json` and took
  `display_week` first. Sleeper's live state that morning, verbatim:
  `week: 2, leg: 2, display_week: 1`. `display_week` is the week the
  Sleeper *app is still showing* and lags until midweek; `week`/`leg`
  advance once the previous week's games are over. So the report asked
  for a 2026 week-1 table -- which correctly never exists (a week-1
  table would have zero current-season plays) -- and every player fell
  back to 2025.
- Reproduced directly on the Fellowship league, roster #2: inferred week
  (1) -> 4 of 4 starters stale, 21 players on the fallback; explicit
  `as_of_week=2` -> 0 of 4 stale, 4 players on the fallback (rookies
  with no week-1 snaps: Horton, Royals, Thornton Jr., Charbonnet --
  exactly the honest labeling 3.6 was built for).
- Same field in the Sleeper ingest: `sleeper.run()` defaulted its
  matchup/transaction week to `display_week`, so the league pull wrote
  `matchups_week_1.json` on Tuesday and `get_current_matchup` would
  have named the opponent you already played.

**Why the earlier "pinned to Sleeper" decision stands.** The signals-
refresh investigation (see "Reports (default week): pinned to Sleeper"
above, and `test_default_report_stays_pinned_to_the_leagues_own_state_
week`) deliberately kept the league's own Sleeper state as the authority
for the DEFAULT week rather than "whatever is newest in the signals
directory", and that reasoning is untouched. The defect was one field
below that decision: the wrong one of Sleeper's three week fields.

**What changed.**
- [x] `src/ingest/sleeper.py`: new `current_week(state)` -- preference
      order `week`, `leg`, `display_week`, then 1 -- with the live
      Tuesday payload in its docstring. `run()` uses it for the
      matchup/transaction week.
- [x] `src/reasoning/recommend.py`: `_infer_season_and_week()` uses the
      same helper (one definition of "this week" for both the ingest and
      the reasoning path). `src/scheduler/refresh.py`'s docstring names
      the field.
- [x] Tests (322/322, was 319): `current_week()` preference order;
      `run()` writes `matchups_week_2.json` (not week 1) under the
      lagging state; `_infer_season_and_week()` returns week 2 under the
      lagging state and still honors display_week-only pulls (every
      existing fixture carries only `display_week`, so they are
      unchanged and still pass). The pinned-to-Sleeper test's docstring
      says which field.
- Validated live before Sleeper caught up: with the lagging state still
  on disk, the inferred default report came back season 2026 / week 2,
  0 stale starters, real week-1 numbers (Dart, J. Williams, Otton). A
  real `python -m src.scheduler.refresh --once` then ran clean (99s, 2
  leagues) and wrote `matchups_week_2.json`; `current_matchup(roster 2,
  week 2)` resolves the week-2 opponent. Honest note: by the time that
  refresh ran, Sleeper had flipped `display_week` to 2 itself, so
  today's symptom would have self-healed within hours -- and come back
  every Tuesday. The regression tests are what pin the fix, not that
  run.

**Flagged, not fixed here.**
- [x] **EPA "trend" is degenerate with one week of data and mislabeled.**
      Fixed in the very next session -- see "Fixed: the early-season EPA
      trend was identically zero and printed as 'trending down'" below.
      Every one of the 313 rows in `signals_2026_week2.parquet` has
      `epa_trend == 0.0` (there is nothing to trend against yet), and
      `ranking.py` renders 0.0 as "efficiency trending down (+0.00
      EPA/play)" (`> 0` is "up", everything else "down"). Every card on
      the Feed says it right now. Two parts: the prose should say
      "no trend yet" (or nothing) when the trend is 0 / spans one week,
      and `matchup_signals.py` should expose how many weeks the trend
      covers so the ranking weight can be muted early in the season.
      Signal-computation + prose, its own session.
- [ ] Transient stale window: between the end of Monday night's game
      (Sleeper flips `week`) and the next 6-hour refresh, the inferred
      week has no table yet and everything falls back to stale, honestly
      labeled. A refresh triggered on "inferred week has no signals
      table" would close it. Not a correctness issue -- the loader never
      reads a later table than requested and the refresh only builds from
      completed weeks, so no future data can leak either way.
- [ ] Residual assumption: Sleeper advances `week` only after a week's
      games are over. If it ever flipped early, the as-of table for the
      new week would simply not exist yet (stale fallback, no leak).

## Fixed: the early-season EPA trend was identically zero and printed as "trending down (+0.00)"
Found on the same first in-season look as the display_week fix above:
once the feed showed real 2026 numbers, every card opened with
"efficiency trending down (+0.00 EPA/play)".

**Cause (signal computation, not prose).** `recent_efficiency_trend()`
is `trailing-window EPA/play minus season-to-date EPA/play`. With
`as_of_week <= trailing_games + 1` (weeks 2-4 on the default 3-game
window) the trailing window reaches back past week 1, so it IS the
season to date: trailing == season and the difference is exactly 0.0
for every player. Measured on the live table: 313 of 313 rows at 0.0.
The prose then called anything not `> 0` "down". The 2024 week-5
validation never saw this because week 5 is the first week with a
baseline outside the window.

**What changed.**
- [x] `src/signals/matchup_signals.py`: the table carries
      `epa_baseline_plays` (season plays before the trailing window),
      and `epa_trend` is null whenever that is 0 -- a difference needs a
      baseline to differ from. `season_plays` is untouched, so the row
      still counts as current-season data for the 3.6 stale-fallback
      threshold. League-agnostic, per CLAUDE.md's principle.
- [x] `src/reasoning/ranking.py`: a null trend with `epa_baseline_plays
      == 0` prints "no efficiency trend yet (too early in the season for
      a trailing-window comparison)" -- stated, not omitted, so a reader
      never infers silence means steady; the other null (no plays in the
      window) stays silent as before; an exact 0.0, should it occur, is
      "flat" not "down". `opportunity_score()` already skipped a null
      trend, so the weight mutes itself early in the season with no
      new knob. `weakness_reasons()` unchanged (`< 0` never fired on 0).
- [x] `src/rag/embed.py`: the chat's signal chunk sentence says the same
      thing, so `get_player_signals` never hands the model a zero trend
      to reason from.
- [x] Tests: 330/330 (was 322). Trend null + baseline 0 when the window
      is the whole season, real again when it isn't; the table carries
      the count; prose/chunk say "no trend yet" and "flat"; weakness
      reasons and the score ignore it.
- Validated live: a real `refresh --once` recomputed `signals_2026_
  week2.parquet` (313 rows, 313 null trends, 313 baseline-0) and
  re-embedded both leagues; the Fellowship start/sit report now opens
  every card with "no efficiency trend yet ...; red zone role share
  56%; target share ..." and Goff's chat chunk reads the same. The only
  remaining "+0.00" is a stale 2025 fallback row, now labeled "flat".
  From week 5 on the trend comes back by itself as the baseline fills.

## Fixed: chat lost every signal during a re-embed, and a league select could take a minute (the rebuild race)

Found 2026-09-20 in real desktop use, one day after the first hosted
prep. Three symptoms Rohan reported; two shared a cause.

**Symptom 1 -- "why does chat say it can't tell me Drake Maye vs Baker
Mayfield when we have week 1 data?"** Both QBs were in the week-2
signals table on disk (B.Mayfield 32 plays, cpoe +15.4; D.Maye 37
plays) and every RB/WR in the same answer had this week's numbers. The
chat tool reported `has_signals: false` for the two QBs anyway.
Reproduced the lookup afterwards (`_tool_get_player_signals` against
the same league's index): both resolved, both `has_signals: true`,
current. So the failure was transient -- and the mechanism was in plain
sight in `embed.embed()`: a rebuild was `delete(all ids)` then
`add(all chunks)`, which leaves the collection EMPTY for the whole
embedding pass (20-40s per league). Any chat that ran during the
scheduler's 6-hourly re-embed of that league saw whichever chunks had
been re-added so far. The `_resync_signals_if_changed` docstring even
admitted a reader "could still see a partial collection for that
moment" and called it small; it is not small once a cycle re-embeds
three leagues every six hours on a server people actually use.

**Symptom 2 -- "huge delay between selecting the league and the feed."**
Measured the warm path against a fresh server: `POST /api/sessions`
0.05s, roster 0.12s, the three reports 0.5-1.3s each. So the normal
case is fast, and the slow one is a rebuild running inside the
request: (a) the first time a league is ever opened (Sleeper pull +
full embed, about a minute, expected and now said so in the UI), or
(b) a request arriving while the scheduler's cycle had already
rewritten the signals table but not yet re-embedded and stamped THAT
league -- the request-path resync saw a stale stamp and ran its own
full rebuild, concurrently with the cycle's, interleaving writes into
the same collection, on the user's clock. The cycle at 14:36 local
covered two leagues; Narcos was stamped at 14:40 by a request.

**Fix, three parts.**
- `src/rag/embed.py`: a rebuild is now `upsert(new chunks)` then
  `delete(ids no longer present)`. Every chunk id is deterministic
  (`team:<roster_id>`, `signal:<season>:week<N>:<player_id>`, ...), so
  the end state is identical to before, but a concurrent reader sees
  at worst a mix of old and new text -- never nothing. Plus a
  per-directory `rebuild_lock()` that `embed()` holds, and
  `rebuild_in_progress()` for callers who would rather not wait.
- `src/scheduler/refresh.py`: `run_cycle()` sets a process-wide
  `cycle_in_progress()` flag for its duration.
- `src/api/leagues.py`: the request-path resync stands down (serves
  what is on disk, leaves the stamp for the rebuilder) when that
  collection is already being rebuilt or a cycle is running. The
  cycle re-embeds and stamps every ingested league itself; the
  request path is for the out-of-process (cron `--once`) and
  ingested-mid-cycle cases, which still work as before.

**Symptom 3 -- "why does Dylan Sampson have stale data?"** Correct
behavior, badly worded. nflverse's 2026 play-by-play has zero plays
with Sampson as rusher or receiver in week 1 (CLE's carries: Judkins
12, Watson 6, Concepcion 3, Sanders 1), so he has no current-season
row and Phase 3.6's prior-season fallback kicked in exactly as
designed. The label said "no current-season signal yet", which reads
like a pipeline problem. It now says why: "no plays recorded for this
player this season before the as-of week" (`ranking.py`'s prefix and
the chat tool's) -- so a rostered RB showing last year's numbers in
week 2 tells the user the actually useful thing, that he hasn't
touched the ball.

**Also in this pass (frontend):** chat now uses the full content width
on desktop (it was capped at 680px, leaving two thirds of a 1900px
window empty); the model's light markdown (`**bold**`, `- ` bullets,
paragraphs) is rendered instead of showing literal asterisks -- escape
first, then convert only those three forms; the league-select message
says the first open of a league takes about a minute and later ones
are instant.

**Validated:** `tests/test_embed.py` -- a rebuild updates changed
chunks and prunes removed ones; a reader polling a real Chroma
collection throughout a 150-chunk rebuild on another thread never sees
fewer than 150 (fails on the old delete-then-add code by construction);
`tests/test_resync_standdown.py` -- a stale league is rebuilt when
nothing else is rebuilding it, and is NOT when its lock is held or a
cycle is running, with the stamp left for the rebuilder; the cycle flag
is set only while `run_cycle` runs and is cleared on an exception.
Headless Chrome at 1900px: `#chat-scroll` no longer has a max-width and
fills the panel; `md()` renders bold/bullets/paragraphs and keeps
injected tags escaped. The exact production sequence (a chat during a
live cycle's re-embed) is not reproduced end-to-end here -- the
concurrent-reader test is the unit that pins it.

## Fixed: start/sit recommended one starter per position in a league that starts two (and never ranked a QB)

Found 2026-09-20 by the first friend to use it (league "Narcos", PPR,
roster_positions QB/RB/RB/WR/WR/TE/FLEX/FLEX/FLEX/K/DEF). Rohan: "why
does it say to only start 1 RB? ... he allows 2 RBs to start so we
should have a rec on what 2 RBs to start". Phase 3.5 had documented
start_sit as "groups by position, does not model roster_positions" and
called it a simplification. For a two-RB league it is a wrong answer:
the report benched the second-best RB.

**What changed (`src/reasoning/report.py`).** start_sit reads the
league's `roster_positions` (already on `LeagueConfig` since 5.1).
Each position with N dedicated slots gets its top N ranked players as
`recommended_starters` and the rest as `alternatives_considered`;
FLEX-type slots (FLEX, SUPER_FLEX, REC_FLEX, WRRB_FLEX -- the map is
`src/rag/lookup.py`'s `FLEX_ELIGIBILITY`) are then filled from what the
dedicated slots left over, ranked together across positions by the same
`rank_candidates()` call. Greedy (dedicated first, then flex), not a
global optimizer -- with one score per player the two only differ when
a player is worth more in a flex slot than a weaker teammate in a
dedicated one, which a single score can't express anyway. `slots` and
(for flex) `eligible_positions` are on each entry; `recommended_starter`
stays as the top pick -- it is the verdict Chat's `rank_players` is
held to (`tests/test_verdict_alignment.py` unchanged). A position with
no more ranked players than slots is skipped as before (nothing to
decide); a league with no known slot structure falls back to one slot
per position, the old behavior. Real Narcos result: RB ×2 = Gibbs,
Henry; WR ×2 = Evans, Nabers; TE = Ferguson; FLEX ×3 = Coker, White,
Washington.

**Two more gaps the same league exposed, fixed alongside:**
- *QBs were never rankable.* `opportunity_score()` scored EPA trend,
  red-zone share and target share; a QB's row has none of the last two
  (and no trend until week 5), so both QBs were "no usable signal" and
  the QB slot was always skipped -- which is also why Chat's
  `rank_players` said `insufficient_data` for Maye vs Mayfield. Passer
  rows (the ones with `cpoe`, NGS completion % over expected) now add
  `CPOE_WEIGHT` x cpoe + `IMPLIED_TOTAL_WEIGHT` x implied total, scaled
  into the same ~0-2 range as the skill terms; `score_description` says
  so and says a QB's score is not comparable to a skill player's (no
  superflex call from this). `fmt_signal_row` prints the cpoe. Narcos:
  Mayfield (+15.4 cpoe, 25.0 implied) over Maye (+3.2, 23.0).
- *Players who cannot play were ranked as if available.* Dylan Sampson
  was a SIT card with 2025 numbers -- Sleeper has him on IR, which is
  also why he has no 2026 touches. `get_my_roster` now returns Sleeper's
  `injury_status`; start_sit leaves Out/IR/PUP/Sus/COV/DNR/NA players
  out of the decision and names them in `notes` ("Not available this
  week, left out of the lineup: Dylan Sampson (IR), Ja'Kobi Lane (IR),
  Kendre Miller (Out)"). Questionable/Doubtful stay in -- that is the
  call the user wants help with -- and every entry carries
  `injury_status` so the card shows it.

**Roster tab, Sleeper-style (Rohan: "mirror sleeper's version of showing
a roster").** `/api/roster` now also returns `lineup` (the starting
slots in league order, each with its player or null -- Sleeper's
`starters` array lines up index-for-index with `roster_positions` minus
bench-type slots, "0" for an empty slot), `bench` and `reserve`, via
`lookup.starting_lineup()`; player entries carry `injury_status` and
`number`, and a team DEF ("JAX") is handled. The Roster tab draws
STARTERS and BENCH as two cards (side by side on desktop) with a colored
slot badge per row (QB amber, RB blue, WR green, TE purple, FLEX teal),
initials ring, name with a Q/O/IR tag, "POS · TEAM · #". The flat
`players`/`counts_by_position` are unchanged for anything else reading
them. The feed's start cards now show the starter's own signal line
(not the entry's whole comparison paragraph) with an "RB ×2"-style slot
chip -- which is most of what the backlog's "Feed card redesign" item
asked for; the position-filter chips from that item are still open.

**Validated:** `tests/test_report.py` (RB/RB/FLEX with four RBs → two RB
starters, third in FLEX, fourth sits; exactly-as-many-as-slots → nothing
to decide; no slot structure → one per position; IR excluded with the
note, Questionable kept and flagged); `tests/test_ranking_trend.py`
(passer score, two-QB ranking); `tests/test_lookup.py`
(`starting_lineup` alignment incl. an empty slot, a DEF, bench, reserve);
`tests/test_api_main.py` (`/api/roster` lineup/bench/reserve). And
against the real Narcos league on a live server in headless Chrome:
the entries above, the notes above, the roster screenshot (starters by
slot, bench, IR/O tags) -- no console errors.

## Fixed: a 5-play player topped the waiver list, and three report notes read like bugs

Found 2026-09-20 by Rohan on the Narcos feed: "why did we recommend
Tahj Washington here? what's his data?" and "peep the notes section,
why did those not land?".

**Tahj Washington #1 waiver target (score 2.739).** His row: 5 plays in
all of 2025, EPA trend +1.35, 1% target share, 1% red-zone share -- and
no 2026 plays at all (stale fallback). Two things went wrong at once:
- *The EPA trend term had no sample floor.* A trailing-window average
  minus a season average on five plays is noise with a large absolute
  value, and at 2.0x it was the entire score (2 x 1.35 = 2.70 of 2.74),
  above every player with an actual role. #2 and #3 (Kinsey, 4 plays;
  Walker, 8 plays) were the same shape. `ranking.opportunity_score()`
  now counts the trend only from `MIN_TREND_PLAYS` (20) plays on record
  (`trend_is_trustworthy()`); below that `fmt_signal_row` prints it as
  "efficiency trend +1.35 EPA/play on only 5 plays (too few to count)"
  rather than "trending up". `score_description` says so. The 2026
  week-2 table has no such rows (0 of 313); the 2025 season-end table
  has 5 of 601, all of which were waiver-list material.
- *Last-season-only players were ranked as pickups.* Once the current
  season has a table with anyone in it, a player whose only row is last
  season's has by construction no plays this season, and a pickup with
  no role now is not a pickup. `_waiver_pickups_report` sets them aside
  with a note ("282 unrostered player(s) with no 2026 plays yet were not
  ranked as pickups -- a role at the end of last season says nothing
  about a role now") whenever `tables.signals_by_id` is non-empty;
  before the season's first table exists, last season is all there is
  and is still used, labeled stale (Phase 3.6, unchanged). Real Narcos
  list afterwards: Elic Ayomanor, Evan Engram, Deshaun Watson, Eli
  Raridon -- all on 2026 usage.

**The notes.** Three notes on the same feed, each true, each reading
like a bug:
- "Could not identity-resolve 1 rostered player(s): Jake Bates" -- he is
  a kicker; nothing in the signals table covers K or DEF, so there was
  never anything to resolve against. `_resolve_roster_with_signals` now
  sets K/DEF aside explicitly: "No matchup signals exist for kickers or
  defenses, so they aren't ranked: Jake Bates (K)."
- "2 player(s) ... fell back to stale 2025 data: Roschon Johnson, Dylan
  Sampson" AND "1 rostered player(s) had no computed signals and were
  excluded from ranking: Roschon Johnson" -- both true (Johnson: 2 plays
  in 2025, none of the scored signals), and together a contradiction.
  The stale note now covers only players who can actually be ranked, and
  the unrankable one is explained once with his usage: "Not enough usage
  on record to rank (no target share, red-zone share, efficiency trend
  or passing numbers): Roschon Johnson (2 play(s) in 2025)." Same rule
  in start_sit and drop.

**Validated:** `tests/test_ranking_trend.py` (the Washington row scores
0.04 not 2.74, prints "too few to count", and the same trend with 40
plays counts as before); `tests/test_report.py` (last-season-only
players skipped with the note once the season has data; still listed
and labeled stale before it does -- the old "marks stale candidates"
test rewritten to that split; the K note; the one-story note for an
unrankable player). Suite 367/367. And the three real Narcos reports
above from a live server.

## Eval numbers, first real runs (2026-09-20): retrieval 80/84, decision run interrupted

Rohan okayed spending up to $30 of Claude credit on the decision eval,
so both harnesses ran for real for the first time.

**Retrieval (free, `evals/run_eval.py`): 80/84 = 95.2% across all four
ingested leagues** (Victorious Secret 3.0 ingested to a scratch dir,
plus the three per-league dirs), roster + matchup-score questions, as-of-
week filtered. Saved as `evals/results/2026-09-20_retrieval_run.json`.
The first pass scored **54/84, one league 6/24** -- and the diagnosis was
a product bug, not an eval quirk: since Phase 2 each league's index is
~97% per-player signal chunks (1,331 of 1,367), which sit close to any
football question in embedding space, so "Who is on X's roster?" came
back as three signal chunks and no roster, and after the as-of cut a
matchup question was left with one hit. The chat's `search_league_info`
tool ran the same unfiltered search despite its own description ("not
for player signals"). Fix: `retrieve.query()` takes an optional Chroma
`where`, `retrieve.LEAGUE_INFO_ONLY` (`type != player_signal`) is what
the tool passes, and `run_eval.query_as_of` applies the same filter so
the number measures the path a user's question actually takes. The
unfiltered default is unchanged (other callers and tests rely on it).
Pinned by `test_search_league_info_never_returns_player_signal_chunks`.

**Decision (`evals/run_decision_eval.py`): run, not finished.** Pilot
of 5 dilemmas: 2/5, $0.037 per dilemma (2 API calls each, Sonnet 4.5) --
far cheaper than the $0.05-0.15 guessed earlier, so the run was sized at
600 of the 1,870-dilemma pool (2024 week 5, `build(max_pairs=10000)`,
seeded shuffle, 4 parallel shards of 150). All four shards died 21
minutes in with `Your credit balance is too low to access the Anthropic
API` -- roughly 460 dilemmas answered (~$17) and **every one of them
lost**, because `run()` buffered results in memory and only wrote at
the end. That was the harness's flaw and this session's mistake to run
it that way. Fixed: `run(progress_path=...)` appends each scored dilemma
as a JSON line the moment it lands and skips those on a rerun (CLI
`--progress`, default `evals/results/decision_progress.jsonl`); pinned
by a test that crashes mid-run and resumes. The landing band now shows
the real retrieval number and still labels decision accuracy as pending.
- [x] **Rerun done (2026-09-20, after Rohan topped up):** 400 dilemmas
      in 4 shards with progress files, one mid-run hang (stuck API
      connections after 292; killed and resumed with a 120s request
      timeout, nothing lost). **Decision accuracy 206/399 = 51.6%**
      (one unscoreable answer). By position: QB 29/53, RB 61/115, TE
      18/34, WR 98/197. By the pair's actual point gap: <5 pts 93/199
      (47%), 5-10 56/108 (52%), 10-20 48/79 (61%), 20+ 9/13 (69%).
      Half the pool is decided by under 5 points -- coin flips by
      construction, since `build()` pairs any two same-position players
      who both scored 8+, and it never looks at margin (by design, so it
      can't bias toward the winner). Reported as measured; the 10+ pt
      subset (57/92 = 62%) is the more meaningful read of whether the
      signals say anything, and it is the baseline Phase 4/6/7 signal
      work has to beat. ~$15. Saved as
      `evals/results/2026-09-20_decision_run.json`; landing band and
      README updated. Old text of this item:
      once the Anthropic account has credit again, rerun with the
      progress file. The exact shard files are
      in this session's scratch dir; a fresh run is
      `python -m evals.run_decision_eval --league-id 1389341490030862336`
      after ingesting that league to `data/raw/sleeper` + embedding, or
      the scratch-dir variant described in the method field of the
      results JSON. ~$0.04/dilemma; 400 dilemmas ≈ $15 (±5% at 95%).
- [x] Decision number on the landing band + README; matchup-fit stays
      a placeholder until Phase 4.
- [x] **Deployed (2026-09-20):** `fly launch` / volume / secret /
      `fly deploy --remote-only` from this session after Rohan logged
      flyctl in from a real terminal (`! fly auth login` in Claude Code
      is non-interactive and fails). First-ever image build succeeded
      (195MB); machine healthy at https://askmadden.fly.dev, `/api/health`
      shows the in-process refresh completed a cycle on the fresh
      volume. `fly launch` rewrote fly.toml (quote style + stripped
      comments only); the commented version was restored. Domain:
      askmadden.com bought on Cloudflare Registrar; `fly certs add` done
      for the apex and www; the four DNS records (A/AAAA for @ and www,
      proxy OFF) are Rohan's to add; a poller is watching for issuance.

## Phase 6 built, plus the two accuracy strategies: points-per-game in the ranking, weights fitted on history (2026-09-21)

Rohan, after the first hosted use: "can we address this?" (the agent
correctly refusing a trade proposal because nothing computed player
value) and "for the strategies to increase the accuracy can you
implement the first and second one?" (points per game this season and
last; fit the weights instead of guessing them). They turned out to be
one build: PROJECT_SPEC.md's Phase 6 signal -- fantasy points scored so
far this season under the league's own scoring -- is exactly the
per-game value the ranking was missing.

**1. League-agnostic stat lines, scored per league at query time.**
`src/signals/player_stats.py`: one row per player per regular-season
week with the raw counting stats (yards, TDs, receptions, INTs, 2-pt,
fumbles lost) from nflverse's weekly player stats, written to
`data/processed/player_stats/player_stats_<season>.parquet` by every
refresh cycle (this season and last, rewritten each time). No points in
it -- this layer never sees a league, and `test_league_module_is_not_
imported_by_signals_or_rag` enforced that the moment a draft put the
scoring code there. `src/reasoning/points_proxy.py` is the per-league
half: `fantasy_points()` (the mapping `evals/build_ground_truth.py` has
used since Phase 1, which now imports it from here) and
`season_points_proxy()` -- games played, points so far, points per game,
strictly from weeks before the as-of week. `ranking.SignalTables.load()`
takes the league's `scoring_settings` and joins `ppg` / `games_played` /
`season_points_so_far_proxy` / `ppg_prior_season` onto every row; a
player with points but no usage row gets a proxy-only row (rankable,
labeled "no usage/matchup signals computed yet"). Stale prior-season
fallback rows now have their last-week matchup fields (opponent, implied
total, run-funnel lean) nulled -- they described last season's final
game, and since the refit those terms count for everyone.

**2. Weights fitted on history (`evals/fit_ranking_weights.py`).** A
no-intercept pairwise logistic model on feature differences (a linear
score by construction) over 19,045 same-position pairs from every 2023
week (both players >= 5 pts that week, features strictly as-of the
week), tested on 19,283 pairs from 2024, numpy only. Held-out results:

| score | 2024 pairs | weeks 2-5 | gap 10+ pts |
|---|---|---|---|
| hand-set weights (pre-fit) | 57.6% | 57.3% | 66% |
| fitted, points shrunk toward last season (adopted) | **61.8%** | 58.7% | 72% |
| fitted, unshrunk | 61.4% | 58.5% | 72% |
| points per game alone | 61.5% | 57.5% | 72% |
| last season's points per game alone | 58.1% | 55.9% | 65% |

The honest reading: points per game carries nearly all the signal; the
usage terms mostly earn their place by making the verdict explainable
(and the EPA trend came out slightly *negative* on held-out data -- a hot
trailing window does not predict next week). The adopted variant blends
this season's ppg with last season's as if last season were 4 games
(`ranking.blended_ppg()`), because the unshrunk fit put a one-game 33.8
(Jalen Coker) over Malik Nabers on the real Narcos roster. Weights and
`SCORE_DESCRIPTION` in `ranking.py` are from
`evals/results/2026-09-20_ranking_fit_blend4.json`; the unshrunk run is
alongside. Rerun: `python -m evals.fit_ranking_weights --blend 4`.
The offline number is what the ranking can do; the live decision eval
(52% on 400 dilemmas, before this) also carries the model's own noise
-- rerunning it is ~$15 and the natural next check.

**3. Trade proposals, labeled (Phase 6's prompt half).** `get_player_
signals` and `get_league_rosters` now carry the proxy per player (with a
`points_proxy_note`), and the system prompt's TRADES section lets the
agent make rough suggestions from tool output only -- "your Goedert,
11.2 ppg, for their Herbert, 18.4 ppg, is not close" -- with the label
in the recommendation itself: a points-per-game comparison, how good a
player has BEEN, not a projection, not position-scarcity-adjusted, not
injury/schedule/other-manager-adjusted, not a market value. Draft picks
still have no value here and stay a data_gaps entry. Phase 3.7's rule
survives in its Phase 6 form: every number must come from a tool call
this turn, never reputation.

**Also:** waiver targets are now ranked within position and interleaved
RB/WR/TE/QB with a `position_rank` -- once points entered the score, a
raw cross-position sort returned five backup QBs as the top pickups.

**Validated:** suite 383/383 (`tests/test_player_stats.py`,
`tests/test_points_proxy_join.py` -- as-of cut, per-league scoring, a
proxy-only player, half- vs full-PPR flipping a start/sit, the tools
carrying the proxy, the refresh writing next to the signals; ranking
tests reference the constants; the Phase 5.1 scoring-independence test
still holds *without* stat lines, and its scoring-dependence
counterpart holds with them). Real Narcos start/sit under the blend:
QB Mayfield; RB Gibbs + Henry; TE Ferguson; WR Coker + Nabers; FLEX
Evans, Washington, Schultz. The first test run reached the network for
stat lines and wrote real tables into `data/processed/player_stats`
(then leaked real points into fixtures); `tests/conftest.py` now stubs
the fetch for every test.
- [ ] Rerun the live decision eval against the fitted ranking (~$15,
      400 dilemmas, progress file) and compare with 52%.
- [x] Real-model check on the hosted app, same day, Dynasty of Chips,
      same two-turn question: the agent now returns four concrete
      proposals with the numbers from get_league_rosters ("Your Dallas
      Goedert (21.7 ppg week 1, 10.34 ppg last season) for their Ladd
      McConkey (16.7 ppg week 1, 9.24 ppg last season)... They only have
      2 TEs vs your 4"), `data_gaps: []`. Two prompt-following gaps: it
      never addressed the "2nd round pick" half (should have been a
      data_gaps entry), and the "crude proxy" caveat did not appear in
      its text. The caveat is now attached by the API deterministically
      (`points_proxy_note` on any turn that called get_league_rosters,
      rendered under the answer) rather than hoped for; the pick gap is
      still prompt-only.

## Fixed: a pre-draft league was told to target Jahmyr Gibbs on waivers; trade pitches now carry the other manager's situation (2026-09-21)

**"It's telling me to target Gibbs in free agency."** A third friend's
league, "Neal in the Endzone" (1389359006669082625). Diagnosed on the
live volume with `fly machine exec`: Sleeper status `pre_draft`, eight
teams, zero players on every roster. So his roster really was empty
(no start/sit, no drop candidates) and "unrostered" was the entire NFL,
which the waiver report ranked honestly and uselessly. Not a loading
problem -- the app was right and silent about why. Fix: `LeagueConfig`
carries Sleeper's `status`; `/api/sessions` returns it; for a
`pre_draft` league `generate_report` returns no start/sit or drop
entries with `PRE_DRAFT_NOTE`, and the waiver report keeps its list but
says it is the full pool ranked -- a draft board, not waivers; the Feed
shows a "league hasn't drafted yet" card at the top. Draft-day use
becomes a (small) feature rather than a bug. (Finding the league took a
detour: Aayush isn't in Victorious Secret 3.0 under that name, and the
username guesses that resolved on Sleeper belonged to other people; the
volume listing is what found it.)

**Trade pitches with reasons.** Rohan: "why can't we implement this?"
about the caveat's list of what the proxy ignores. Everything on that
list except a real market value was already in league data:
`get_league_rosters` now gives each team `starting_slots` (dedicated,
from the league's roster_positions; FLEX-type slots count for nobody),
`healthy_by_position`, `needs` (healthy players <= slots) and `surplus`
(>= slots + 2); each player carries Sleeper's `injury_status` (Out/IR/
PUP/Sus/COV/DNR/NA don't count as healthy), `bye_week` from the
nflverse schedule (`nflverse.bye_weeks()`, lazy on the context, empty
on failure) and `on_bye_this_week`. The TRADES prompt section tells the
model to pitch with those reasons ("they start 2 RBs and have 5
healthy ones; they're down to 1 healthy TE"), to say "on 1 game" when
that's the sample, never to propose acquiring an Out/IR player without
saying so, and to invite refinement ("no tight ends", "only from teams
that need a QB", "avoid week-7 byes") so the chat becomes the
springboard. Still not modeled, and still in the caveat: position
scarcity, remaining schedule strength, market value.

**Offline test on 2025** (Rohan asked): the adopted weights, trained on
2023 only, score 59.9% on 19,674 held-out 2025 pairs (70.6% at 10+ pt
gaps) vs 57.6% for the old hand weights and 59.5% for ppg alone;
60.8% across 2024+2025 combined (38,957 pairs). Consistent with 2024,
a point lower. Weights unchanged.

**Live decision eval rerun against the fitted ranking: 209/400 = 52.2%**
(was 206/399 = 51.6%), ~$15, `evals/results/2026-09-21_decision_run.json`.
By gap: <5 pts 48%, 5-10 56%, 10+ 58%. Flat -- and the diagnosis is the
useful part:
- The agent picked the deterministic ranking's player on **400 of 400**
  dilemmas (the verdict guard from the Chat/Feed alignment fix is doing
  its job). So the live number IS the ranking's accuracy on this set,
  plus nothing. A paid rerun measures the ranking with extra noise and a
  bill; the free offline harness (38,957 pairs across two seasons) is
  the measurement, and the live eval's only remaining job is to check
  that the agent follows the ranking and handles gaps -- which 400/400
  now answers.
- Why flat when the offline gain was ~4 points: the eval is ONE week
  (2024 week 5, both players >= 8 pts). Scored offline on every such
  pair that week (1,875): fitted 53.1%, old hand weights 54.1%,
  points-per-game alone 52.4% -- a hard, noisy week where no scorer
  does well, and where the hand weights happen to edge the fit by one
  point (inside the noise). Across all 2024 weeks the same fit is 61.8%
  and across 2025 59.9%. The 400-dilemma subset (52.2 / 51.6) is simply
  representative of that week.
- Next, when it's worth doing: extend `evals/ground_truth.jsonl` to
  every 2024 week (`build_ground_truth.py` already takes a week list)
  so `build_decision_questions` samples across the season, and report
  the live eval on that -- it would then track the offline number, and
  it never needs to run at volume again because the agent follows the
  ranking. Not started.

**Also:** the iMessage/Slack preview title is "Ask Madden: Your
League's Cheat Code" (the em-dash form was showing as just the
tagline).

**Validated:** suite 390/390 (`tests/test_predraft_and_trade_context.py`:
pre-draft notes for all three reports, in-season unaffected, bye weeks
from a schedule slice, needs/surplus/injury/bye on the roster tool, the
prompt phrases; `/api/sessions` carries status).

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

5.1, 5.2, 5.3, 5.4 and 5.7 are implemented (see their sections
below). 5.5's landing copy is done and its eval band is still gated on
eval runs at volume; 5.6's "one app serves everything" step and the
deployment config are done, the actual deploy is not (see 5.6 below).
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

### Signals refresh on a running server (investigation + fix)
Prompted by a manual-testing question, not a known bug: after
re-running `src.ingest.nflverse` + `src.signals.matchup_signals` for a
new week while the API server keeps running, does the server serve the
new signals, or stale data from the per-league Chroma client PR #24
warms? Investigated with real repros before assuming either answer.
Full `pytest` suite 274/274 at the time (266 before this unit + 8 new in
`tests/test_api_signals_refresh.py`). One file changed under src/:
`src/api/leagues.py`. **Superseded in part by the merge write-up at the
end of this section** -- PR #28 landed first, so the "separate defect,
not fixed here" notes below are now historical: both halves live in
`src/api/leagues.py` today.

**The suspicion was half right, and which half matters.** chromadb has
two read paths and they behave differently under `warm_chroma()`; the
first draft of this entry claimed the warming was wholly innocent, which
was wrong and is corrected here (caught by PR #28, then re-verified
directly rather than taken on faith):
- **`collection.get()` -- metadata lookups, i.e. `query_player_signal`
  and so `get_player_signals`** -- reads SQLite directly and DOES see
  another process's rewrite on the next call. Reproduced: embed in a
  subprocess, warm + read here, re-embed different numbers from a SECOND
  subprocess, read again -- 11.10% -> 99.90%, no restart. This is the
  path this unit's gap lives in, and it needed no cache fix. Pinned as
  `test_warmed_client_metadata_lookups_see_another_process_rewrite`.
- **`collection.query()` -- semantic search, i.e. `retrieve.query` and
  so `search_league_info`** -- answers from a per-process in-memory
  vector index that a warmed process does NOT refresh. Reproduced: after
  an out-of-process re-embed the warm process returns the OLD ids with
  `documents`/`metadatas` of `None`, so a caller doing
  `r["metadata"].get("type")` raises `AttributeError` (PR #28 measured
  this as a real HTTP 500 on `/api/chat`). That is a genuine defect, but
  a separate one: it is neither introduced nor fixed here, and the resync
  below re-embeds IN-process, which leaves that process's own semantic
  index correct (asserted in the same test). PR #28's mtime/size stamp on
  `warm_chroma()` is the fix. Pinned as a known limitation in
  `test_warmed_client_semantic_queries_go_stale_after_another_process_rewrite`,
  which says in its docstring to delete it once that lands.

**What the three layers actually do**, each reproduced end-to-end
through `TestClient` against the real endpoints with the league already
ingested and the server already warm:
- **Reports (parquet): fresh, no restart needed.**
  `ranking.load_signals_table()` re-globs `signals_dir` on every single
  call -- there is no cached DataFrame anywhere. Asking
  `GET /api/reports/drop?as_of_week=<new week>` right after the refresh
  returns the new numbers from the same running process.
- **Reports (default week): pinned to Sleeper, and that is correct.**
  `generate_report()` infers `as_of_week` from
  `recommend._infer_season_and_week()`, which reads the league's
  `nfl_state.json` -- written by the *Sleeper* ingest, which a
  signals-only refresh doesn't re-run. So the default report keeps
  showing the old week, because the newer rows are future data relative
  to the league's own state. That is CLAUDE.md's as-of-date rule
  working, not a cache, so it is documented rather than "fixed":
  inferring the current week from whatever happens to be in the signals
  directory would be a weaker definition. Pinned as
  `test_default_report_stays_pinned_to_the_leagues_own_state_week` so
  changing it is a deliberate act. (Still true. But the FIELD it read
  was wrong: Sleeper's `display_week` lags `week` until midweek, which
  made every card stale the day after real games -- see "Fixed: the day
  after real games, every feed card was [STALE -- 2025]" above.)
- **Chat (Chroma): genuinely broken for a different reason, and a
  restart would NOT have fixed it.** (Distinct from the semantic-index
  staleness above -- that one a restart *would* fix; this one it would
  not.) Chroma is only ever written by `embed()`, and
  `ensure_league_data()` skipped ingest entirely whenever
  `is_ingested()` was true -- which stays true across restarts. So for
  an already-ingested league `embed()` never ran again and the
  refreshed week was never embedded at all: `query_player_signal(...,
  as_of_week=<new week>)` returned `None` forever, and
  `get_player_signals` would keep answering from the old week's chunk.
  The only escapes were re-running the embed by hand or
  `POST /api/sessions {"refresh": true}` (which also re-hits Sleeper).
  This is the reason a "just restart the server after refreshing"
  operational note would have been actively wrong.

**The fix** (`src/api/leagues.py`, ~40 lines): stamp each league's
Chroma directory with a fingerprint of the signals files its collection
was built from (name + size + mtime of every `signals_*.parquet`), and
have `ensure_league_data()` re-embed that league when the fingerprint
no longer matches. Local only -- it re-runs `embed()` against the
already-ingested `raw_dir`, never Sleeper. Serialized on a lock and
called from `ensure_league_data()`, which every league-scoped request
already passes through before touching Chroma, so a request arriving
mid-resync waits rather than reading a half-rebuilt collection.
Fingerprint-gated, so a warm server does not re-embed per request:
`test_resync_runs_once_per_refresh_not_once_per_request` asserts three
requests with no change cause zero re-embeds and three requests after
one refresh cause exactly one.

**One deliberate trade, found by breaking two existing tests.** A league
with no stamp at all -- i.e. every league already on disk before this
landed -- adopts the current fingerprint WITHOUT re-embedding, rather
than treating "unknown" as "stale". Treating it as stale would make the
first request after upgrading silently re-embed every existing league,
and it turned a partially-written `raw_dir` into a *failed* request
where today it is merely incomplete: `is_ingested()` only checks that
`league.json` exists, so a raw_dir missing `teams.json` passes it and
then blows up inside `build_chunks()` (exactly what
`test_ensure_league_data_does_not_re_ingest_when_data_exists` caught).
The cost is that a league whose signals were refreshed BEFORE this code
landed stays stale until the next refresh -- a one-time migration,
fixed by one `POST /api/sessions {"refresh": true}` (or
`python -m src.rag.embed`), documented here and in README rather than
left to be discovered. Pinned by
`test_a_league_with_no_stamp_adopts_it_instead_of_rebuilding`.
- [x] Reproduce the scenario end-to-end (already-ingested league, warm
      server, out-of-band refresh, no restart)
- [x] Locate the staleness precisely (nothing re-ran `embed()`; not the
      warmed client, not a cached DataFrame)
- [x] Close it cheaply (fingerprint + resync-on-demand), no scheduler
- [x] Regression tests, verified load-bearing: with the fix reverted,
      `test_chat_signals_are_reembedded_after_a_refresh` fails and the
      three tests documenting already-correct behaviour still pass
- Explicitly NOT built here: `src/scheduler/refresh.py`. Automating
  *when* the refresh runs is still the separate, already-tracked Phase
  3.5 gap; this unit only makes a manual refresh land correctly.

**Operational note (how Rohan refreshes signals from now on).** With
the server running, in another terminal:
```
python -m src.ingest.nflverse --season 2025
python -m src.signals.matchup_signals --season 2025 --as-of-week N
python -m src.ingest.sleeper          # only needed to move onto week N
```
No server restart. The first request after that is slower (it re-embeds
that league). The third command is what advances `nfl_state.json` so
the default reports actually move to week N; without it the reports
stay on the previous week by design, and only an explicit
`?as_of_week=N` shows the new numbers. Also in README.md.

One-time only, for leagues that were already on disk before this change:
do a single `POST /api/sessions` with `{"refresh": true}` for each (or
re-run `python -m src.rag.embed`) so their Chroma collection starts from
a known-good fingerprint. Every refresh after that is automatic.

#### Merging this with Phase 5.7 (PR #28) — what it actually required

PR #28 merged first (`bde259f`), so this branch was brought up to date
and the overlap in `src/api/leagues.py` resolved deliberately rather
than mechanically. **This closes the loop: trigger-side and read-side
freshness are both handled now**, and neither is redundant with the
other.

The two fixes are complementary halves of one guarantee:

| | what it fixes | without it |
|---|---|---|
| **#28 — read side** (`warm_chroma()`'s `(mtime, size)` stamp) | once `embed()` has re-run **by any means**, a warm server notices | an out-of-process re-embed is invisible; `collection.query()` returns removed ids with `documents`/`metadatas` of `None` → `AttributeError` → HTTP 500 on `/api/chat` |
| **#27 — trigger side** (`_resync_signals_if_changed()`'s fingerprint) | `embed()` re-runs **at all** for an already-ingested league | `is_ingested()` stays true forever, so a manual/fast/`ASKMADDEN_REFRESH_ENABLED=0` refresh is never embedded — and a restart doesn't help |

Git auto-merged the file cleanly (the two changes sit in different
regions), but a clean textual merge was not a correct one. Three things
needed doing by hand:

1. **A real defect in the combination, reproduced before fixing.**
   `refresh_league()` re-embeds each league every cycle but never wrote
   this unit's fingerprint stamp, so the stamp kept describing the
   *previous* signals table and the first request after **every**
   scheduler cycle hit a mismatch and rebuilt an already-current
   collection. Measured: the cycle embedded once, the next request
   embedded again — a full rebuild charged to whichever user's request
   landed first. Fixed with a new public
   `leagues.record_signals_fingerprint()`, called from refresh.py's
   `_note_reembedded()` (renamed from `_invalidate_server_cache`, which
   now does both post-embed jobs). Kept separate from
   `invalidate_chroma()` on purpose: that one is about this process's
   in-memory client (a no-op in a cron run), this one about durable
   on-disk state every future process reads.
2. **The limitation test became a regression test.**
   `test_warmed_client_semantic_queries_go_stale_after_another_process_rewrite`
   carried a docstring saying to delete it once #28 landed. Inverted
   instead of deleted, as
   `..._semantic_queries_are_fresh_after_another_process_rewrite`: a
   test proving the freshness we now depend on is worth more than the
   absence of a test documenting a limitation we no longer have. It
   still pins the *boundary* — querying without going through
   `warm_chroma()` is still stale, because chromadb's per-process vector
   index has no way to know — which is what keeps it visible *why* #28's
   fix sits in `warm_chroma()` specifically.
3. **The prose that said #28 was unlanded.** `leagues.py`'s resync note
   and the test module docstring both described the semantic staleness
   as "a real, separate defect, not fixed here, PR #28 is the fix".
   True when written, misleading now; rewritten as the read/trigger
   split above.

**Tests for the combination**, which neither PR had (each only tested
its own half):
- `test_a_refresh_reaches_semantic_search_on_a_warm_server` — refresh on
  disk → one ordinary request → the resync fires and semantic search
  serves the new chunks.
- `test_an_out_of_process_cycle_reaches_semantic_search_without_a_restart`
  — the Phase 5.6 cron shape, and **the only test where both halves are
  strictly required**. An in-process `embed()` goes through the same
  cached System, so that process's vector index is correct for free;
  that convenient property disappears once the scheduler is its own
  process. Verified by neutering each half in turn: without #28's stamp
  it fails with *"a warm server never saw the out-of-process re-embed"*,
  without the fingerprint recording it fails with *"re-embedded work the
  out-of-process cycle had already done"*.
- `test_a_scheduler_cycle_does_not_leave_a_redundant_re_embed_behind` —
  pins finding 1 above.

**Validation.** Full suite **319 passed** with both fixes present
together (308 on `main` after #28 + 11 in
`tests/test_api_signals_refresh.py`). PR #24's concurrency guard
explicitly re-run and still green (`test_api_concurrency.py`, 2 passed)
— three fixes now share this file's neighbourhood. Every new and
converted test checked load-bearing by neutering the fix it guards, not
assumed.

- [x] Merge `main` (post-#28) into this branch, resolve `leagues.py` so
      both mechanisms coexist
- [x] Reproduce and fix the redundant per-cycle re-embed the merge
      created
- [x] Convert (not delete) the known-limitation test
- [x] Test the combination end-to-end, in-process and out-of-process
- [x] Full suite + PR #24's concurrency guard green together

### 5.4 — PWA installability
Implemented. Static/frontend work only: four new files under design/
(manifest.json, sw.js, icons/ with a generator script and three PNGs),
a `<head>` + one registration block in design/askmadden-ui-mockup.html,
and a one-line bind change in web/dev_server.py. No file under src/
changed. Full `pytest` suite is 248/248 (unchanged -- nothing here is
covered by pytest; validation is browser-level, see below).

**What the investigation found before building (iOS first).**
- iOS Safari, verified against Apple's own Safari release notes (the
  JSON behind developer.apple.com, since the pages are JS-rendered and
  webkit.org/MDN/web.dev are blocked by this sandbox's egress policy)
  plus Apple's archived "Configuring Web Applications" doc: on iOS
  16.4-18, a site added to the Home Screen launches standalone only if
  it has either the manifest's `display: standalone` or the legacy
  `<meta name="apple-mobile-web-app-capable">`; the Home Screen icon
  is `<link rel="apple-touch-icon">` (Safari ignores manifest icons;
  falls back to a page screenshot if the link is missing); the label
  is `apple-mobile-web-app-title`. Safari 26's release notes say "Added
  support for any website to become a web app on iOS or iPadOS" -- on
  iOS 26 everything added to the Home Screen opens as a web app by
  default ("Open as Web App" toggle in the add sheet), so the tags stop
  being the gate but still control icon/title/status bar. The
  historical "tags in the HTML head, independent of the manifest"
  picture is still accurate for icon and title on every version, and
  for standalone launch on 16.4-18. Both mechanisms are set so every
  supported version behaves the same. No service worker or HTTPS is
  needed for the iOS install: it works over plain http://<LAN IP>.
- Android Chrome (secondary, per this session's scope): the install
  prompt needs a manifest with name/short_name, start_url in scope,
  display standalone, 192 + 512 icons, AND a secure context. A service
  worker with a fetch handler has not been required for installation
  since Chrome 108 (mobile) -- built anyway, per spec, for the
  precached launch. **Secure context is the catch for a LAN test:**
  `http://192.168.x.x:8000` is not one, so Chrome will not offer
  "Install app" there and `navigator.serviceWorker` is absent
  (registration is a silent no-op by design). Real Android testing
  needs either USB port forwarding through chrome://inspect (then the
  phone opens http://localhost:8000, which IS a secure context) or the
  HTTPS deployment in 5.6. Documented in PR #25's test instructions.
- web/dev_server.py bound to 127.0.0.1: unreachable from any other
  device, so "Add to Home Screen" was untestable as shipped. Now
  0.0.0.0 (still local-network-only).

**What was built.**
- [x] design/manifest.json: name/short_name "Ask Madden", `display:
      standalone`, `background_color` --bg (#05070a, so the Android
      splash matches the app's ground), `theme_color` --green
      (#39e39a), relative `start_url`/`scope`/icon paths so it keeps
      working when 5.6 moves the file under src/api/'s static mount,
      a stable `id`, 192/512 `any` icons + a 512 `maskable` entry.
- [x] design/icons/: `build_icons.py` renders the icon programmatically
      with Pillow -- --bg square, green "AM" in the real Anton face
      (fetched from Google Fonts at build time, never committed;
      DejaVu Sans Bold fallback offline), monogram inside Android's
      maskable safe zone. Outputs `icon-192.png`, `icon-512.png`, and
      `apple-touch-icon.png` (180x180, the size iPhones use).
- [x] design/sw.js: scope is the file's own directory (/ui/ today), so
      /api/ is never intercepted -- verified, see below. Cache-first
      for manifest/icons/Google Fonts; the HTML document is
      network-first with cache fallback, a deliberate deviation from
      "cache-first for everything" so an edit to the mockup (5.5 will
      edit it) shows on reload instead of waiting for a worker update.
      Versioned cache name, old caches dropped on activate. ~80 lines.
- [x] Mockup `<head>`: manifest link, theme-color, favicon,
      apple-touch-icon, apple-mobile-web-app-capable (+ the
      standard mobile-web-app-capable), status-bar-style "black"
      (matches --bg; "black-translucent" would need safe-area layout
      work, which this session must not do), apple-mobile-web-app-
      title. One registration block at the end of the existing
      `<script>`; the 5.3 wiring and layout are untouched.
- [x] web/dev_server.py binds 0.0.0.0.

**Validation -- exactly what was run, and what it proves.**
- Lighthouse 11.7.1 PWA category (the last Lighthouse line with a PWA
  category -- 12 removed it because Chrome folded those checks into
  DevTools; its `installable-manifest` audit is Chrome's own
  installability engine, not a lint) against the real dev server in
  the pre-installed Chromium 141, headless: **score 100**, all six
  automated audits pass (installable-manifest, service worker +
  start_url, splash-screen, themed-omnibox, content-width, viewport,
  maskable-icon); the three remaining audits are manual by design.
- A puppeteer-driven Chromium session against the same server: the
  worker registers with scope `/ui/`, precaches exactly the five
  assets, controls the page after one reload; a probe to `/api/...`
  went through to the real API (404 in the server's own log, i.e. not
  served from cache); the landing view still shows on load and
  `goView('login')` still works; all three icons load at their stated
  sizes.
- Bind: the kernel's listening socket for :8000 is `00000000:1F40`
  (0.0.0.0) in /proc/net/tcp, and all seven URLs were fetched over the
  sandbox's real non-loopback interface (192.0.2.2), 200s logged from
  that address. Not just read from the code.
- Server-side serving: manifest as application/json, sw.js as
  text/javascript, PNGs as image/png -- all from the existing
  StaticFiles mount, no server change beyond the bind.
- [ ] **Only Rohan can confirm, on a real phone -- iOS first.** NOT
      done before PR #25 merged, nor before PR #27 -- the box below was
      skipped, not passed. The first real-iPhone look came after both
      merges and found the phone-frame bug (see the "5.4 follow-up"
      entry below), so this check is now superseded by that entry's
      real-phone checklist, which covers the install AND the layout. Open
      `http://<Mac's LAN IP>:8000/` in iOS Safari on the same WiFi,
      Share -> Add to Home Screen; the sheet should preview the green
      "AM" icon and the name "Ask Madden"; the Home Screen tile should
      be the icon (not a page screenshot); launching should be
      full-screen with no Safari address bar, a black status bar, and
      a dark launch background. Then Android Chrome via USB port
      forwarding (see PR #25). Lighthouse says Chrome's engine deems it
      installable; nothing automated here can stand in for the actual
      iOS sheet.
- Not done, on purpose: `apple-touch-startup-image` splash screens
  (iOS wants one PNG per device size; the dark `background_color` and
  icon are enough for a portfolio install), CORS/static in src/api/
  (5.6), any layout change.

### 5.4 follow-up — the phone frame rendered inside real phones (fix + device-emulated audit)
Implemented. Frontend only: design/askmadden-ui-mockup.html (CSS, two
markup blocks, one line of JS). design/sw.js was reviewed and left
unchanged -- nothing in it contributed (the HTML is network-first, so
an installed app picks the fix up on its next launch with the server
reachable). No file under src/ changed. Full `pytest` suite: 319/319
before and after (nothing here is covered by pytest; validation is
browser-level, see below).

**First, the honest record: PR #25 (PWA installability) and PR #27
(signals-refresh trigger fix) both merged WITHOUT the real-iPhone
"Add to Home Screen" check in 5.4's checklist ever being run.** That
gate was skipped, not passed. The first real-device look at the app --
a screenshot from an actual installed iPhone PWA, after both merges --
is what found this bug. Every browser check before it (PR #23/#24/#25:
"58 checks at 390px and 1280px", Lighthouse 100) was a desktop headless
browser resized to 390px, which is not a phone and is specifically why
this never surfaced.

**The bug.** `.app-shell` was a fixed 390x820 phone illustration --
rounded bezel, box-shadow, a `.notch`, a fake "9:41" `.statusbar` --
built as a design-preview device for looking at the phone layout
inside a wide desktop window. The only breakpoint that stripped it was
the >=900px desktop one, so every real phone got the illustration too:
a phone drawn inside the phone, with broken scrolling and taps that
landed on nothing.

**What the investigation found (measured in the pre-fix audit, not
inferred).**
- There were exactly two CSS states: the base styles (the illustration)
  and `@media (min-width:900px)` (the desktop reflow). A real phone is
  under 900px, so it fell into the illustration. No third state existed.
- The shell did not fit: 390px + `#view-app`'s 20px side padding on a
  393px iPhone / 412px Pixel, so flex shrank it (Pixel 7: shell at
  x=20, y=40, 372x820 in an 412x839 viewport) and the 820px height plus
  80px vertical padding made the *document* scroll (scrollHeight 900 in
  an 839px viewport; 240px of document scroll on the iPhone) -- the tab
  bar moved with it (y 754 -> 514 after one swipe).
- Content could never scroll on a phone, not just "scrolled badly":
  `.main-col` only had its `display:flex; flex-direction:column` rule
  inside the desktop query, so below 900px `.app-content` had no bounded
  height to scroll within (clientHeight == scrollHeight == 1685px);
  `.screen`'s `overflow:hidden` simply clipped it. The chat input bar,
  at the bottom of the chat panel, sat at y=807 in a 660px viewport --
  unreachable.
- The header's `justify-content:space-between` had nothing to span:
  `.main-col` was a shrink-to-fit flex item (header width 3626px on the
  Pixel, clipped).
- `env(safe-area-inset-*)` could never be non-zero: the viewport meta
  lacked `viewport-fit=cover`.
- Two things a phone does that no desktop check models: iOS Safari
  zooms the page when a focused input's font-size is under 16px (the
  login input was 13.5px, the chat input 12.5px), and mobile browsers
  paint a translucent tap-highlight box over any tapped element with a
  click handler (the header's league row is a plain div -- visible in
  the emulated screenshot as a blue rectangle).

**Decision: retire the illustration, don't gate it.** There is no
viewport left where it is correct -- below 900px it is wrong on every
real phone (the only devices that narrow, other than a narrowed desktop
window) and at >=900px it was already stripped. Keeping it behind a
flag would preserve code with no user and keep the trap that caused
this bug. Git history has it. Detection is **width, not
`display-mode: standalone`**, on purpose: a phone opening the LAN URL
in plain Safari had exactly the same bug, so "installed" is the wrong
signal; the env() insets simply resolve to 0 outside standalone mode,
so one layout serves both.

**What was built (design/askmadden-ui-mockup.html).**
- [x] `.app-shell` is the viewport below 900px: `width:100%;
      height:100dvh` (dvh, not vh -- iOS Safari's collapsing toolbar
      changes the visible height and 100vh would hide the tab bar
      behind it), no border/radius/shadow/padding, `#view-app` padding
      0. `.notch` and `.statusbar` removed from CSS and markup.
- [x] `.screen` pads by `env(safe-area-inset-top/left/right)`; the
      viewport meta gained `viewport-fit=cover` so those resolve.
- [x] `.main-col` gets its flex-column rule in the base styles (the
      desktop query's copy is unchanged); `.app-content` is the one
      vertical scroller: `min-height:0; overflow-y:auto;
      -webkit-overflow-scrolling:touch; overscroll-behavior-y:contain`,
      bottom padding `90px + env(safe-area-inset-bottom)`.
- [x] `.tabbar-mobile` bottom is `14px + env(safe-area-inset-bottom)`;
      `.modal-sheet` bottom padding is `26px + env(safe-area-inset-bottom)`
      -- neither sits on Apple's home-indicator gesture bar.
- [x] `goView()` toggles `html.app-open`; below 900px that sets
      `html{overflow:hidden}` and `body{min-height:0}` so the document
      itself cannot scroll while the app view is open (body's
      `min-height:100vh` had left a toolbar's-height of document scroll
      under the shell). Scoped to <900px so desktop is byte-for-byte
      unchanged.
- [x] `@media (pointer:coarse)`: the two text inputs are 16px (iOS
      auto-zoom guard) and tappable rows/chips/tabs/buttons get
      `-webkit-tap-highlight-color:transparent`.
- [x] `@media (max-width:479.98px)`: the login/picker card gets 16px
      side margins (it had none below its own 420px max-width, so its
      border ran edge to edge).
- [x] Head comment on `apple-mobile-web-app-status-bar-style` updated:
      still "black" (opaque, content starts below the clock), which
      cannot put content under the status bar even if an inset comes
      back 0. "black-translucent" is now a one-line switch once a real
      iPhone confirms the insets.
- [x] Desktop (>=900px) untouched: the only edit inside that media
      query is deleting `.notch, .statusbar` from a `display:none` list
      (selectors that no longer exist).

**Validation -- exactly what was run, and what it proves.**
- Tooling: Playwright 1.63 (Python) driving the machine's installed
  Google Chrome 150 via `channel="chrome"` -- Playwright's bundled
  Chromium refuses to install on this macOS 12 machine ("does not
  support chromium on mac12-arm64"), and device emulation is a CDP
  feature so the installed Chrome is equivalent. Device descriptors
  are Playwright's own: **iPhone 14 Pro** (393x660 CSS px, DPR 3, touch,
  iOS Safari UA) and **Pixel 7** (412x839, DPR 2.625, touch, Android
  Chrome UA), plus a 1280x800 non-touch desktop context. Playwright's
  descriptors do NOT carry safe-area insets, so they were set through
  CDP `Emulation.setSafeAreaInsetsOverride` -- 59px top / 34px bottom
  for the iPhone (its documented values), 0 for the Pixel -- and
  confirmed to reach `env()` in the page (top 59, bottom 34). Every tap
  is `page.tap()` (touch events, not mouse clicks); vertical and
  horizontal scrolls are CDP `Input.synthesizeScrollGesture` with
  `gestureSourceType: touch`; flings are raw `Input.dispatchTouchEvent`
  start/move/end sequences with fast final moves, checking that the
  scroller keeps moving after touchEnd. The API boundary is a mock
  server (FastAPI, in the audit's scratch dir) at the same boundary
  tests/test_api_main.py mocks -- login, sessions, roster, the three
  reports (15 start/sit cards, 5 drops, 25 waivers so the lists
  overflow), and a chat that returns real `data_gaps` /
  `signals_consulted` shapes -- serving the actual mockup file from
  design/ at /ui like web/dev_server.py does.
- The same audit, run against the pre-fix file (origin/main) to prove
  it catches the bug: **iPhone 14 Pro 38 pass / 24 FAIL**, **Pixel 7
  42 pass / 20 FAIL**. The failures are the bug: shell not filling the
  viewport, notch + fake status bar present, 46px radius + shadow +
  border, document scrolling, `.app-content` not a bounded scroller,
  header 3626px wide, tab bar 8px from the bottom, horizontal swipe on
  the rec-card row moved nothing (scrollLeft 0 -> 0), vertical touch
  scroll moved nothing (scrollTop 0 -> 0) while the tab bar moved
  (document scrolled instead), long chat conversation could not scroll,
  chat input bar off-screen (bottom 807 in a 660 viewport), league
  sheet extending past the viewport (bottom 845 of 660), inputs at
  12.5/13.5px, login card edge to edge.
- Against the fixed file: **iPhone 14 Pro 63 pass / 0 fail / 2 n/a**,
  **Pixel 7 63 pass / 0 fail / 2 n/a**, **Desktop 1280x800 53 pass /
  0 fail / 2 n/a**. Every surface in the brief: landing (hero CTA, nav
  "Log in", no horizontal overflow), login (field focus, Continue,
  unknown-user error), picker (league card, paste-a-league-ID
  fallback), all four tabs each actually swapping the visible panel
  (with an `elementFromPoint` hit-test proving nothing overlays the
  button), Feed (touch swipe on the rec-card row, fling continues after
  finger lift, vertical touch scroll + fling on `.app-content`, tab bar
  stays put, document and view wrapper stay at 0, in-content "See all"
  link), Chat (focus, type, send, response with stale + no-signal
  chips, suggestion chip, 8-turn conversation overflows and
  touch-scrolls, input bar reachable above the tab bar at the bottom),
  Roster, Moves (Waivers/Trades segments both ways, composition-request
  button renders an out-of-scope chip and flips to "Ask again"), league
  switcher (opens inside the viewport, sheet padding clears the inset,
  ACTIVE pill, switching closes + updates the header + resets to Feed,
  ✕ closes), no console/page errors, no failed requests beyond the
  deliberate unknown-user 404. The 2 n/a are the brief's waiver "Add"
  button and Roster Starters/Bench control -- neither exists, see
  findings below.
- Desktop regression check, separate from the audit: 26 computed
  layout properties (shell/screen/sidebar/header/content boxes, tab bar
  display, radius/shadow/border/background, content padding and scroll
  extents, rec-card width and wrap, input font sizes, active-nav color,
  document scroll extents, sheet padding, flex directions) probed on
  the pre-fix and fixed files at 1280x800 and 1000x700: **identical on
  all 26 at both sizes**.

**Findings triage -- fixed here vs. flagged.**
Fixed here (all CSS/JS in design/askmadden-ui-mockup.html):
1. The phone-frame illustration on real phones (the reported bug).
2. Content could never scroll below 900px (`.main-col` rule missing
   outside the desktop query) -- the "broken scrolling" half of the
   report was this, not touch physics.
3. Document scrolled under the shell (body `min-height:100vh`).
4. No safe-area handling at all (`viewport-fit=cover` missing; tab bar
   and league sheet sat on the home-indicator area).
5. Inputs under 16px -> iOS auto-zoom on focus (login 13.5px, chat
   12.5px). Chrome never auto-zooms, so this fix is by the book, not
   verified by emulation.
6. System tap-highlight flash on the header league row / league cards
   / chips / tabs.
7. Login/picker card ran edge to edge under 480px.
Found, needs its own session (out of scope or bigger than this one):
- Waiver "Add" button: absent, and by design since 5.3 -- the API has
  no write-back ("nothing is ever written back to your team"), rows
  carry a priority number and score instead. If the product ever wants
  a real Add, that is a src/api/ + Sleeper write-scope decision, not UI.
- Roster Starters/Bench segmented control: absent since 5.3 because
  `/api/roster` has no starters/bench split. Sleeper's roster JSON does
  carry a `starters` array, so this is buildable, but it is a
  src/rag/lookup.py + src/api/ change (out of this session's scope).
- iOS keyboard vs. the chat input: in standalone mode the layout
  viewport does not shrink for the keyboard, so whether Safari scrolls
  the (in-flow) input bar into view above the keyboard is a real-phone
  question; no emulation models the iOS keyboard. Likely fine, unproven.
- `apple-mobile-web-app-status-bar-style`: "black" is kept. With it,
  iOS should report a 0 top inset and place content below the clock; if
  a real iPhone instead shows a doubled gap (opaque bar + 59px padding),
  switch to "black-translucent" (one line in the head) -- the layout is
  ready for either.
- Feed rec-cards are still walls of text at 393px (one card fills most
  of the viewport height in the screenshot) -- already in the Backlog
  as "Feed card redesign + position filters"; unchanged here.
- Landing page at phone width was only checked for overflow and
  tappability, not visual polish -- that is 5.5's job.
- If an installed PWA is launched while the dev server is unreachable,
  the service worker serves the last cached HTML; if that copy predates
  this fix, the illustration shows until the server is back. Network-
  first by design; no sw.js change made.

**What device emulation cannot stand in for (so TODO.md doesn't
overclaim).** Chrome's device mode gives a real viewport, DPR, touch
event stream, mobile UA and (via CDP) safe-area inset values. It does
not give: WebKit (every iPhone browser and the installed PWA are
WebKit, and the fixes lean on `100dvh`, `env()`, `overscroll-behavior`
and the 16px zoom rule being honored by Safari specifically); iOS's
actual safe-area values under each status-bar style; native iOS
scroll momentum and rubber-banding (the fling checks prove Chrome's
gesture pipeline flings inside the new scroller -- they say nothing
about iOS's physics); the iOS keyboard / visual viewport; the "Add to
Home Screen" sheet; the standalone launch itself. So: this audit proves
the layout is right for a phone-shaped, touch-driven viewport and that
the pre-fix file was wrong for one. It does not prove the iPhone
experience.

- [x] **Confirmed on a real iPhone (Rohan, 2026-09-16): the phone-in-
      phone rendering is gone** -- the first real-device pass this
      project has had. The items below it (safe-area gaps, home-screen
      launch, keyboard vs. chat input) are not yet individually
      confirmed.
- [ ] **Still needs a real physical phone for the rest of the list.** From the Mac: `python -m web.dev_server`, then on
      the iPhone (same WiFi) open `http://<Mac's LAN IP>:8000/` in
      Safari. Check in the browser first: log in, pick a league -- the
      app should fill the screen edge to edge with no bezel, the feed
      should scroll with your finger, the tab bar should stay put, all
      four tabs should switch, the chat input should NOT zoom the page
      when tapped. Then Share -> Add to Home Screen, launch from the
      Home Screen: no Safari chrome, header just below the clock with
      no doubled gap, tab bar floating above the home-indicator bar
      (not on it), league sheet's bottom clear of the bar, keyboard
      does not hide the chat input. Same walkthrough on an Android
      phone via USB port forwarding (chrome://inspect, per PR #25's
      notes) for the install path; the browser-mode layout on Android
      needs no forwarding.

### 5.5 — Landing page (front door)
Reframed from "static marketing site, separate from the app": the
landing page is now the `#view-landing` view inside the same
responsive design/askmadden-ui-mockup.html — the entry point users
see first, with a CTA into the login view. Still its own small,
independent piece of content work (copy, feature cards, the eval
band), but one view in one file, not a separate build target, and
it shares the app's design tokens by construction.
- [x] Landing copy/feature cards finalized (2026-09-18, with 5.6):
      reviewed against what the product actually does today and
      edited, not rewritten. Each feature card now names real things --
      the signals a card cites are the ones in the signals table
      (target share, red-zone share, efficiency trend, run-funnel
      defense, implied total; an earlier draft of this pass said "snap
      share", which no signal computes, and was corrected before
      commit), the chat card says data gaps are declared and prior-
      season numbers are labeled stale (Phases 3.6/3.7), the trade card
      still says composition-only (Phase 3.8, no valuation). The
      `<title>` is now "Ask Madden" rather than "Responsive UI" -- it is
      the browser-tab and installed-app title, not a mockup label
      anymore. Rohan should still read it once; "final" copy on a
      portfolio page is his call, and this pass only made sure nothing
      on it overclaims.
- [ ] Eval-numbers band (`#eval-numbers`): its three metrics are
      placeholders ON PURPOSE, labeled as such in the mockup, never
      invented. Retrieval accuracy and decision accuracy are gated on
      `evals/run_eval.py` / `evals/run_decision_eval.py` running at
      volume (no Phase 4 dependency). Matchup-fit accuracy is
      specifically gated on Phase 4's coverage classification landing
      — per the signals table's own "modeled proxy" note for that
      signal, there is no matchup-fit number to report until then.
      Populate each only when its real number exists. **Status
      2026-09-18:** still placeholders. Running the decision eval at
      volume is a real Claude API spend (one recommend() loop per
      dilemma) on Rohan's key, so it is his call when to run it and how
      many dilemmas; `evals/eval_questions.jsonl` and
      `evals/decision_questions.jsonl` are both generated, not
      committed, so a run starts with the two `build_*` scripts. The
      retrieval eval costs nothing but runs against one league's
      Chroma index, so its number should be reported for what it is.
- [x] No auth, no API calls from the landing view itself -- confirmed
      and pinned by `tests/test_api_static.py`: the landing view's
      markup contains no `api(`/`fetch(` call and every handler on it
      is `goView('login')`. (The service-worker registration runs on
      page load regardless of view; it is not an API call and it
      never sees `/api/` -- see 5.6.)

### 5.6 — Deployment
- [x] FastAPI mounts the one responsive frontend (landing + app in
      one file) as a static route — one deployment, one URL, no CORS
      (2026-09-18, see below)
- [x] Deployment config written: `Dockerfile`, `deploy/entrypoint.sh`,
      `.dockerignore`, `fly.toml`, `constraints.txt` (see below for
      what each does and why)
- [ ] Deploy to a host (Fly.io is the configured one; Railway/Render
      work with the same image) — needs Rohan's account, a payment
      method (no current free tier gives a persistent disk plus 2GB),
      and `flyctl`; the README's Deploying section is the copy-paste
      sequence. Not done: neither Docker nor flyctl is installed on
      the machine this was written on, so **the image has never been
      built** — the first `fly deploy` is the build test.
- [ ] Get 2-3 friends in different leagues to actually use it
- [ ] README: document the "started as one league, generalized to a
      product" story, with real eval numbers from run_decision_eval.py
      (the story is there; the numbers are 5.5's open item)

#### What was built (2026-09-18)
**`src/api/main.py` now serves the frontend itself.** `design/` is a
`StaticFiles` mount at `/ui`, `/` redirects to `/ui/`, and an explicit
`/ui/` route returns the HTML (registered *before* the mount, because
FastAPI matches in registration order and StaticFiles has no
`index.html` to answer a bare directory with). The `/ui` prefix is
kept deliberately rather than serving the page at `/`: `design/sw.js`'s
scope is the directory it is served from, so under `/` it would
intercept `/api/*`; under `/ui/` it can't. Every relative href the page
already used (manifest, icons, `sw.js`) resolves unchanged — that was
the point of making them relative in 5.4. The Phase 5.7 refresh thread
moved from `web/dev_server.py`'s outer lifespan into the API app's own
lifespan (same `ASKMADDEN_REFRESH_ENABLED` switch; still off in tests
because a plain `TestClient(app)` runs no lifespan, and the two tests
that do use `with TestClient(app)` stub the thread). `web/dev_server.py`
is now a thin launcher — it imports the same app and binds 0.0.0.0 for
the phone-on-WiFi case — so the documented `python -m web.dev_server`
still works, and so does `uvicorn web.dev_server:app`.

**`GET /api/health`.** For the host's health check and for "is the data
moving" at a glance: `{ok, version, refresh: {enabled,
running_in_process, last_outcome, last_finished_at, season,
as_of_week, consecutive_failures, next_run_after}}`, read from
`refresh_status.json`. Always 200 while the process is up — a health
check that restarted a healthy server because nflverse was down for a
cycle would make things worse, so a failed cycle is reported, not
treated as an outage.

**The deployment files.** `Dockerfile` (python:3.11-slim, pip-installs
`requirements.txt` under `constraints.txt` so the image gets the
versions the suite was last green against, not whatever is newest;
copies the repo; keeps a copy of the committed reference signals
tables outside `/app/data`). `deploy/entrypoint.sh` (seeds those
tables onto the volume when missing — a persistent volume mounted at
`/app/data` hides everything the image had there, which would
silently remove Phase 3.6's prior-season fallback table on first boot;
symlinks chromadb's `~/.cache/chroma` onto the volume so the ~80MB
embedding-model download happens once, not per deploy; `exec`s uvicorn
on `$PORT` or 8080). `.dockerignore` (no `.git`, `.venv`, `.env*`,
`data/raw`, `data/chroma`, SQLite, status file — and explicitly *not*
the tracked parquet tables). `fly.toml` (one always-on
`shared-cpu-1x`/2GB machine — 256MB is not enough to load a season of
pbp into polars and re-embed; a volume at `/app/data`; `force_https`;
`/api/health` check with a 60s grace period; `auto_stop_machines =
"off"` because the in-process refresh needs the machine up, with the
multi-machine alternative documented inline). The server needs no
`SLEEPER_LEAGUE_ID`/`MY_ROSTER_ID` — with them unset every league,
including Rohan's own, gets a per-league directory under
`data/raw/leagues/`, which is the cleaner layout for a host anyway.

#### Validation actually run
- `tests/test_api_static.py` (16 new tests, suite now 346/346): `/` →
  307 → `/ui/`; `/ui` and `/ui/` both land on the page; the page by
  filename is byte-identical; manifest/sw/icons served with the right
  content types; every relative href in the `<head>` (and the
  `sw.js` registration) resolves to a 200 under the mount; manifest
  `start_url`/`scope` stay relative and the SW keeps its scope guard;
  API routes and `/docs` not shadowed; `/api/health` before any cycle
  and with a failed cycle on disk; the lifespan starts the refresh
  and sets its stop event on shutdown; the off switch is honored; the
  landing view makes no API calls.
- **Against a real `uvicorn src.api.main:app` process** (not
  TestClient): curl of every route above with the expected status,
  content type and redirect target; `/api/health` returning the real
  last cycle from this machine (2026-09-16, `ok`, season 2026,
  as-of-week 2).
- **In real headless Chrome** (Playwright, `channel="chrome"`, 390px
  viewport) against that server: `/` lands on `/ui/` with the landing
  view visible; the service worker registers with scope
  `http://127.0.0.1:8765/ui/`; a `fetch('/api/health')` from the page
  returns 200 straight from the network; the CTA opens the login view;
  zero console errors.
- **One real, unplanned refresh cycle from the API app's own
  lifespan.** A `uvicorn src.api.main:app` process was started by
  accident (a shell-quoting slip while opening the PR) with the real
  `.env` and the refresh switch unset, i.e. on. Before it was stopped,
  the lifespan-started thread ran a complete cycle against real
  Sleeper + nflverse: 76.5s, `outcome: ok`, week-2 table (313 rows),
  both leagues `sleeper=ok embed=ok` (1459 / 1366 chunks). So "the
  refresh starts from `src/api/main.py` now, not the dev server" is
  confirmed live, not just by the stubbed lifespan test.

#### Flagged, not done
- [ ] **The image has never been built.** No Docker here. The
      Dockerfile is straightforward (slim Python, wheels for every
      pinned package exist for linux/x86_64) but "it builds" is a claim
      only `docker build` or `fly deploy` can make. If it fails, the
      likely suspects are a pinned wheel missing for linux (relax that
      line in `constraints.txt`) or `cp -r` of the seed tables.
- [ ] **The deploy itself** — account, card, `flyctl`, secrets, the
      one-time `--backfill`. README has the sequence.
- [ ] **A real friend's league on the hosted URL** — the first login
      from someone else's Sleeper account ingests their league on the
      host and is the actual multi-league proof 5.6 is for.
- [ ] `constraints.txt` was frozen from Rohan's macOS venv. It has no
      platform-only packages in it (checked), but it does pin
      transitive deps the Linux resolver might legitimately want at a
      different version; constraints only bind packages pip is
      installing, so the failure mode is a loud resolver error, not a
      silent mismatch.

### 5.7 — Automated data refresh (`src/scheduler/refresh.py`)
**Closes the longest-standing gap in the project.** `src/scheduler/
refresh.py` has been in `PROJECT_SPEC.md`'s repo structure since Phase 1
and flagged as not-built since Phase 2 (see Phase 3.5's entry above, and
`retrieve.py`'s own "a `src/scheduler/refresh.py` cadence gap" note):
until now every report and every `recommend()` answer was only as current
as whenever someone last ran `matchup_signals.py` + `embed.py` +
`sleeper.py` by hand. Numbered 5.7 rather than 5.4 because 5.4 (PWA
installability) already exists and is unstarted; implemented out of
numeric order because it is a real prerequisite for 5.6 (a deployed
server nobody is babysitting), not something to do after it.

#### What the Step 1 investigation actually found (measured, not assumed)

- [x] **Does a running server notice updated data on disk without a
      restart? Mostly yes -- with one real, serious exception.**
      Reproduced directly against a real `python -m web.dev_server`
      (fixture league on the flat dirs, real 2025 signals computed live
      from nflverse): with the server untouched, a separate process
      advanced `nfl_state.json` from week 3 to 4, added a player to
      `teams.json`, wrote `signals_2025_week4.parquet` and re-embedded --
      and `GET /api/roster` immediately showed the new player (Jaxon
      Smith-Njigba), `GET /api/reports/drop` moved from `as_of_week` 3 to
      4 and switched to the week-4 numbers (Nacua's red-zone share
      4% -> 3%, opponent PHI -> IND). Nothing in `src/rag/lookup.py`,
      `ranking.SignalTables.load()` or `_infer_season_and_week()` caches:
      they re-read JSON/parquet per call.
- [x] **The exception: the semantic path went stale AND crashed.**
      chromadb answers `collection.query()` from a per-process in-memory
      vector index, and PR #24's `warm_chroma()` guarantees every league's
      path is warmed at session time. So after a re-embed, a warmed
      process never surfaces a chunk the refresh ADDED and returns the
      ids of chunks the refresh REMOVED -- with `documents`/`metadatas`
      of `None`. `collection.get()` (the metadata path:
      `query_player_signal`, so `get_player_signals`) reads SQLite
      directly and stays fresh; only `collection.query()` (so
      `retrieve.query`, so the `search_league_info` chat tool) is
      affected. Isolated to a reader-only warm process, not just a
      process that did its own writing.
- [x] **And the user-visible symptom is worse than stale data: an HTTP
      500.** `_tool_search_league_info` does
      `r["metadata"].get("type")`, so a phantom hit whose metadata is
      `None` raises `AttributeError` and the whole `/api/chat` request
      fails. Demonstrated end to end on the real server with only the
      Claude model faked (the project's established boundary): before a
      refresh, `HTTP 200` + `'Week 3 matchup 1: ...'`; after a separate
      process re-embedded to week 8 with no restart, `HTTP 500 Internal
      Server Error`. With the fix in place the same sequence returns
      `HTTP 200` + `'Week 8 matchup 1: ...'`. **This means building the
      scheduler without fixing this would have started 500ing every chat
      that reached for `search_league_info`** -- the fix is not optional
      polish.
- [x] **nflverse's real data latency** (checked against the live
      repositories and release assets, 2026-09-13, not from memory):
      `nflverse-pbp`'s `update_data.yaml` cron is daily at 09:00 UTC plus
      Fri 05:30 (post-TNF), Sun 22:00 (early window), Mon 00:05 (late
      window), Mon 05:30 (SNF) and Tue 05:30 (MNF) -- at most two builds
      on any calendar day. `ngs-data`'s `update_ngs.yaml` is once a day
      at 07:00 UTC. Schedules rebuild every 5 minutes. nflreadr's own
      schedule article adds that raw pbp JSON is available ~15 min after
      a game but `load_pbp()` is the nightly build, and that the NFL's
      stat corrections land Monday-Wednesday, so Thursday's pull is the
      cleanest. Corroborated against the live assets: `play_by_play_2026.parquet`
      Last-Modified Sat 12 Sep 12:50 UTC, `ngs_receiving.parquet` Sat 12
      Sep 11:23 UTC, both containing exactly the two games finished at
      that point.
- [x] **Both pipeline steps are already safe to re-run repeatedly.**
      `sleeper.run()` rewrites each JSON file whole (`_save_json` ->
      `write_text`), one file per week for matchups/transactions so weeks
      accumulate rather than collide; `save_signals_table()` overwrites
      `signals_{season}_week{N}.parquet`; `embed.embed()` deletes every
      existing id then re-adds, by documented design. Pinned now by
      `test_running_a_cycle_twice_changes_nothing` and
      `test_repeated_embeds_of_the_same_data_do_not_accumulate_chunks`.
      One real finding: the signals parquet is NOT byte-identical across
      identical runs -- polars' `group_by`/`join` don't promise a stable
      row order -- so idempotency is asserted on content, which is all
      any consumer reads (everything keys by `player_id`).
- [x] **How many leagues can be registered: no limit, and the SQLite
      `leagues` table is the wrong list to refresh from.** It holds every
      league Sleeper listed for everyone who has logged in, including
      ones nobody ever opened. A league gets local data the first time a
      session selects it (`ensure_league_data`), so the refresh
      enumerates disk instead: new `leagues.ingested_league_ids()` returns
      the flat `SLEEPER_LEAGUE_ID` league plus every
      `data/raw/leagues/<id>/` that is actually ingested. Side benefit
      that matters for 5.6: the scheduler needs no database access at all.

#### What was built
- [x] `src/scheduler/refresh.py`: one cycle = (a) the shared,
      league-agnostic signals table computed ONCE (asserted by
      `test_the_signals_table_is_computed_once_per_cycle_not_once_per_league`
      -- three leagues, one nflverse pull), (b) every ingested league's
      Sleeper pull, (c) every league's Chroma collection re-embedded,
      then the running server's Chroma cache invalidated and a status
      record written. Reuses `sleeper.run()` / `build_signals_table()` /
      `save_signals_table()` / `embed.embed()` rather than
      reimplementing any of them.
- [x] **The as-of-date rule decides the target week, and nflverse -- not
      Sleeper -- is the authority.** `target_as_of_week()` = last
      fully-completed week + 1, completeness read off nflverse's
      `result` column (null until a game is final). Sleeper's
      `display_week` advances on its own clock and would let a cycle
      compute week N while week N-1 was still being played. Two
      deliberate details: it walks up from week 1 rather than taking
      `max(completed) + 1`, so a week still in play can never be jumped
      over (week 3 done, week 4 live, week 5 somehow scored -> target 4,
      not 6); and a `COMPLETION_GRACE_DAYS = 7` clause stops one
      postponed game from pinning the target week for the rest of the
      season. Verified against real live schedules: 2024 -> 19, 2025 ->
      19, 2026 (week 1 in progress today) -> 1.
      The only week-N data a cycle reads is the upcoming opponent and the
      Vegas implied total, both published before kickoff -- documented in
      the module docstring as accepted, not hidden.
- [x] **Cadence: every 6 hours**, derived from the latency findings above
      rather than guessed -- pbp builds at most twice a day and NGS once,
      so polling faster is wasted work, and six hours picks up every pbp
      build within a few hours. Configurable via
      `ASKMADDEN_REFRESH_INTERVAL_SECONDS` (floored at 60 so a typo can't
      busy-loop nflverse); `ASKMADDEN_REFRESH_ENABLED=0` turns it off.
      Both documented in `.env.example`. The target week's table is
      recomputed every cycle rather than skipped when the file exists --
      that is how the NFL's Monday-Wednesday stat corrections land.
- [x] Wired into `web/dev_server.py`'s lifespan as a daemon thread that
      runs a cycle at startup and then on the interval, stopped cleanly
      on shutdown via a `threading.Event`. **Documented clearly, in both
      `refresh.py` and `dev_server.py`, that this is NOT the answer for
      5.6**: on a host that runs more than one web replica every replica
      would run its own cycle, and a host that sleeps idle processes
      would run none. `python -m src.scheduler.refresh --once` is the
      entry point for whatever 5.6 picks (cron, a worker process, a
      scheduled cloud function) -- idempotent, no database, non-zero exit
      on failure. Deliberately no distributed lock, queue or retry
      backoff: that is building for a hosting setup that doesn't exist yet.
- [x] **The API freshness fix** (`src/api/leagues.py`, the one `src/api/`
      change the investigation justified): `warm_chroma()` now records the
      on-disk index's `(mtime, size)` stamp and re-checks it on every
      request (it is already called per request by `ensure_league_data`),
      forgetting chromadb's cached System for that path when it changed
      so the next client reads the index off disk. Works for a refresh in
      this process OR in a cron job. Three deliberate choices:
      per-path rather than chromadb's public `clear_system_cache()`
      (which drops every path, letting another league's request thread
      re-create its System outside the warm-up lock -- exactly PR #24's
      race); nothing is stopped, only forgotten, so a query already in
      flight on the old System keeps working (verified under four
      concurrent query threads, zero errors); and the stamp is read
      AFTER the open, because opening a `PersistentClient` itself bumps
      the SQLite mtime (measured) -- a pre-open stamp would make every
      request rebuild the client.
- [x] **Status visibility**: `data/processed/refresh_status.json`
      (gitignored), written atomically via temp file + `os.replace`, with
      last-run start/finish/duration, outcome (`ok` / `partial` / `error`),
      season and as-of-week, per-table row counts, per-league
      sleeper/embed status and error text, `next_run_after`, and a
      `consecutive_failures` counter -- one failed cycle is a blip, six in
      a row is an outage. `python -m src.scheduler.refresh --status`
      prints it and exits non-zero when the last cycle wasn't `ok`.
      Plus `logging` lines on every step.
- [x] `--once` / `--status` / `--season` / `--as-of-week` / `--backfill` /
      `--league-id` / `--interval-seconds` / `--status-path` CLI.
      `--backfill` exists for first use: `ranking.load_signals_table()`
      unions every week <= as_of_week when deciding whether a player has
      ANY current-season data, so a fresh deployment wants the earlier
      weeks too.

#### Two real defects found in this unit's own code while validating it
- [x] `run_cycle`/`main()` took `signals_dir=SIGNALS_DIR` as a *signature
      default*, which binds once at import -- so reassigning the module
      constant (tests do) silently didn't apply and the CLI path wrote
      into the real `data/processed/signals/`. Caught by spotting a
      fixture table (`signals_2024_week3.parquet`) in `git status` after
      a test run. Every public function now resolves `signals_dir` /
      `status_path` from the module constants at call time, with a real
      `_UNSET` sentinel so an explicit `status_path=None` ("don't write a
      status file") stays distinguishable from "argument omitted".
- [x] A failure enumerating the leagues set `record["error"]` but left
      `outcome` as `"ok"`, because the outcome was derived only from
      per-league failures. It now counts as `"partial"`, so the status
      file can't report a clean cycle alongside an error message.
- [x] Also added a `.gitignore` rule for `data/processed/signals/*.parquet`:
      the scheduler now regenerates these on a cadence, so a running
      server would otherwise leave a growing pile of untracked parquet in
      every `git status`. The two reference tables the tests and Phase
      3.6's fallback rely on were committed before the rule and stay
      tracked (an ignore rule doesn't untrack a tracked file); a new
      reference table needs `git add -f`.

#### Validation actually run
- [x] **A real cycle, end to end, outcome `ok`**: real nflverse pbp/NGS/
      schedules, real signals computation (452 rows for 2025 week 7), real
      `sleeper.run()` writing 8 files, real re-embed of 3078 chunks, 64
      seconds -- with ONLY Sleeper's HTTP boundary stubbed from the
      fixture on disk, because Sleeper is blocked in this sandbox.
- [x] **Real signal files updated for real**: `signals_2025_week{3,4,5,6,7}`
      and `signals_2026_week1` all computed live from nflverse during this
      session (the week 3/4 pair is what the server-freshness
      reproduction used).
- [x] **The background loop actually ticks inside the real dev server**:
      `python -m web.dev_server` with a 75-second interval logged
      `background refresh started: every 75s`, ran a cycle at startup and
      again on the interval, each completing the full signals -> Sleeper
      -> embed sequence and rewriting the status file.
- [x] **The API-freshness fix proved the same way Step 1 reproduced the
      bug**: the real server, no restart, `HTTP 500` before the fix and
      `HTTP 200` with the refreshed week-8 chunk after it (see above).
- [x] **Full suite green**: 306 passed, including 40 new tests in
      `tests/test_refresh.py` (week resolution incl. the never-jump-a-live-week
      and grace cases, env config, the whole cycle, shared-signals-once,
      backfill, partial/total failure handling, idempotency, the status
      file and exit codes, `ingested_league_ids()`, and the Chroma
      freshness fix including a subprocess test of the real
      `warm_chroma` + `retrieve.query` path). `tests/test_api_concurrency.py`
      updated for `_chroma_warmed`'s new dict type.

#### Flagged, not done
- [ ] **Needs a real game day on Rohan's machine** -- the one thing no
      sandbox session can prove: let the server run across an actual NFL
      Sunday/Monday and confirm a cycle picks up a finished game, i.e.
      that `target_as_of_week()` advances exactly once the last game of
      the week goes final in nflverse and that the newly-computed table
      really does contain that game's plays. The logic is verified against
      real completed seasons and real in-progress 2026 data, but "it
      advanced at the right moment, live" is a multi-day observation.
      Check it with `python -m src.scheduler.refresh --status`; watch for
      `consecutive_failures` climbing. **Partial, 2026-09-16:** one real
      advance is on record -- the overnight cycle after week 1's Monday
      game (`refresh_status.json`: `completed_weeks: 1, as_of_week: 2`,
      313-row week-2 table, both leagues clean in 107s) -- so "it
      advanced after the week finished" has happened once for real. Not
      yet watched *at* the moment of the flip, and not yet across a
      full week's Thursday/Sunday/Monday sequence; keep the box open.
- [x] **The live Sleeper refresh is no longer stubbed**: two real
      `python -m src.scheduler.refresh --once` cycles on Rohan's machine
      on 2026-09-16 (both leagues, real Sleeper + nflverse, 99-107s each,
      `outcome: ok`), plus the overnight background cycle from the dev
      server. Closed.
- [ ] **First-run command to remember on a new machine**: `python -m
      src.scheduler.refresh --once --backfill` before relying on the loop,
      so the season's earlier weeks exist, not just the current one.
- [ ] **A narrow, documented window in the freshness stamp**: if a
      refresh's final commit lands inside the server's own
      `PersistentClient` construction, the recorded stamp already covers
      a write the new System may have opened just before, and nothing
      re-warms until the next refresh. Milliseconds wide;
      `invalidate_chroma()` closes it outright for an in-process refresh.
      Closing it for an out-of-process one too would mean a generation
      marker written by `src/rag/embed.py` itself -- a `src/rag/` change,
      out of this unit's scope.
- [ ] **`src/ingest/realtime.py`'s "tighter cadence" tier is not wired
      in**, and not because the data is missing: nflverse's injury feed is
      live again (182 rows for 2026 as of 2026-09-13, despite nflreadr's
      docs still saying the source died after 2024). It's that nothing
      downstream consumes `realtime.py` -- no signal in
      `matchup_signals.py` reads it, no chunk in `embed.py` carries it --
      so refreshing it would write data no report or answer can reach.
      Wiring injuries into the signals table is signal-computation work,
      not scheduling work, and belongs with Phase 2's remaining gaps.
- [ ] **A 0-row signals table is written before a season's first game**
      (real today: `signals_2026_week1.parquet`, 0 rows). Verified
      harmless -- `load_signals_table` returns `{}`, Phase 3.6's
      prior-season fallback takes over with the real 2025 numbers, and
      `load_signal_chunks` builds nothing -- and honest (it records that
      we looked and there was nothing), so it is left as is rather than
      special-cased.

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
Resolved -- product decision made (guaranteed agreement on the
*verdict*; Chat adds explanation on top) and implemented. See the
"Fixed: Chat and the Feed's start_sit report could reach different
verdicts on the same signals" section above for what was actually found
and changed, including what is still only prompt-enforced and needs
real-model re-validation.
- [x] Product decision: guaranteed agreement vs. accepted divergence
      -- guaranteed agreement on the verdict, Chat free to add context
- [x] Code change: `src/reasoning/ranking.py` (shared), `rank_players`
      tool + prompt rule + code-level verdict guard in `recommend.py`,
      `report.py` refactored onto the shared module with unchanged output
