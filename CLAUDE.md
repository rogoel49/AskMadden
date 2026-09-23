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
  grouped by position rather than a league's full Sleeper roster-slot
  structure -- **no longer true as of 2026-09-20**, start_sit now fills
  the league's actual `roster_positions`, dedicated slots first then
  FLEX-type slots, see TODO.md's "Fixed: start/sit recommended one
  starter per position..." entry; waiver_pickups' "rising" target share is a point-in-time
  value, not an actual week-over-week delta the project doesn't compute
  yet). The `src/scheduler/refresh.py` gap this phase flagged — every
  report and `recommend()` call only as current as the last manual
  ingest/signals/embed run — is **closed by Phase 5.7**.
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
  deliberately not touched there; built in Phase 5.7, which makes this
  kind of stale index self-correcting) rather than a defect in the fallback
  logic itself; either way, `has_signals: false` is exactly the signal
  this fix now requires the model to act on explicitly. See TODO.md's
  Phase 3.7 section for full detail, including the live-model validation
  gap this sandbox still can't close (no `ANTHROPIC_API_KEY`).
  **Addendum, found during this phase's own real-model validation of Gap
  1's fix:** Gap 2 validated clean for real (`Jeremiyah Love`, `Kenyon
  Sadiq` both correctly flagged `has_signals: false`). Gap 1's original
  fix did not hold up, and failed worse than the original bug: the real
  model answered the real compound trade question with `data_gaps: []`
  but `reasoning` containing specific, fabricated trade strategy ("package
  one QB... to upgrade at TE") backed by zero cross-team tool calls — it
  found a way to *sound* responsive to the out-of-scope half without
  actually engaging it, which slipped past the original "record a gap"
  instruction entirely. Strengthened the system prompt with an explicit,
  stricter rule (every specific claim in `recommendation`/`reasoning`
  must be backed by an actual tool call made that turn, or it must become
  a `data_gaps` entry instead) and an explicit capability boundary
  (`find_owner` only answers "who owns this named player," is NOT a
  roster-comparison/trade-fit tool, and no tool here inspects another
  team's roster or compares needs/value across teams). Honestly, this
  addendum's test coverage is a documented non-safeguard, not a fix
  verification: a fake-client test can prove the orchestration loop
  passes through whatever the model says (including a deliberately
  fabricated "bad" response) and that the new prompt text exists, but
  neither proves a real model follows it — only real-model re-validation
  of the exact same question can confirm the fix, and that's still
  outstanding (flagged, not done in this sandbox).
- Phase 3.8 (roster-composition visibility across the league):
  implemented. Closes the specific gap Phase 3.7's honest decline left
  behind — "no tool inspects another team's roster" was true then, but
  closeable: Sleeper's roster data is already ingested league-wide
  (`lookup.py`'s `all_rostered_players()` already proved that), nothing
  exposed it *per team* to the reasoning agent. New tool
  `get_league_rosters`, backed by new `lookup.py` functions
  `all_team_rosters()`/`team_roster_for_owner()` — every team's roster
  grouped by position with a per-position count; omit
  `owner_display_name` to survey the whole league at once, give one to
  look at a single team. Composition only, deliberately — updated the
  system prompt carefully (not just additively) so a compound trade
  question now gets a real, grounded partial answer (which teams have
  surplus/need at the weak position) plus an honest `data_gaps` entry for
  the still-missing valuation piece, without reopening Phase 3.7's
  anti-fabrication fix: `get_league_rosters` tells the agent WHAT a team
  has, never whether a trade is fair or what to offer, and the prompt
  says so explicitly. Also corrected `find_owner`'s own tool description,
  which previously (accurately, at the time) claimed no tool inspects
  another team's roster at all — now false, and left uncorrected would
  have had the model believe something false about its own capabilities.
  Trade valuation itself is deferred to Phase 6, not attempted here. See
  TODO.md's Phase 3.8 section for full detail, including the live-model
  validation gap this sandbox still can't close (no `ANTHROPIC_API_KEY`).
- Phase 4 (coverage classification stretch): optional, not started
- Phase 5 (productization — final deliverable): **complete 2026-09-21**
  — live at https://askmadden.com (Fly.io, custom domain), five leagues
  and four real users, every checklist item ticked; the open work is
  ranking quality (see TODO.md's "Where the accuracy actually stands").
  History of how 5.5/5.6 landed, kept for context: 5.6: `src/api/
  main.py` now serves the frontend itself (`design/` mounted at `/ui`,
  `/` redirects there, the `/ui` prefix kept on purpose so `sw.js`'s
  scope never covers `/api/`), starts the 5.7 refresh from its own
  lifespan, and has `GET /api/health` for the host's health check;
  `web/dev_server.py` is a thin 0.0.0.0 launcher over the same app.
  Deployment config exists (`Dockerfile`, `deploy/entrypoint.sh`,
  `.dockerignore`, `fly.toml`, `constraints.txt`) but **the image has
  never been built or deployed** — no Docker/flyctl on the machine it
  was written on; the README's Deploying section is the sequence.
  Verified against a real uvicorn process and real headless Chrome
  (SW scope `/ui/`, `/api/` fetched straight from the network, no
  console errors). 5.5: landing copy reviewed and edited so nothing
  overclaims (signals named are real ones), `<title>` is "Ask Madden",
  "no API calls from the landing view" pinned by a test; the eval band
  is still labeled placeholders — running the decision eval at volume
  is real Claude spend on Rohan's key, his call. See TODO.md's 5.5 and
  5.6 entries. 5.4 (PWA installability):
  `design/manifest.json` + `design/sw.js` + `design/icons/`
  (programmatic Anton "AM" icon, `build_icons.py`) and the iOS
  `apple-touch-icon` / `apple-mobile-web-app-*` head tags in the
  mockup; iOS Safari was the priority target (verified against
  Apple's Safari release notes: iOS 26 opens anything added to the
  Home Screen as a web app, 16.4-18 need the manifest display member
  or the legacy meta -- both set; icon always comes from
  apple-touch-icon). Lighthouse 11 PWA audit scores 100 in headless
  Chromium; `web/dev_server.py` now binds 0.0.0.0 so a phone on the
  same WiFi can reach it. The real-phone "Add to Home Screen" is the
  one thing only Rohan can confirm -- and it was NOT confirmed before
  PR #25/#27 merged; the first real-iPhone screenshot found the phone-
  frame illustration rendering inside the phone, fixed in the "5.4
  follow-up" entry in TODO.md (device-emulated audit: iPhone 14 Pro
  and Pixel 7 profiles 63/63 each, desktop parity identical, real-phone
  checklist still open). Android's install prompt needs a
  secure context, so over plain http://LAN-IP it's iOS-only until
  5.6's HTTPS. See TODO.md's Phase 5.4 entry. 5.3 (wire the mockup): the
  mockup's `<script>` and hardcoded data markup now call the seven
  real routes (login, sessions, roster, three reports, multi-turn
  chat); three distinct chat chips (stale / no_signal_data /
  out_of_scope_capability); Moves→Trades asks chat a fixed
  composition question on request rather than inventing an endpoint.
  Served same-origin — by `web/dev_server.py` at the time because
  `src/api/` had no static mount; as of 5.6 by `src/api/main.py`
  itself. Validated in a real headless browser at both
  breakpoints against the real server with boundary mocks (58
  checks); live login / real-model chat still need Rohan's machine.
  See TODO.md's Phase 5.3 entry. 5.2 (API + storage): `src/api/`
  — `auth.py` (Sleeper username → user_id → leagues, documented
  shapes, mocked in tests, live run outstanding), `storage.py`
  (SQLite: users, leagues, sessions, daily query counts),
  `leagues.py` (per-league data dirs + ingest-on-first-use reusing
  the existing ingest/embed), `main.py` (FastAPI; chat exposes
  `recommend()`'s `messages` for multi-turn; `data_gaps` verbatim +
  per-player `signals_consulted` stale markers; reports verbatim;
  per-user/day cap). `roster_id` is now an explicit parameter down
  to `src/rag/lookup.py` (`MY_ROSTER_ID` remains the CLI fallback —
  the one `src/rag/` change, by instruction), and `.env` loads once
  per process via `recommend.load_dotenv_once()`. See TODO.md's
  Phase 5.2 entry for what's mocked vs. real. 5.1 (league/scoring
  parameterization):
  `recommend()`/`generate_report()` take a required `league_id`,
  resolved and verified against the ingested `league.json` by new
  `src/reasoning/league.py` (`LeagueMismatchError` on a different
  league, never a silent wrong-league answer); CLIs take `--league-id`
  defaulting to `SLEEPER_LEAGUE_ID`. The investigation found the
  plan's "currently assumes half-PPR" premise was a single-league
  blind spot, not a magic number — `recommend.py` already read the
  real `scoring_settings`, `report.py`'s ranking is scoring-independent
  (now pinned by a test), `matchup_signals.py` has no scoring concept.
  `MY_ROSTER_ID` is still env-implicit inside `src/rag/lookup.py`
  (out of 5.1's scope) — 5.2's job. See TODO.md's Phase 5.1 entry.
- Phase 5.7 (automated refresh — `src/scheduler/refresh.py`):
  implemented, closing the project's longest-standing gap (listed in
  PROJECT_SPEC.md's repo structure since Phase 1, flagged not-built
  since Phase 2). Numbered 5.7, not 5.4 — 5.4 (PWA installability) is
  a separate item, done on its own branch. One cycle = the shared, league-agnostic
  signals table computed **once**, then every ingested league's Sleeper
  pull, then each league's re-embed; it reuses `sleeper.run()` /
  `build_signals_table()` / `embed.embed()` rather than reimplementing
  them. **nflverse, not Sleeper, is the authority for the as-of week**:
  the target is `last fully-completed week + 1` with completeness read
  off nflverse's `result` column, because Sleeper's `display_week`
  advances on its own clock and would let a cycle compute week N while
  week N-1 was still being played. Cadence is 6 hours, derived from
  nflverse's measured build schedule (pbp at most twice a day, NGS once
  a day — checked against the live repos, not guessed), configurable via
  `ASKMADDEN_REFRESH_INTERVAL_SECONDS`. Wired into `web/dev_server.py`
  as a daemon thread; Phase 5.6 must use `python -m
  src.scheduler.refresh --once` from a cron job/worker instead (an
  in-process thread is wrong for a host with multiple replicas or one
  that sleeps idle processes) — documented in both modules. The
  investigation also found and fixed a **real API bug, not just
  staleness**: chromadb answers `collection.query()` from a per-process
  in-memory vector index, so after a re-embed a warmed server returned
  phantom hits with `None` documents/metadata and `/api/chat` 500'd on
  `search_league_info` — reproduced and fixed against a real running
  server (`warm_chroma()` re-checks the index's on-disk stamp per
  request). The structured paths (parquet signal tables, Sleeper JSON,
  Chroma's metadata-filtered `get()`) were already fresh — confirmed
  live, not assumed. Status visibility is
  `data/processed/refresh_status.json` + `--status`. **Flagged not
  done:** a real game-day observation on Rohan's machine (the week logic
  is verified against real completed seasons and the real in-progress
  2026 season, but "it advanced live at the right moment" is a multi-day
  check), the live Sleeper call (still blocked in this sandbox), and
  `src/ingest/realtime.py`'s tighter cadence (deliberately skipped —
  nothing downstream consumes it yet). See TODO.md's 5.7 entry.
  **Follow-up fix (2026-09-16, first real in-season use):** the
  reasoning path and the Sleeper ingest read Sleeper's `display_week`,
  which lags `week` until midweek, so the day after real games every
  card fell back to last season while the refresh's week-N+1 table sat
  on disk. Both now read `sleeper.current_week()` (`week` > `leg` >
  `display_week`). The "default week is pinned to Sleeper's state"
  decision is unchanged. See TODO.md's "Fixed: the day after real
  games..." entry. The degenerate one-week EPA trend it flagged (0.0
  for every player until week 5, printed as "trending down") is fixed
  too: `epa_trend` is null with a new `epa_baseline_plays == 0` until
  there is a baseline outside the trailing window, and prose/chunks say
  "no efficiency trend yet" -- see TODO.md's "Fixed: the early-season
  EPA trend..." entry. **Follow-up fix (2026-09-20, first desktop use
  by a friend):** a Chroma rebuild was delete-all-then-add, so a chat
  during the cycle's re-embed saw an emptied index (`has_signals:
  false` for players with data on disk), and a request arriving
  mid-cycle ran its own concurrent rebuild (a minute-long league
  select). `embed.embed()` is now upsert-then-prune under a per-
  directory lock, and the request-path resync stands down while a
  rebuild or cycle is running -- see TODO.md's "Fixed: chat lost every
  signal during a re-embed..." entry. **Same day, from the same friend's
  league:** start_sit recommended one RB in a league that starts two;
  now slot-aware (`recommended_starters` per entry, FLEX filled from the
  leftovers, `recommended_starter` kept as the top pick Chat is held
  to). QBs were unrankable (no target/red-zone share) so the QB slot
  was always skipped; `ranking.opportunity_score()` has a passer branch
  (CPOE + implied total, labeled in `score_description`). Players Out /
  on IR are left out of the lineup and named in `notes`;
  `get_my_roster` returns `injury_status`. `/api/roster` returns the
  lineup by slot (`lineup`/`bench`/`reserve`) and the Roster tab draws
  it Sleeper-style. See TODO.md's "Fixed: start/sit recommended one
  starter per position..." entry. **Later the same day:** the EPA trend
  term now needs `MIN_TREND_PLAYS` (20) plays to count (a 5-play player
  had topped the waiver list on trend noise), waiver targets set aside
  last-season-only players once the current season has data, and the
  K/DEF and unrankable-player notes say what they mean -- TODO.md's
  "Fixed: a 5-play player topped the waiver list..." entry. **Eval
  numbers (2026-09-20):** retrieval 80/84 across four leagues (after
  fixing a real bug it found: the chat's league-info search now excludes
  signal chunks via `retrieve.LEAGUE_INFO_ONLY`); the 600-dilemma
  decision run died on an exhausted API credit balance with ~460
  answers lost because `run_decision_eval.run()` buffered them -- now
  resumable via `progress_path`, rerun the same day: **decision
  accuracy 206/399 = 52%** (62% where the actual gap was 10+ pts) --
  honest and unflattering, the baseline later signal work must beat.
  Never run it without the progress file. **Deployed the same day:**
  https://askmadden.fly.dev (Fly.io, `fly launch`/volume/secret/deploy
  from this session; first image build succeeded); askmadden.com bought
  on Cloudflare, certs added, DNS records pending on Rohan's side. See
  TODO.md's "Eval numbers, first real runs" entry.
- Phase 6 (crude, explicitly-labeled trade-value proxy): implemented
  2026-09-21, together with the two accuracy strategies from the first
  eval. `src/signals/player_stats.py` (league-agnostic weekly stat
  lines, refreshed each cycle) + `src/reasoning/points_proxy.py` (scored
  under each league's own settings at query time, strictly as-of) join
  `ppg` / `season_points_so_far_proxy` / `ppg_prior_season` onto every
  ranking row and onto `get_player_signals` / `get_league_rosters`. The
  ranking weights are now **fitted** (`evals/fit_ranking_weights.py`,
  2023 → 2024 held-out: 61.8% vs 57.6% hand-set; ppg alone 61.5%), with
  this season's ppg shrunk toward last season's (`blended_ppg`). The
  system prompt's TRADES section allows rough, labeled proposals from
  tool output only; picks remain a data_gaps entry. Waiver targets rank
  within position. See TODO.md's "Phase 6 built..." entry. Same day:
  `get_league_rosters` also carries each team's needs/surplus vs. its
  real starting slots and each player's injury status and bye week, and
  the prompt pitches trades with those reasons and invites refinement;
  a `pre_draft` league (Sleeper `status`, now on `LeagueConfig` and
  `/api/sessions`) gets an honest "hasn't drafted yet" instead of the
  whole NFL as waiver targets; the fit also holds on held-out 2025
  (59.9%). Live decision eval rerun: 209/400 = 52.2% (flat), and the
  agent picked the ranking's player on 400/400 -- **the live eval is
  the ranking's accuracy on one hard week (2024 wk 5; every scorer is
  52-54% there); the offline harness is the measurement, don't pay to
  rerun the live one until ground truth spans many weeks.** See
  TODO.md's "Fixed: a pre-draft league..." entry. **Later 09-21:**
  calibrated confidence on every verdict (accuracy plan Step 1:
  `ranking.pairwise_confidence`, tempered 0.85, capped 85%; on ranked
  entries, start/sit refs, `rank_players`, Feed cards, and required in
  the prompt); `resolve_named_player()` settles an ambiguous name from
  the user's own question (the "Malachi Fields" -> Corley RCA);
  `league_type` (redraft/keeper/dynasty from Sleeper `settings.type`)
  on `LeagueConfig`, with rookies and second-year players held out of
  dynasty/keeper drop lists; structured `key_stats` on every entry and
  a one-component card Feed on a consistent grid; FORMAT and league-
  type guidance in the prompt. **09-22:** accuracy plan Step 2 done --
  `evals/ground_truth.jsonl` spans 2024 weeks 2-18 and the season-wide
  decision number is **59%** (28,330 dilemmas; 69% at 10+ pt gaps), so
  the 52% was one hard week; a restarted server no longer reruns a
  just-finished refresh cycle (every deploy was pinning the CPU;
  `/api/health` shows `cycle_in_progress`); `embed.embed()` batches
  writes under Chroma's 5,461-record cap (a league's index would have
  crossed it mid-season). See TODO.md's "Fixed: 'Tutu or Malachi
  Fields'..." entry and the Step 2 item in the accuracy plan.
- Phase 7 (coaching-scheme fit signal): not started, backlog. Sequenced
  after Phase 6 — a real signal (Tier 1: a "new offensive coordinator
  this season" fact; Tier 2: an eval-gated, explicitly-labeled scheme-fit
  proxy) for the reasoning agent to reach for when current-season
  signals are stale, instead of unlabeled general knowledge. See
  PROJECT_SPEC.md's Phase 7 section for detail and TODO.md's Backlog
  section for the checklists.

Always check TODO.md for the up-to-date task list within the active phase.

## Frontend architecture (Phase 5)
One responsive frontend, `design/askmadden-ui-mockup.html`, with a
landing view (`#view-landing`) as its entry point, then the
login/league-picker flow, then a single app shell that fills the real
viewport below 900px (100dvh + safe-area insets, `.app-content` as
the one scroller -- the old phone-frame illustration was retired after
a real installed-iPhone screenshot showed it rendering as a phone
inside the phone; see TODO.md's "5.4 follow-up" entry) and reflows to
a sidebar-nav desktop layout at 900px+, via CSS media queries only.
There is no separate static marketing site and no second build target — the landing page is a view inside the same file.
The landing page's eval-numbers band shows labeled placeholders until
real eval numbers exist (see TODO.md's 5.5 entry for what gates each).
As of Phase 5.3 the file is wired to the real API and is served on the
API's own origin (no CORS needed) — since 5.6 by `src/api/main.py`
itself, mounted at `/ui` with `/` redirecting there; it keeps state
in JS for the life of the tab, plus one remember-me token
({username, league_id}) in localStorage so a return visit skips the
sign-in flow (2026-09-22). As of Phase 5.4
it is installable: `design/manifest.json`, `design/sw.js` (scope is
the file's directory, so `/api/` is never intercepted — which is why
the page lives under `/ui/` and not at `/`; the HTML is network-first
so edits show on reload), and `design/icons/`. Keep the manifest/icon
hrefs relative; `tests/test_api_static.py` checks each one resolves
under the mount.

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
