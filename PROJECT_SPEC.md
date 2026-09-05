# Ask Madden — Project Spec

An AI fantasy football assistant that goes beyond rankings — it explains *why*.
Built and validated first against Victorious Secret 3.0 (Sleeper league ID
1389341490030862336, 12 teams, half-PPR), then generalized into a small
real product: paste in any Sleeper league, get the same signals-backed
recommendations for your own roster.

## Pitch (for README.md)
Most start/sit tools give you a projection number. Ask Madden combines
retrieval-augmented generation over live league/player data with a
computed matchup-signals layer (defensive tendencies, coverage-adjusted
efficiency, game script) so recommendations come with real reasoning:
not just "start Player A," but "start Player A — this defense allows
the 4th-most rush yards to RBs and Player A's efficiency trend is up
over his last 3 games." It started as a tool for one league and shipped
as a small hosted product anyone with a Sleeper league can use.

## Architecture
```
Sleeper / nflverse / NGS / odds / realtime
        → signals layer (derived matchup features, league-agnostic)
        → RAG corpus (chunked, embedded, retrievable, league-agnostic)
        → per-league join (roster, scoring settings, matchup schedule)
        → Claude (tool-use agent)
        → recommendation + explanation
        → web UI (hosted, multi-league)
```
Agent = Claude API with tool use (function calling). The model calls
retrieval/signals functions as tools rather than relying on a heavy
framework — a lighter, more legible pattern.

**Key design property that makes productization cheap**: the signals
table and RAG corpus are computed from NFL-wide sources (nflverse, NGS,
odds) and are not specific to any one league. What's actually
league-specific is thin — roster ownership, matchup schedule, and
scoring settings. This was true from Phase 1 onward; Phase 5 below is
mostly about exposing that existing separation through a real
multi-tenant interface rather than rearchitecting anything.

## Repo structure
```
ask-madden/
├── README.md
├── TODO.md
├── requirements.txt
├── .env.example
├── CLAUDE.md                    # persistent context for Claude Code sessions
├── data/
│   ├── raw/                    # gitignored, too large to commit
│   └── processed/              # small computed samples, can commit
├── src/
│   ├── ingest/
│   │   ├── sleeper.py           # league/roster/matchup/scoring-settings pulls
│   │   ├── nflverse.py          # play-by-play, EPA/WPA
│   │   ├── ngs.py               # CROE, aDOT, RYOE
│   │   ├── odds.py              # Vegas lines / game script
│   │   └── realtime.py          # injury/inactive status, weather, line movement (fast-refresh tier)
│   ├── signals/
│   │   └── matchup_signals.py   # derived features, league-agnostic, see table below
│   ├── rag/
│   │   ├── embed.py             # chunk + embed into ChromaDB (league-agnostic corpus)
│   │   └── retrieve.py
│   ├── reasoning/
│   │   └── recommend.py         # retrieval + signals + per-league roster/scoring → recommendation + explanation
│   ├── scheduler/
│   │   └── refresh.py           # runs ingest → signals → embed on a schedule (shared across all leagues); realtime.py on a tighter cadence
│   ├── api/
│   │   ├── main.py              # FastAPI (or similar) app: register league, ask question, get recommendation
│   │   ├── auth.py              # minimal — league ID + display name is enough for v1, no password/OAuth needed
│   │   └── storage.py           # user_id → league_id/team_id mapping (SQLite is fine at this scale)
│   └── cli.py                    # kept for local dev/debugging, not the shipped interface
├── web/
│   └── ...                      # minimal frontend: paste Sleeper league ID → see your team → ask questions
├── evals/
│   ├── eval_questions.jsonl     # question set (qualitative + nflverse-generated)
│   ├── ground_truth.jsonl       # actual fantasy points, generated from nflverse — never hand-authored
│   ├── run_eval.py              # backtest harness, as-of-date filtered (no future leakage)
│   └── results/
│       └── YYYY-MM-DD_run.json  # dated snapshots to show accuracy trend over time
└── tests/
```

## Signals table

| Signal | Captures | Source | Note |
|---|---|---|---|
| Defense run-funnel rate | Run vs. pass yards allowed vs. average | nflverse | |
| Red zone role share | TD equity proxy | nflverse | |
| Recent efficiency trend | EPA/play or YPRR, last 3 games vs. season | nflverse | |
| Opponent-adjusted target share | Target share weighted by opposing pass defense rank | nflverse + derived | |
| Game script / implied total | Vegas spread + total | odds API (free tier) | |
| CROE (catch rate over expected) | Route-winning / contested-catch ability | NGS (free public site) | |
| aDOT + target depth distribution | Deep-threat vs. possession profile | NGS (free) | |
| RYOE (rush yards over expected) | RB efficiency independent of blocking | NGS (free) | |
| Matchup-fit score (derived) | Player profile × opponent coverage-shell tendency | NGS + Sharp Football/FantasyPoints coverage-shell writeups | Modeled proxy, not a direct measurement — label clearly in README |
| Injury/inactive status | Availability | Sleeper (existing field) | Fast-refresh tier |
| Weather | Passing/kicking conditions | free weather API | Fast-refresh tier |

All signals above are league-agnostic — computed once, shared across every registered league.

## Phase 3.5: Multi-turn conversation + report generation
Two extensions to Phase 3's `recommend.py`, both backend work that
doesn't need Phase 5's web UI to exist first — validated now via
`recommend.py --interactive`'s CLI REPL; Phase 5 later wires the same
conversation object to a UI instead of a terminal.

**Multi-turn conversation.** `recommend()` accepts an optional prior
`messages` history and returns the updated history, so a clarifying
question the agent asks (e.g. "which of your flex-eligible players do
you mean?") can be answered in the same conversation instead of forcing
a fresh, context-free question. Single-question callers (the CLI's
non-interactive mode, the eval harness) simply never pass `messages` —
each of those stays a genuinely independent conversation, which matters
for `evals/run_eval.py` and `evals/run_decision_eval.py`: carrying
state across supposedly-independent eval questions would leak context
between them and invalidate the eval.

**Report generation.** A standalone `generate_report(report_type,
raw_dir, ...)` function, reusing `recommend()`'s existing tools
(`get_my_roster`, `get_player_signals`, `get_team_record`, etc.) rather
than bolting report logic onto the single-question chat path — these
are genuinely different orchestration patterns (iterate over a whole
roster or league, not resolve one question) sharing the same
underlying tools and per-league context, not different prompts wedged
into the same code path. All four report types live in the same web
chat UX as the Q&A flow per Phase 5's original pitch, not a separate
bot wrapper (that's explicitly out of scope — see Phase 4's Discord
bot bullet, which stays a *stretch*, not a requirement).

Four report types, sequenced by **data readiness**, not implementation
difficulty:
1. **Start/sit** — generalizes Phase 4's "weekly auto-generated lineup
   recommendations" bullet into every roster spot, not just one
   question. Fully buildable now with the existing matchup signals.
2. **Drop** — identify the weakest roster contributors using existing
   signals (recent efficiency trend, role share). Buildable now.
3. **Waiver-wire pickups** — rank unrostered players (the full NFL
   player pool minus every league roster, both already ingested) by
   the same existing opportunity signals (rising target share, red
   zone role, etc.). Buildable now, no new signal work needed.
4. **Trade suggestions** — deliberately sequenced last, and **not
   attempted yet**. Needs signal work this project hasn't built:
   season-long/rest-of-season player value (every existing signal is
   this-week's-matchup-scoped, not a value-over-a-longer-horizon
   measure) and positional need assessment relative to both the user's
   own roster construction and other teams' rosters (a genuinely new
   kind of computation — nothing today compares roster composition
   across teams). Scope and flag the specific new signals this needs
   as a follow-up; do not build the report itself until that signal
   work exists — a trade suggestion grounded in the wrong kind of
   signal (single-week matchup context standing in for value) would be
   actively misleading, worse than not having the feature at all.

## Phase 3.6: Prior-season signal fallback
Discovered during Phase 3.5's real-data validation, not planned ahead of
time: every signal in the signals table (`matchup_signals.py`) is
trailing/current-season by construction (EPA trend, red zone share,
target share, ...). Confirmed live against the real, unstarted 2026
season: nflreadpy has no 2026 plays published yet (games haven't been
played), so `generate_report()` and `recommend()` both correctly return
an empty/near-empty result with an honest "no computed signals" note --
correct behavior for a genuinely empty table, but a bad first-use
experience if a friend opens this to set a Week 1 lineup and gets
nothing useful back, right when a first impression matters most.

**Fix**: when a player has no current-season signal at all, fall back to
their most recent PRIOR season's final numbers as a reference point --
but never silently. Every fallback number is explicitly labeled stale
end-to-end: a `stale: true` / `source_season` / `source_as_of_week`
marker on the raw row, on every report entry, AND a `[STALE -- ...]`
text prefix on any prose (`signals_summary`, `reasoning`, and the chat
path's `get_player_signals` tool output) -- a report or chat answer must
never present last season's numbers as if they were this season's.

**Where it lives**: the signal-*loading* layer only, not the signals
*computation* module (`matchup_signals.py` is untouched) --
`src/reasoning/report.py`'s `_load_signals_table()` /
`_load_prior_season_fallback_table()` / `_signal_row()` for the report
path (reads the raw parquet table directly, same reason the rest of that
module does -- see its docstring), and
`src/rag/retrieve.py`'s `query_player_signal_with_fallback()` for
`recommend.py`'s `get_player_signals` chat tool (reads Chroma). These are
two separate, parallel implementations against two different data
sources, not one shared function -- consistent with how the rest of the
report/chat split already works.

**Fallback threshold: N=1.** A player with ANY current-season signal at
all (even from an earlier week than as_of_week) is used as-is, never
blended with a stale prior-season number -- only a player with ZERO
current-season data falls back. A larger N (waiting for 1-2 weeks of
current-season data before trusting it, to smooth out one noisy early
game) would need a real "distinct weeks active this season" field that
`matchup_signals.py` doesn't compute today -- that's signals-computation
work, explicitly out of scope for this unit. Revisit if
`matchup_signals.py` ever gains that field.

**Explicitly out of scope for this unit**: trade suggestions (still
Phase 3.5's own deferred item, unrelated), `src/scheduler/refresh.py`
(still its own separate, un-started gap, see TODO.md), and the existing
signal-weight constants in `report.py` (`_EPA_TREND_WEIGHT` etc. --
unchanged).

## Phase 3.7: Compound questions + structured data gaps in the chat path
Discovered during Phase 3.6's own real-model validation (`recommend.py`
run for real, against a real roster, with `ANTHROPIC_API_KEY`), not
planned ahead of time -- two real gaps, both specific to `recommend.py`'s
chat path (`report.py`'s reports already handle both correctly via their
`notes` field, see below):

**Gap 1: a compound question fails entirely instead of partially.**
"What's my weakest position, and who should I trade with to strengthen
it" burned all `max_turns` tool-use turns and returned a bare "I don't
have enough information" -- even though the first half (weakest
position) is fully answerable with `get_my_roster`/`get_player_signals`,
and only the second half (a trade partner suggestion) is genuinely out
of scope (needs the same season-long-value + cross-roster
positional-need signals Phase 3.5's trade-suggestions writeup already
flags as not built). The fix is a system-prompt instruction: recognize
when a question has an answerable part and an out-of-scope part, and
answer the answerable part explicitly instead of letting the
unanswerable part fail the whole question.

**Gap 2: the chat path doesn't consistently flag a player with NO signal
data at all, as distinct from stale prior-season data.** Live validation:
asking "what's my weakest position on my roster right now" for a real
2026-preseason roster correctly identified TE as the weak spot (Harold
Fannin, Kenyon Sadiq) but cited zero actual signal numbers for either --
just generic "rookies/unproven players with uncertain roles," the
model's own background knowledge, not anything `get_player_signals`
returned. `get_player_signals` already distinguishes this case internally
(a chunk truly doesn't exist vs. Phase 3.6's stale fallback), but nothing
made that distinction an explicit, structured signal the model was told
to act on -- so it silently papered over the gap with generic reasoning
instead of saying so.

**The fix, combining both gaps into one structural addition**: a
`data_gaps` field (list of `{player_name?, reason, detail}`) alongside
`reasoning` in `submit_recommendation`'s schema (and `RecommendResult`).
`reason: "no_signal_data"` covers Gap 2 -- `get_player_signals` now
returns an explicit `has_signals: false/true` field (distinct from
`stale`/`source_season`, which mean data exists but is old), and the
system prompt tells the model to record a `data_gaps` entry rather than
reasoning about that player generically. `reason:
"out_of_scope_capability"` covers Gap 1 -- a part of the question no
tool/signal can resolve yet, surfaced structurally instead of failing
the whole question. `reasoning` (prose) still names the gap in plain
language for the CLI/human-readable case, same as always --
`data_gaps` is the structured, UI-renderable version of the same fact
for Phase 5's web UI to eventually render distinctly, not a replacement.
Empty case is `[]`, always present, never omitted -- consistent with
`tool_calls`/`messages` already always being lists.

**`report.py` is explicitly untouched here** -- its `drop`/`start_sit`
reports already handle a no-signal player correctly (excluded from
ranking, named in a `notes` string). Whether `report.py`'s `notes`
strings should eventually migrate to this same structured `data_gaps`
shape for consistency is a documented open question, not attempted in
this unit.

**Explicitly out of scope**: building trade suggestions themselves,
`src/scheduler/refresh.py`, and any change to `report.py`.

**Addendum, found during this phase's own real-model validation of Gap
1's fix (`ANTHROPIC_API_KEY` configured locally, real roster).** Gap 2's
fix validated cleanly for real (`Jeremiyah Love` and `Kenyon Sadiq` both
correctly flagged `has_signals: false`, no fabricated reasoning). Gap 1's
fix did NOT hold up, and failed in a worse way than the original bug:
asked the real compound trade question, the model returned `data_gaps:
[]` but `reasoning` contained concrete, specific trade strategy ("package
one QB (Herbert) or a mid-tier RB... to upgrade at TE") with zero
supporting tool calls -- every call in `tool_calls` was
`get_my_roster`/`get_player_signals` on the user's own roster; no
`find_owner`, no lookup of any other team's roster anywhere. The model
found a way to *sound* like it engaged the missing capability without
actually grounding anything in real cross-team data, which let it slip
past the original "record a gap for the part you couldn't do" instruction
entirely -- the model apparently decided offering plausible-sounding
asset-packaging advice counted as "doing" the trade-partner half.

**The strengthened fix**: the system prompt now states an explicit,
stricter rule, not just an instruction to record a gap -- any specific
claim in `recommendation`/`reasoning` (a trade to make, assets to
package, a player to target from another team, anything about another
team's roster/needs/value) must be backed by an actual tool call made
that turn; if none exists, the model must say so and record an
`out_of_scope_capability` gap rather than improvising. This also closes
the structural ambiguity the user flagged: the prompt (and `find_owner`'s
own tool description) now explicitly states that `find_owner` only
answers "who owns this one named player" and is NOT a roster-comparison
or trade-fit tool, and that no tool here inspects another team's full
roster or compares needs/value across teams -- an explicit capability
boundary instead of leaving the model to infer one.

**Honest limitation on test coverage for this addendum**: the actual
failure is model behavior (choosing to fabricate plausible prose instead
of admitting a gap), not a code branch -- a fake-client test can prove
the orchestration loop faithfully passes through whatever the model
says (including a deliberately-fabricated "bad" response, which is what
`tests/test_recommend.py` now has as a documented, explicit
non-safeguard), and can prove the system prompt string contains the new
rule, but neither proves a real model actually follows it. Real-model
validation is the primary check for this addendum, not incidental --
re-running the exact real compound question is the only thing that can
confirm the fix.

## Phase 3.8: Roster-composition visibility across the league
Closes one real gap Phase 3.7's honest decline left behind: after #15,
asking "who should I trade with" correctly declined to fabricate an
answer and correctly explained why -- but the explanation ("no tool
inspects another team's roster") was itself a closeable gap, not a
permanent one. The data already exists and is already ingested (Sleeper's
own roster endpoint returns every team's roster; `lookup.py`'s
`all_rostered_players()` already proves the data is accessible
league-wide) -- nothing exposed per-team roster *composition* to the
reasoning agent. This phase closes that, and only that -- composition,
never valuation (that's Phase 3.9).

**New tool: `get_league_rosters`.** Backed by a new
`src/rag/lookup.py:all_team_rosters()` (every team's roster, grouped by
position, with a per-position count) and `team_roster_for_owner()` (the
same, filtered to one team) -- the league-wide, per-team-grouped
counterpart to the existing `my_players()`. The tool itself takes an
optional `owner_display_name`: omit it to survey every team in the
league at once (the useful shape for "which teams have surplus/need at a
position"), or give one team's owner display name to look at just that
team. Wired into `recommend.py`'s tool dispatch and
`submit_recommendation`'s tool list the same way every other tool is.

**Keeping Phase 3.7's anti-fabrication rule intact was the whole point.**
This tool tells the agent WHAT a team has -- it still says nothing about
whether a trade is fair, what a player is worth, or what to offer. The
system prompt is updated carefully, not just additively: the compound-
question guidance now says the trade-partner half is *partially*
answerable (composition, via `get_league_rosters`) but the valuation half
still isn't, and the anti-fabrication paragraph now explicitly allows a
composition claim when `get_league_rosters` actually supports it while
still blocking any claim about trade value, fairness, or what to offer.
`find_owner`'s own tool description is also corrected -- it previously
said "no tool here inspects another team's full roster," which was true
before this phase and is no longer true; leaving it unedited would have
had the model believe a now-false thing about its own capabilities. A
compound trade question should now get a real, grounded partial answer
(which teams have surplus at the weak position) plus an honest
`data_gaps` entry for the still-missing valuation piece -- not a
fully-resolved trade recommendation (Phase 3.9's job) and not a bare
decline (the old Phase 3.7 behavior, now an under-answer given the new
tool).

**Explicitly out of scope**: trade valuation itself (Phase 3.9),
`src/scheduler/refresh.py`, `report.py`.

## Phase 3.9: A crude, explicitly-labeled trade-value proxy
The remaining gap after Phase 3.8: the agent can say which teams have
surplus/need at a position, but still can't reason about whether a
specific trade is fair -- is your TE worth their spare RB? Real value
modeling (season-long projections, position-scarcity curves) is
genuinely hard and not appropriate to build in one pass. This phase
builds the honest, cheap version instead, clearly labeled as a proxy --
the same pattern `report.py`'s waiver `opportunity_score` already
establishes (an explicitly-labeled heuristic, never presented as a real
valuation model).

**The single new signal**: season-long fantasy points scored so far this
season, per player, computed from real nflverse weekly stats using this
league's actual Sleeper scoring settings (the same settings
`matchup_signals.py`/`report.py` already read), summed as-of the current
week -- strictly respecting the existing as-of-date-filtering rule, never
a future week. This is the cheapest, most defensible single-number proxy
for "how good has this player actually been this season" -- explicitly
NOT "how good will they be," NOT position-scarcity-adjusted, and NOT a
real trade-value model. Labeled exactly that everywhere it appears
(field/variable names, system prompt language, docstrings) -- e.g. a
field literally named `season_points_so_far_proxy`, never a bare `value`.

**Where it's exposed**: through `get_player_signals`'s existing output
(one more per-player number, not a new capability category), not a new
tool.

**The harder half is the prompt wording, not the computation.** With
`get_league_rosters` (3.8) plus this points-so-far proxy, the agent CAN
now attempt a rough trade suggestion -- "Team X rosters a
proxy-comparable RB to your weak-TE need" -- but the system prompt must
make it describe this as a rough, points-based comparison, never as real
trade value, and must make it name the proxy's real limitations (no
position-scarcity weighting, no accounting for the other side's own
roster needs, no injury/schedule adjustment) as part of its reasoning,
not silently omit them. Getting this qualification language right is the
actual point of this phase -- the risk is the model overstating a crude
proxy's reliability, which would quietly reopen the exact fabrication
problem Phase 3.7 fixed, just dressed up in a real-looking number instead
of pure invention.

**Explicitly out of scope**: real position-scarcity modeling,
`src/scheduler/refresh.py`, `report.py`.

## Phase 4 stretch: derived coverage classification
Real man/zone or coverage-shell labels aren't free (PFF/SIS charting is
paywalled), but they can be *derived* from NFL Big Data Bowl tracking
data on Kaggle (free, raw player x/y/speed/direction tracking).
Community precedent: unsupervised clustering and neural approaches
have been used to distinguish man vs. zone from tracking data alone.
Known limits: typically only ~9 weeks of one season released per year
(fine for backtesting, not live in-season use), and defender
orientation isn't in the data, so even published methods have real
gaps. Scope as a stretch phase, optional — not a blocker for Phase 5.
```
- [ ] Pull Big Data Bowl tracking data (Kaggle)
- [ ] Feature-engineer DB motion (distance to nearest WR, closing speed, etc.)
- [ ] Unsupervised clustering (man vs. zone) as first pass
- [ ] Backfill matchup-fit scores for eval-set weeks using classified coverage
```

## Phase 5: Productization (the final deliverable)
The goal is a real, live, multi-league product — not a polished
business, just something true: a friend can go to a URL, paste their
own Sleeper league ID, and get the same signals-backed recommendations
Ask Madden gives for Victorious Secret 3.0. Deliberately scoped small:

- **Auth**: no password/OAuth needed for v1. Sleeper's API is public
  and read-only, so registration is just "Sleeper league ID + your
  team name within that league" mapped to a session or lightweight
  account.
- **Scoring-format generalization**: `matchup_signals.py` and
  `recommend.py` currently assume half-PPR. Sleeper's league API
  already returns scoring settings per league — pull and pass those
  through instead of hardcoding. This is real work but contained to
  a handful of functions.
- **Per-league join at query time**: global signals/RAG corpus stay
  shared and refreshed on one schedule; `recommend.py` takes
  `league_id` (and derives roster/scoring/matchups from it) as a
  parameter instead of assuming one league.
- **Minimal web UI**: one page. Paste league ID, see your roster,
  ask a question, get a recommendation with reasoning. No design
  polish required — functional over pretty.
- **Platform scope**: Sleeper only for v1. ESPN/Yahoo integration is
  explicitly out of scope — their APIs are messier and this isn't
  the point being proven.
- **Cost controls**: per-user or per-day query caps to keep Claude
  API spend bounded once more than one person can hit it. No
  subscription/payment layer needed — this is a portfolio deliverable,
  not a business.
- **Hosting**: free tier (Railway/Render/Fly.io) is sufficient at
  friend-group scale.

**Success criteria**: a live URL, a handful of real users across
different leagues (not just Victorious Secret 3.0), and a README that
can honestly say "built for one league, then shipped as a product."

### Phase 5 TODO
- [ ] Pull Sleeper scoring settings per league; parameterize signals/recommend accordingly
- [ ] Build storage layer: league_id/team_id → user mapping (SQLite)
- [ ] Build API layer wrapping recommend.py, scoped per league_id
- [ ] Build minimal web frontend: register league → view roster → ask/recommend
- [ ] Add per-user/day query caps
- [ ] Deploy to free-tier host
- [ ] Get 2-3 friends in different leagues to actually use it
- [ ] README: document the "started as one league, generalized to a product" story explicitly

## Eval methodology
- **Qualitative seed set**: real, researched pregame dilemmas (e.g. Week 5 2025
  Dobbins/Harvey flex split, Addison vs. Jeudy) — verified, not invented,
  used for narrative variety and hard test cases.
- **Systematic set**: generated programmatically via nflverse — pull real
  box scores for chosen weeks, auto-build `ground_truth.jsonl`. Far more
  reliable and scalable than manual research.
- **As-of-date filtering is mandatory**: retrieval/signals must only use
  data available before kickoff for the eval week, or the eval leaks
  the answer.
- **Score two things separately**: retrieval accuracy (did it pull the
  right facts) and decision accuracy (did it recommend the
  higher-scoring option, and did the stated reasoning hold up).
- **Known limitation to state in README**: "higher score" is a proxy
  for good decision-making, not a perfect measure — a sound process
  can still lose to a fluke game. Naming this explicitly is a plus in
  interviews, not a weakness to hide.
- **Iteration loop**: if eval performance is weak without a signal
  (e.g. coverage classification), that's the trigger to revisit the
  signal list — don't add complexity speculatively.

## Cost estimate
**Single-league (Phase 1-3, personal use)**
- Data sources (nflverse, NGS, Sleeper): $0
- Claude API calls: ~$5-20/month at friend-group query volume
- Embeddings: <$5/month at this scale
- Hosting: free tier or $5-10/month VPS
- **Realistic total: $10-30/month**

**Multi-league product (Phase 5)**
- Signals/RAG refresh cost doesn't scale with number of leagues (shared, computed once)
- Claude API cost scales per-query, not per-league — bounded by per-user/day caps
- Hosting: free tier likely sufficient at friend-group-of-friend-groups scale
- **Realistic total: still $15-40/month with caps in place; revisit only if usage meaningfully exceeds friend-group scale**

## Phased TODO

### Phase 1: Foundation (RAG basics)
- [ ] Sleeper ingest: league, rosters, matchups, player pool
- [ ] Store as structured JSON/SQLite
- [ ] Chunk + embed into ChromaDB
- [ ] CLI loop: question → retrieve → answer
- [ ] Pull exact box scores via nflverse for chosen eval weeks
- [ ] Auto-generate ground_truth.jsonl from nflverse weekly stats
- [ ] Build evals/run_eval.py backtest harness (as-of-date filtering)

### Phase 2: Signals layer
**Chunk granularity (locked in, not up for re-litigation when this phase starts):**
signal/analysis content (NGS writeups, matchup-fit commentary, injury
notes, etc.) is chunked **per player** (or per player-per-week for
time-varying content like a weekly matchup writeup) — never bundled
into a team-sized blob. A whole-team chunk dilutes semantic search: a
question about one player's efficiency trend would retrieve a chunk
padded with 15 other players' irrelevant text instead of the specific
fact needed for a recommendation. The existing Phase 1 `team:{roster_id}`
whole-roster chunk (`src/rag/embed.py`) stays exactly as it is for its
own use case ("who's on this roster") — it is not replaced or forced
into this pattern, just not the model for anything new.
- [ ] nflverse ingest: play-by-play, EPA/WPA, personnel/formation
- [ ] NGS ingest: CROE, aDOT, RYOE
- [ ] Odds ingest: game script / implied totals
- [ ] Compute matchup signals (see table above)
- [ ] Store signals alongside RAG corpus, retrievable by player/matchup
- [ ] realtime.py: injury/inactive, weather, line movement (tighter refresh cadence)

### Phase 3: Reasoning layer
- [ ] recommend.py: retrieved facts + signals → Claude tool-use call → recommendation + explanation
- [ ] Expand eval set to grade recommendation quality, not just retrieval
- [ ] README write-up: architecture diagram, eval numbers, example Q&A

### Phase 3.5: Multi-turn conversation + report generation
- [x] Multi-turn conversation support in `recommend()` (optional `messages` in/out)
- [x] CLI REPL (`recommend.py --interactive`) to validate multi-turn without Phase 5's UI
- [x] `generate_report()`: start/sit report (reuses existing tools/signals)
- [x] `generate_report()`: drop report (reuses existing tools/signals)
- [x] `generate_report()`: waiver-wire pickup report (reuses existing tools/signals)
- [x] Scope (don't build yet): the season-long value + positional-need
      signals trade suggestions need, as a named follow-up -- still not
      built, see Phase 3.5's writeup above

### Phase 3.6: Prior-season signal fallback
- [x] `src/rag/retrieve.py`: `query_player_signal_with_fallback()` for the chat path
- [x] `src/reasoning/report.py`: `_load_prior_season_fallback_table()` / `_signal_row()` for the report path
- [x] Explicit `stale`/`source_season`/`source_as_of_week` markers on every affected output (raw row, report entries, prose text)
- [x] Document and justify the N=1 fallback threshold
- [x] Test coverage: no data at all, prior-season-only data, current-season data present (never stale)

### Phase 3.7: Compound questions + structured data gaps in the chat path
- [x] `submit_recommendation` tool schema + `RecommendResult`: new `data_gaps` field
- [x] `get_player_signals` tool: explicit `has_signals: true/false`, distinct from `stale`
- [x] System prompt: answer the answerable part of a compound question instead of failing it whole
- [x] System prompt: record a `no_signal_data` gap instead of reasoning from background knowledge
- [x] CLI (`--interactive` and single-question) prints `data_gaps` when non-empty
- [x] Test coverage: partial-answer compound question, zero-signal-data player, no-regression on a fully-grounded question
- [x] Addendum (found in this phase's own real-model validation): system
      prompt explicitly bars ungrounded trade advice, not just "record a
      gap" -- Gap 2 validated clean for real; Gap 1's original fix did
      not, see Phase 3.7's addendum above. Real-model re-validation still
      needed (flagged, not done in this sandbox).

### Phase 3.8: Roster-composition visibility across the league
- [x] `src/rag/lookup.py`: `all_team_rosters()` / `team_roster_for_owner()`
- [x] New `get_league_rosters` tool, wired into `recommend.py`'s dispatch and tool list
- [x] System prompt updated: composition is now partially answerable, valuation still isn't -- without contradicting itself
- [x] `find_owner`'s tool description corrected (no longer claims league-wide roster inspection is entirely impossible)
- [x] Test coverage: named-team lookup, all-teams lookup, unknown team, compound-question orchestration (composition answered, valuation still gapped)
- [ ] Real-model validation (this sandbox still has no `ANTHROPIC_API_KEY`)

### Phase 3.9: A crude, explicitly-labeled trade-value proxy
- [ ] Not started

### Phase 4: Stretch (optional)
- [ ] Derived coverage classification (Big Data Bowl tracking data)
- [ ] Discord bot wrapper
- [ ] Weekly auto-generated lineup recommendations

### Phase 5: Productization (final deliverable — see above for detail)
- [ ] Parameterize scoring settings from Sleeper API
- [ ] Storage layer for league/user mapping
- [ ] API layer wrapping recommend.py per league_id
- [ ] Minimal web frontend
- [ ] Cost/query caps
- [ ] Deploy to free-tier host
- [ ] Get real multi-league usage
