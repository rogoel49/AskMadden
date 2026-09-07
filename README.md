# Ask Madden

An AI fantasy football assistant that goes beyond rankings — it
explains *why*.

Most start/sit tools give you a projection number. Ask Madden combines
retrieval-augmented generation over live league/player data with a
computed matchup-signals layer (defensive tendencies, coverage-adjusted
efficiency, game script) so recommendations come with real reasoning:
not just "start Player A," but "start Player A — this defense allows
the 4th-most rush yards to RBs and Player A's efficiency trend is up
over his last 3 games."

Built and validated first against Victorious Secret 3.0 (Sleeper
league ID 1389341490030862336, 12 teams, half-PPR); the end goal
(Phase 5, below) is a small hosted product where anyone can connect
their own Sleeper account, pick a league, and get the same
recommendations for their own roster. **Current status: Phases 1–3.8
(ingest, signals, RAG, reasoning agent, reports, league-wide roster
composition) are complete and validated against real data —
productization (Phase 5) has not started.** Phase 4 is optional
stretch work; Phase 6 (a trade-value proxy) is deliberately deferred
past Phase 5.

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
The agent is the Claude API with tool use (function calling) calling
retrieval/signals functions directly as tools — no LangChain or
similar framework. Key property that makes productization cheap:
signals and the RAG corpus are computed from NFL-wide sources, not
tied to any one league — only roster ownership, scoring settings, and
matchup schedule are league-specific. That separation has held from
Phase 1 onward and is verified as part of every phase's own
validation (no signals or RAG module takes a league_id, roster, or
scoring parameter anywhere).

Today, this repo implements the full single-league pipeline: Sleeper
ingest, the signals layer, RAG retrieval with per-player identity
resolution, a multi-turn Claude tool-use reasoning agent, structured
report generation (start/sit, drop, waiver pickups), and league-wide
roster-composition visibility via the agent's `get_league_rosters`
tool (which teams have surplus/need at a position — composition only,
never a trade valuation). What's still ahead is the multi-league
API/storage/web layer — see `TODO.md` for the detailed phase-by-phase
log and `PROJECT_SPEC.md` for the full architecture, signals table,
and Phase 5 plan. A static UI prototype for Phase 5 already exists at
`design/askmadden-ui-mockup.html`.

## Setup
```
pip install -r requirements.txt
cp .env.example .env
```

Set `ANTHROPIC_API_KEY` in `.env` to use `recommend.py`'s chat/report
path — everything else (ingest, signals, embedding, retrieval evals)
works without it. `recommend()` and `generate_report()` load `.env`
themselves, so they work the same whether called via the CLI or
imported directly.

## Usage
Pull the latest Sleeper league data (league, rosters, users, matchups,
transactions, player pool) to `data/raw/sleeper/`:
```
python -m src.ingest.sleeper
```

Pull nflverse play-by-play and NGS data, and compute the signals table
(defense run-funnel rate, red zone share, efficiency trend, target
share, game script, aDOT/RYOE/CROE-proxy — all as-of-week filtered):
```
python -m src.ingest.nflverse --season 2025
python -m src.signals.matchup_signals --season 2025 --as-of-week N
```

Chunk and embed league data and computed signals into a local ChromaDB
collection at `data/chroma/`:
```
python -m src.rag.embed
```

Ask a single question, backed by retrieval + signals + a Claude
tool-use agent. The agent can look at your own roster, a named
player's signals, your team record and current matchup, and — league-
wide, via `get_league_rosters` — every team's roster composition (e.g.
"which teams have surplus at a position I'm weak at"):
```
python -m src.reasoning.recommend "should I start Player A or Player B this week?"
```

Or hold a multi-turn conversation (the agent can ask a clarifying
question and continue in the same thread):
```
python -m src.reasoning.recommend --interactive
```

Generate a structured report across your whole roster instead of one
question at a time:
```
python -m src.reasoning.recommend --report start_sit
python -m src.reasoning.recommend --report drop
python -m src.reasoning.recommend --report waiver_pickups
```

("Trade suggestions" and real trade valuation are deliberately not
built yet — the agent can tell you which teams have roster surplus at
a position you're weak at, but not whether a specific trade is fair or
what to offer. See `PROJECT_SPEC.md`'s Phase 6 section for the
planned, explicitly-labeled proxy approach.)

Every report and chat answer labels its own gaps rather than papering
over them: a player with only last season's data gets an explicit
`[STALE]` prefix and a `source_season` marker instead of being
presented as current; a player with no data at all is flagged
`has_signals: false` instead of the model reasoning from general
knowledge; and a question with an unanswerable part (e.g. trade
valuation) gets a `data_gaps` entry instead of the model improvising
an answer it can't back up.

## Evals
Pull real weekly box scores from nflverse and turn them into ground
truth (fantasy points computed using this league's actual scoring
settings, never hand-authored):
```
python -m src.ingest.nflverse --season 2024
python -m evals.build_ground_truth --season 2024 --weeks 1 2 3 4 5
```

Retrieval accuracy (did the RAG pipeline surface the right facts),
as-of-week filtered so a question about week N is never graded using
week N+1 data:
```
python -m evals.build_eval_questions
python -m evals.run_eval
```

Decision accuracy (did the agent recommend the higher-scoring player,
scored separately from retrieval accuracy per this project's own
eval-methodology rule):
```
python -m evals.build_decision_questions
python -m evals.run_decision_eval
```

Known limitation, stated rather than hidden: "higher score" is a
proxy for good decision-making, not a perfect measure — a sound
process can still lose to a fluke game.

## Tests
```
pytest
```
Currently 148/148 passing. See `TODO.md` for the session-by-session
log of what was validated against real data versus what still needs a
live re-run (a few items are flagged as needing a machine with both
`ANTHROPIC_API_KEY` and live Sleeper API access, which this project's
dev sandbox doesn't have).
