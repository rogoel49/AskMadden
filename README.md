# Ask Madden

An AI fantasy football assistant that goes beyond rankings — it explains *why*.

Most start/sit tools give you a projection number. Ask Madden combines
retrieval-augmented generation over live league/player data with a
computed matchup-signals layer (defensive tendencies, coverage-adjusted
efficiency, game script) so recommendations come with real reasoning:
not just "start Player A," but "start Player A — this defense allows
the 4th-most rush yards to RBs and Player A's efficiency trend is up
over his last 3 games."

Built and validated first against Victorious Secret 3.0 (Sleeper league
ID 1389341490030862336, 12 teams, half-PPR); the end goal (Phase 5,
below) is a small hosted product where anyone can paste in their own
Sleeper league ID and get the same recommendations for their own roster.
**Current status: still in Phase 1 (single-league foundation) —
multi-league productization hasn't started.**

## Architecture
Target end-state (Phase 5):
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
retrieval/signals functions directly as tools — no LangChain or similar
framework. Key property that makes productization cheap later: signals
and the RAG corpus are computed from NFL-wide sources, not tied to any
one league — only roster ownership, scoring settings, and matchup
schedule are league-specific. Today, this repo only implements the
Sleeper ingest → RAG pipeline for one league (see Usage below); the
signals layer, reasoning agent, and multi-league API/web layers are
all still ahead — see `TODO.md`.

See `PROJECT_SPEC.md` for the full architecture, signals table, eval
methodology, and phased plan (including Phase 5). See `TODO.md` for
current progress.

## Setup
```
pip install -r requirements.txt
cp .env.example .env
```

## Usage
Pull the latest Sleeper league data (league, rosters, users, matchups,
transactions, player pool) to `data/raw/sleeper/`:
```
python -m src.ingest.sleeper
```

Chunk that data (team rosters, league settings, matchups, transactions)
and embed it into a local ChromaDB collection at `data/chroma/`:
```
python -m src.rag.embed
```

Ask questions interactively:
```
python -m src.cli
```

## Evals
Pull real weekly box scores from nflverse and turn them into ground truth
(fantasy points computed using this league's actual scoring settings):
```
python -m src.ingest.nflverse --season 2024
python -m evals.build_ground_truth --season 2024 --weeks 1 2 3 4 5
```

Generate retrieval eval questions from the current Sleeper pull, and run
the backtest harness (as-of-week filtered — a question about week N is
never graded using week N+1 data):
```
python -m evals.build_eval_questions
python -m evals.run_eval
```
This currently scores **retrieval accuracy** only (did the RAG pipeline
surface the right facts). **Decision accuracy** (did it recommend the
higher-scoring player) needs a reasoning agent to grade against
`evals/ground_truth.jsonl` — that's Phase 3's `recommend.py`, not yet
built. See `PROJECT_SPEC.md`'s eval methodology for why these are scored
separately.

## Tests
```
pytest
```
