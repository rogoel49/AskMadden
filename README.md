# Ask Madden

An AI fantasy football assistant that goes beyond rankings — it explains *why*.

Most start/sit tools give you a projection number. Ask Madden combines
retrieval-augmented generation over live league/player data with a
computed matchup-signals layer (defensive tendencies, coverage-adjusted
efficiency, game script) so recommendations come with real reasoning:
not just "start Player A," but "start Player A — this defense allows
the 4th-most rush yards to RBs and Player A's efficiency trend is up
over his last 3 games."

Built for the Victorious Secret 3.0 Sleeper league (12 teams, half-PPR).

## Architecture
```
Sleeper / nflverse / NGS / odds / realtime
    → signals layer (derived matchup features)
    → RAG corpus (chunked, embedded, retrievable)
    → Claude (tool-use agent)
    → recommendation + explanation
```
The agent is the Claude API with tool use (function calling) calling
retrieval/signals functions directly as tools — no LangChain or similar
framework.

See `PROJECT_SPEC.md` for the full architecture, signals table, eval
methodology, and phased plan. See `TODO.md` for current progress.

## Setup
```
pip install -r requirements.txt
cp .env.example .env
```

## Usage
Pull the latest Sleeper league data (league, rosters, users, matchups,
player pool) to `data/raw/sleeper/`:
```
python -m src.ingest.sleeper
```

## Tests
```
pytest
```
