# Ask Madden

AI fantasy football assistant. RAG + computed matchup signals + Claude
tool-use reasoning agent, for the "Victorious Secret 3.0" Sleeper league
(league ID 1389341490030862336, 12 teams, half-PPR).

Goal beyond the code itself: daily commits, real progress, eventually a
portfolio/resume piece. Prefer small real commits over padding.

## Architecture
```
Sleeper / nflverse / NGS / odds / realtime
    → signals layer (derived matchup features)
    → RAG corpus (chunked, embedded, retrievable)
    → Claude (tool-use agent)
    → recommendation + explanation
```
Agent = Claude API with tool use (function calling), calling
retrieval/signals functions as tools. No LangChain or similar framework
— keep the tool-use loop native and legible.

## Repo structure
```
src/
├── ingest/       # sleeper.py, nflverse.py, ngs.py, odds.py, realtime.py
├── signals/      # matchup_signals.py — derived features
├── rag/          # embed.py, retrieve.py — ChromaDB
├── reasoning/     # recommend.py — combines retrieval + signals via Claude
├── scheduler/     # refresh.py — runs ingest→signals→embed; realtime.py on tighter cadence
└── cli.py
evals/
├── eval_questions.jsonl   # qualitative (researched) + nflverse-generated
├── ground_truth.jsonl      # generated from nflverse, never hand-authored
├── run_eval.py              # as-of-date filtered backtest harness — no future leakage
└── results/YYYY-MM-DD_run.json
```

## Key conventions
- **Data provenance matters**: mark every signal as measured (direct
  from source) or derived/modeled (inferred, e.g. matchup-fit score).
  Never present a modeled proxy as a direct stat.
- **Eval integrity**: `run_eval.py` must only use data available before
  the eval week's kickoff. No leaking post-game data into retrieval.
- **Two eval scores, tracked separately**: retrieval accuracy (right
  facts pulled) and decision accuracy (right recommendation + sound
  stated reasoning).
- Free data sources only for v1-v3: Sleeper, nflverse, NGS public site,
  a free-tier odds API. Coverage-shell classification (Big Data Bowl
  tracking data) is a Phase 4 stretch goal, not a v1-v3 dependency.

## Current phase
Phase 1: Foundation (RAG basics). Start here:
1. Sleeper ingest (league/rosters/matchups)
2. Store as structured JSON/SQLite
3. Chunk + embed into ChromaDB
4. CLI Q&A loop
5. Pull box scores via nflverse for eval weeks → generate ground_truth.jsonl
6. Build run_eval.py backtest harness

Full phased plan (Phase 2 signals, Phase 3 reasoning, Phase 4 stretch)
lives in `PROJECT_SPEC.md` in this repo — read it for anything not
covered here.

## Commands
- Install deps: `pip install -r requirements.txt`
- Run Sleeper ingest: `python -m src.ingest.sleeper`
- Run tests: `python -m pytest`
