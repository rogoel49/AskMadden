# TODO

See `PROJECT_SPEC.md` for the full phased plan and signals table.

## Phase 1: Foundation (RAG basics)
- [x] Sleeper ingest: league, rosters, matchups, player pool
- [x] Store as structured JSON (`data/raw/sleeper/`)
- [ ] Chunk + embed into ChromaDB
- [ ] CLI loop: question → retrieve → answer
- [ ] Pull exact box scores via nflverse for chosen eval weeks
- [ ] Auto-generate ground_truth.jsonl from nflverse weekly stats
- [ ] Build evals/run_eval.py backtest harness (as-of-date filtering)

## Phase 2: Signals layer
- [ ] nflverse ingest: play-by-play, EPA/WPA, personnel/formation
- [ ] NGS ingest: CROE, aDOT, RYOE
- [ ] Odds ingest: game script / implied totals
- [ ] Compute matchup signals
- [ ] Store signals alongside RAG corpus, retrievable by player/matchup
- [ ] realtime.py: injury/inactive, weather, line movement

## Phase 3: Reasoning layer
- [ ] recommend.py: retrieved facts + signals → Claude tool-use call → recommendation + explanation
- [ ] Expand eval set to grade recommendation quality, not just retrieval
- [ ] README write-up: architecture diagram, eval numbers, example Q&A

## Phase 4: Stretch
- [ ] Derived coverage classification (Big Data Bowl tracking data)
- [ ] Discord bot wrapper
- [ ] Weekly auto-generated lineup recommendations
