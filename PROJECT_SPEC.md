# Ask Madden — Project Spec

An AI fantasy football assistant that goes beyond rankings — it explains *why*.
League: Victorious Secret 3.0 (Sleeper league ID 1389341490030862336), 12 teams, half-PPR.

## Pitch (for README.md)
Most start/sit tools give you a projection number. Ask Madden combines
retrieval-augmented generation over live league/player data with a
computed matchup-signals layer (defensive tendencies, coverage-adjusted
efficiency, game script) so recommendations come with real reasoning:
not just "start Player A," but "start Player A — this defense allows
the 4th-most rush yards to RBs and Player A's efficiency trend is up
over his last 3 games."

## Architecture
```
Sleeper / nflverse / NGS / odds / realtime
        → signals layer (derived matchup features)
        → RAG corpus (chunked, embedded, retrievable)
        → Claude (tool-use agent)
        → recommendation + explanation
```
Agent = Claude API with tool use (function calling). The model calls
retrieval/signals functions as tools rather than relying on a heavy
framework — a lighter, more legible pattern.

## Repo structure
```
ask-madden/
├── README.md
├── TODO.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                    # gitignored, too large to commit
│   └── processed/              # small computed samples, can commit
├── src/
│   ├── ingest/
│   │   ├── sleeper.py           # league/roster/matchup pulls
│   │   ├── nflverse.py          # play-by-play, EPA/WPA
│   │   ├── ngs.py               # CROE, aDOT, RYOE
│   │   ├── odds.py              # Vegas lines / game script
│   │   └── realtime.py          # injury/inactive status, weather, line movement (fast-refresh tier)
│   ├── signals/
│   │   └── matchup_signals.py   # derived features, see table below
│   ├── rag/
│   │   ├── embed.py             # chunk + embed into ChromaDB
│   │   └── retrieve.py
│   ├── reasoning/
│   │   └── recommend.py         # combines retrieval + signals → recommendation + explanation
│   ├── scheduler/
│   │   └── refresh.py           # runs ingest → signals → embed on a schedule; realtime.py on a tighter cadence
│   └── cli.py
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

## Phase 4 stretch: derived coverage classification
Real man/zone or coverage-shell labels aren't free (PFF/SIS charting is
paywalled), but they can be *derived* from NFL Big Data Bowl tracking
data on Kaggle (free, raw player x/y/speed/direction tracking).
Community precedent: unsupervised clustering and neural approaches
have been used to distinguish man vs. zone from tracking data alone.
Known limits: typically only ~9 weeks of one season released per year
(fine for backtesting, not live in-season use), and defender
orientation isn't in the data, so even published methods have real
gaps. Scope as a stretch phase, not a v1 blocker.
```
- [ ] Pull Big Data Bowl tracking data (Kaggle)
- [ ] Feature-engineer DB motion (distance to nearest WR, closing speed, etc.)
- [ ] Unsupervised clustering (man vs. zone) as first pass
- [ ] Backfill matchup-fit scores for eval-set weeks using classified coverage
```

## Eval methodology
- **Qualitative seed set**: real, researched pregame dilemmas (e.g. Week 5 2025
  Dobbins/Harvey flex split, Addison vs. Jeudy) — verified, not invented,
  used for narrative variety and hard test cases. **Status: deferred, tracked
  in `TODO.md`, not started.** Needs actual research per dilemma (can't be
  generated), so it's scoped for whenever Phase 3's decision-accuracy
  grading makes these gradeable rather than built speculatively now.
- **Systematic set**: generated programmatically via nflverse — pull real
  box scores for chosen weeks, auto-build `ground_truth.jsonl`. Far more
  reliable and scalable than manual research. **Note:** `evals/build_eval_questions.py`
  today generates a related but distinct thing — plain fact-retrieval
  questions from our own Sleeper data (rosters, matchup scores), not the
  nflverse box-score comparison dilemmas described here. Don't conflate the
  two when building Phase 3's decision-accuracy grading against this
  systematic set.
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

## Cost estimate (running for you + friends)
- Data sources (nflverse, NGS, Sleeper): $0
- Claude API calls: ~$5-20/month at friend-group query volume
- Embeddings: <$5/month at this scale
- Hosting: free tier (Railway/Render/Fly.io) or $5-10/month VPS for always-on
- Domain (optional): ~$12-15/year
- **Realistic total: $10-30/month**

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

### Phase 4: Stretch
- [ ] Derived coverage classification (Big Data Bowl tracking data)
- [ ] Discord bot wrapper
- [ ] Weekly auto-generated lineup recommendations
