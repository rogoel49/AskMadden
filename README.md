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
composition) are complete and validated against real data, and
productization (Phase 5) is most of the way there: 5.1 league/scoring
parameterization, 5.2 the multi-user API + storage layer, 5.3 the
responsive frontend wired to it, 5.4 PWA installability, and 5.7 the
automated data refresh are done; 5.6's "one app serves everything"
step is done and the deployment config (Dockerfile + fly.toml) is
written but not yet deployed; 5.5's landing copy is done and the eval
band shows real numbers (retrieval 80/84 across four leagues, decision
206/399 on 2024 dilemmas, both 2026-09-20 -- see Evals below). It is
deployed at https://askmadden.fly.dev (Fly.io, one machine + volume). It runs locally against live in-season data today (first real
in-season use: 2026-09-16, two leagues).** Phase 4 is optional stretch work; Phase 6
(a trade-value proxy) is deliberately deferred past Phase 5.

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
never a trade valuation) — plus the multi-league API/storage layer,
the responsive web UI (`design/askmadden-ui-mockup.html`, one file:
landing view, login/league picker, and an app shell that fills a real
phone's viewport below 900px and reflows to a sidebar layout above
it), Home Screen installability, and a background refresh that keeps
signals and league data current. What's still ahead is the public
deployment (5.6) and the landing page's real eval numbers (5.5) — see
`TODO.md` for the detailed phase-by-phase log and `PROJECT_SPEC.md`
for the full architecture, signals table, and Phase 5 plan.

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
python -m src.ingest.nflverse --season 2026
python -m src.signals.matchup_signals --season 2026 --as-of-week N
```

Chunk and embed league data and computed signals into a local ChromaDB
collection at `data/chroma/`:
```
python -m src.rag.embed
```

Or let the scheduler do all three on a cadence (Phase 5.7) — the
signals table once, then every ingested league's Sleeper pull and
re-embed. On a new machine, backfill the season's earlier weeks first,
then run the loop (or just start the server, which runs it for you):
```
python -m src.scheduler.refresh --once --backfill   # first run
python -m src.scheduler.refresh --once              # one cycle (what a cron job calls)
python -m src.scheduler.refresh                     # the loop, foreground
python -m src.scheduler.refresh --status            # did it work?
```
It picks the as-of week from nflverse's own completed-game data (last
fully-completed week + 1), never from Sleeper's clock, so a cycle can
never compute a week whose history is still being played. Cadence
defaults to 6 hours — nflverse rebuilds play-by-play at most twice a
day and Next Gen Stats once a day, so polling faster is wasted work —
and is configurable via `ASKMADDEN_REFRESH_INTERVAL_SECONDS`.

Every command below answers for one specific Sleeper league: pass
`--league-id <id>`, or set `SLEEPER_LEAGUE_ID` in `.env` (the same
variable the ingest step uses) and omit the flag. The league ID is
checked against what's actually ingested — asking about a league whose
data isn't in `data/raw/sleeper/` is an error, never a silent answer
from a different league's roster.

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

## The app: API + web UI (Phases 5.2-5.7)
The same `recommend()` / `generate_report()` behind HTTP, multi-user
and multi-league: log in with a Sleeper username (no password — the
API is public and read-only), pick one of your leagues, ask questions.
One FastAPI app serves all of it — the JSON API under `/api/`, the
responsive frontend at `/ui/` (`/` redirects there), and, in the
background, the Phase 5.7 data refresh:
```
uvicorn src.api.main:app --reload
```
then open http://127.0.0.1:8000/ — landing page, log in with your
Sleeper username, pick a league. To reach it from a phone on the same
WiFi (Phase 5.4 — "Add to Home Screen"; manifest, icons, and iOS head
tags are in design/, see TODO.md's 5.4 entry for the real-device
steps), run it bound to your network interface instead:
```
python -m web.dev_server
```
and open `http://<your machine's LAN IP>:8000/`. Local network only;
the public deployment is the Deploying section below.

The refresh thread keeps data current while the server is up
(`ASKMADDEN_REFRESH_ENABLED=0` turns it off; `python -m
src.scheduler.refresh --status` or `GET /api/health` shows the last
cycle). It is the right shape for one long-running process; a host
that runs more than one replica or sleeps idle ones should turn it off
and run `python -m src.scheduler.refresh --once` from a cron job or
worker instead.

Endpoints: `GET /api/health` (process up + last refresh cycle),
`POST /api/leagues` (username → your leagues), `POST
/api/sessions` (pick a league; its data is ingested on first use),
`GET /api/roster`, `GET /api/reports/{start_sit|drop|waiver_pickups}`,
`POST /api/chat` (pass back the returned `messages` to continue a
conversation). Chat responses carry `data_gaps` and per-player
`signals_consulted` stale markers as separate fields. Claude-backed
chat is capped per user per day (`ASKMADDEN_DAILY_QUERY_CAP`, default
25); reports are free. Interactive docs at `/docs` once it's running.

### Deploying (Phase 5.6)
The whole product is one container: `Dockerfile` runs
`src.api.main:app` (API + `/ui` frontend + in-process refresh) behind
`deploy/entrypoint.sh`, which seeds the committed reference signals
tables onto the host's persistent volume and honors the host's `$PORT`.
`fly.toml` is the checked-in config for Fly.io; Railway or Render run
the same image given a persistent volume at `/app/data` and the env
vars listed at the top of `deploy/entrypoint.sh` (only
`ANTHROPIC_API_KEY` is required). Everything the server writes lives
under `data/`, so the one volume persists leagues, the Chroma index,
the signals tables, and the SQLite users/sessions DB across deploys.

Fly.io, from the repo root (install `flyctl` first:
https://fly.io/docs/flyctl/install/):
```
fly auth login
fly launch --copy-config --no-deploy        # creates the app from fly.toml; say no to Postgres/Redis
fly volumes create askmadden_data --size 3 --region iad
fly secrets set ANTHROPIC_API_KEY=sk-ant-...
fly deploy
fly ssh console -C "python -m src.scheduler.refresh --once --backfill"   # once: this season's earlier weeks
fly open /api/health
```
Notes: the machine stays up between requests on purpose (the refresh
runs in-process, and `fly.toml` says so), which is a few dollars a
month at 2GB — there is no free tier that gives a persistent disk and
enough memory for a refresh cycle. The first embed after a deploy
downloads chromadb's embedding model once (~80MB, kept on the volume
after that). `fly logs` shows each refresh cycle; `fly ssh console -C
"python -m src.scheduler.refresh --status"` is the same check as
locally. HTTPS is automatic, which is also what Android Chrome needs
for the install prompt (Phase 5.4).

**Custom domain** (e.g. askmadden.com), once `fly deploy` works and
`https://askmadden.fly.dev/` loads. Buy the domain at any registrar
(Cloudflare Registrar, Porkbun, and Namecheap all sell .com at roughly
$10/year), then from the repo root:
```
fly certs add askmadden.com
fly certs add www.askmadden.com
fly ips list
```
`fly certs add` prints the DNS records to create at the registrar: an
`A` record for `askmadden.com` pointing at the app's IPv4 from `fly ips
list`, an `AAAA` record pointing at its IPv6, and a `CNAME` for `www`
pointing at `askmadden.fly.dev`. If the registrar is Cloudflare, leave
the proxy (orange cloud) OFF for these records so Fly can issue the
certificate. Then:
```
fly certs check askmadden.com
```
until it reports the certificate as issued (usually a few minutes after
DNS propagates; up to an hour). HTTPS on the custom domain is automatic
after that, and `fly.toml`'s `force_https` already redirects http://.
Note: `app = "askmadden"` in `fly.toml` is a global Fly name; if `fly
launch` says it's taken, pick another (`askmadden-rg`, say) — only the
`*.fly.dev` URL changes, the custom domain doesn't care.

The image has not yet been built or deployed for real — Docker and
flyctl aren't installed on the machine this was written on. See
TODO.md's 5.6 entry for what was verified (the app serving everything
on one origin, against a real server and a real browser) versus what
the first `fly deploy` still has to prove.

### Refreshing signals while the server is running
New games get played, so the signals table needs recomputing. Normally
the Phase 5.7 scheduler does this for you on its own cadence (see
`src/scheduler/refresh.py` above) — this section is about the times you
do it by hand: a refresh you want *now* rather than at the next cycle,
or a deployment running with `ASKMADDEN_REFRESH_ENABLED=0`.

Run the two ingest/compute commands from the Setup section above for the
new week:
```
python -m src.ingest.nflverse --season 2026
python -m src.signals.matchup_signals --season 2026 --as-of-week N
```
You do **not** need to restart the server. Reports re-read the signals
parquet on every request, and the server now re-embeds each league's
Chroma collection automatically the first time it sees the signals
table has changed, so the chat path picks the new numbers up too (the
one request that triggers the re-embed is slower than usual).

One thing the signals refresh alone does *not* change: which week the
reports are bounded to. `as_of_week` is inferred from Sleeper's own
`nfl_state.json` (its `week` field — not `display_week`, which lags
until midweek and, until 2026-09-16, made every card stale the day
after real games; the scheduler's own cycle re-pulls this file, so
this only matters for a by-hand refresh), so until you also re-run
```
python -m src.ingest.sleeper
```
the newly computed week is treated as future data and filtered out of
the default report — which is the as-of-week rule doing its job, not a
cache. Re-run the Sleeper ingest (or pass `?as_of_week=N` explicitly)
to move the reports onto the new week.

One-time, if a league was already ingested before this auto-re-embed
existed: give it a single `POST /api/sessions` with `{"refresh": true}`
(or re-run `python -m src.rag.embed`) so its collection starts from a
known-good state. Refreshes after that are picked up on their own. See
TODO.md's "Signals refresh on a running server" entry for the full
investigation, including how this and the scheduler's own re-embed fit
together (they are the trigger and read halves of the same guarantee,
and a scheduler cycle deliberately records the fingerprint so the next
request does not rebuild what it already built).

## Evals
Pull real weekly box scores from nflverse and turn them into ground
truth (fantasy points computed using the league's actual scoring
settings — read from the same ingested `league.json` the agent uses,
`--league-id`/`SLEEPER_LEAGUE_ID` as above — never hand-authored):
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


### Results so far (2026-09-20)
| Metric | Result | Scope |
|---|---|---|
| Retrieval accuracy | **80 / 84 (95%)** | roster + matchup-score questions, 4 real leagues, as-of-week filtered; `evals/results/2026-09-20_retrieval_run.json` |
| Decision accuracy | **206 / 399 (52%)** | 400 programmatic 2024 week-5 start/sit dilemmas (same position, both players ≥ 8 pts), signals as-of week 5, Sonnet 4.5; by actual point gap: <5 pts 47%, 5-10 52%, 10-20 61%, 20+ 69%; `evals/results/2026-09-20_decision_run.json` |
| Ranking score, offline (`evals/fit_ranking_weights.py`) | **61.8%** pairwise on 19,283 held-out 2024 pairs (72% when the gap was 10+ pts) | weights fitted on 2023; the hand-set weights they replaced scored 57.6%, points-per-game alone 61.5%; `evals/results/2026-09-20_ranking_fit_blend4.json` |
| Matchup-fit accuracy | n/a | needs Phase 4's coverage classification |

The decision number above predates the fitted, points-aware ranking
(2026-09-21); the offline row is what that ranking does on its own, and
a live rerun is the open item. The 52% is honest and unflattering: barely above a coin
flip overall. Half the dilemmas were pairs whose actual outcomes were
within 5 points of each other, which no week-4-signals model should be
expected to call, and on those it scored 47%; where the gap was 10+
points it scored 62% (57/92). Whether a better signal set moves that is
exactly what Phase 4/6/7 work should be graded on -- this is the
baseline they have to beat.

The retrieval run found a product bug on its first pass (54/84): the
per-player signal chunks that make up ~97% of each league's index were
crowding roster and matchup chunks out of the chat's league-info search.
That search now excludes signal chunks (`retrieve.LEAGUE_INFO_ONLY`);
the eval measures that same path. Both numbers are scored separately, as
CLAUDE.md requires.

## Tests
```
pytest
```
Currently 346/346 passing. See `TODO.md` for the session-by-session
log of what was validated against real data versus what still needs a
live re-run (a few items are flagged as needing a machine with both
`ANTHROPIC_API_KEY` and live Sleeper API access, which this project's
dev sandbox doesn't have).
