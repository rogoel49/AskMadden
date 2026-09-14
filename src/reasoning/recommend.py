"""Phase 3 reasoning layer: retrieved facts + computed signals -> a
Claude tool-use agent -> a recommendation with an explicit explanation.

This supersedes src/cli.py's hardcoded routing (semantic retrieve.query()
by default, with a single carved-out string-matching rule for "my"
questions). Here the model itself decides which tool answers a given
question -- structured roster lookup (Sleeper data), structured
name-resolved signal lookup (nflverse data), or semantic search over
general league info -- and is required to ground its final answer in
whatever those tools actually returned, ending with a single
submit_recommendation call so the result is a parseable structure, not
free text the caller has to guess at.

**Per-league join lives here, not in matchup_signals.py or rag/**: this
module is the one place that reads this league's Sleeper roster and
scoring_settings and puts them in front of the model (in the system
prompt) alongside the league-agnostic signal corpus. Nothing about a
specific league or scoring format leaks into src/signals/ or src/rag/ --
see CLAUDE.md's key architectural principle.

**Which league (Phase 5.1)**: recommend() takes an explicit, required
league_id and resolves it via src/reasoning/league.py's load_league(),
which reads the league's real Sleeper scoring_settings from the
ingested league.json in raw_dir and refuses (LeagueMismatchError) if
that directory holds a different league. Before Phase 5.1 the league
was implicit -- whatever src.ingest.sleeper last wrote to
data/raw/sleeper/ -- with no check at all. Scoring was never hardcoded
here (this module has read the real scoring_settings since Phase 3;
see league.py's docstring for what the Phase 5.1 trace actually found);
what changed is that the league is now named and verified rather than
assumed. Per-league storage/auth/API is still Phase 5.2's job.

**Named-player bug fix**: retrieve.py's query() is pure embedding
similarity search, which can't tell two same-surname NFL players apart
(see src/rag/player_index.py's docstring for the concrete bug this
caused). get_player_signals below never calls query() for a named
player -- it always resolves the name to a specific player_id via
player_index.resolve_player() first (structured, exact-then-fuzzy
against the real player list) and only then fetches that exact player's
chunk via retrieve.query_player_signal()'s metadata filter. An
ambiguous resolution is reported to the model as ambiguous, with every
candidate, rather than ever being silently guessed.

**Head-to-head / start-sit verdicts come from the shared ranking, not
free reasoning (Chat/Feed alignment)**: real usage testing found Chat and
the Feed's start_sit report disagreeing on the same real decision over
the same signals ("Herbert or Mahomes?" -- Chat "defaulted to Mahomes
based on his consistently elite performance history" once the
current-season signal went stale, an ungrounded claim of exactly the kind
Phase 3.7's addendum bars for trade advice). The rank_players tool below
calls src/reasoning/ranking.py's rank_candidates() -- the identical code
generate_report()'s start_sit report uses -- and the system prompt
requires the model to call it for any comparison between named players
and to state its verdict, or honestly report a tie / insufficient data,
never a pick from general knowledge. recommend()'s loop also enforces the
part of that which is checkable in code: a submit_recommendation whose
player_id contradicts a rank_players result from the same call (a
different compared player than the ranking recommended, or a pick at all
when the ranking was tied/ungroundable) is bounced back to the model as a
tool error to resubmit, rather than passed through. See ranking.py's
docstring.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from src.rag import lookup, player_index, retrieve
from src.rag.embed import CHROMA_DIR, RAW_DIR
from src.reasoning import ranking
from src.reasoning.league import LeagueConfig, load_league
from src.reasoning.ranking import SIGNALS_DIR

DEFAULT_MODEL = os.environ.get("ASKMADDEN_MODEL", "claude-sonnet-4-5-20250929")
MAX_TOOL_TURNS = 8

_DOTENV_LOADED = False


def load_dotenv_once() -> bool:
    """Load .env into the environment the first time this is called in a
    process; a no-op afterwards. Returns True only on the call that
    actually loaded.

    PR #18 made recommend()/generate_report() call load_dotenv()
    themselves so a direct import behaves like the CLI. Phase 5.2's
    investigation checked whether that placement is safe for a
    long-running API server: it was correct but per-request -- every
    call walked the filesystem for a .env and re-parsed it, which is
    harmless for a one-shot CLI process (exactly one call) and pure
    waste for a server handling many requests (load_dotenv() never
    overrides an existing variable, so repeat calls could never even
    change anything). This memo keeps the direct-import guarantee (the
    first call, however it's reached, loads .env) while making every
    later call free. Tests reset _DOTENV_LOADED via tests/conftest.py.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return False
    load_dotenv()
    _DOTENV_LOADED = True
    return True


TOOLS: list[dict] = [
    {
        "name": "get_my_roster",
        "description": (
            "Structured lookup of the user's own fantasy roster in this league (exact Sleeper "
            "data, not a guess). Use for any 'my'-flavored question (my roster, my QBs, is "
            "Player X on my team, ...)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "position": {
                    "type": "string",
                    "description": "Optional position filter, e.g. 'RB'. Omit for the full roster.",
                }
            },
        },
    },
    {
        "name": "find_owner",
        "description": (
            "Structured lookup of which fantasy team in this league rosters a given player "
            "(exact Sleeper data). Use to answer 'who owns Player X' or to check whether a "
            "player is rostered in this league at all. This only resolves one named player to "
            "one owning team -- it does NOT compare rosters or evaluate trade value itself (use "
            "get_league_rosters for roster composition across teams; there is still no tool for "
            "trade value/fairness -- see get_league_rosters and submit_recommendation's data_gaps "
            "for that boundary)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"player_name": {"type": "string"}},
            "required": ["player_name"],
        },
    },
    {
        "name": "get_league_rosters",
        "description": (
            "Structured, league-wide roster-COMPOSITION lookup (exact Sleeper data) -- every "
            "team's roster grouped by position, with a per-position count for each team. Use this "
            "to identify which teams are deep or shallow at a given position (e.g. 'Team X rosters "
            "only 1 TE, Team Y rosters 4 RBs') -- genuine, grounded composition reasoning. Omit "
            "owner_display_name to get every team in the league at once (the useful shape for "
            "surveying who might have surplus/need at a position across the whole league); give it "
            "to look at one specific team instead. This tool tells you WHAT a team has -- it does "
            "NOT tell you whether a trade is fair, what a player is worth, or what to offer. Never "
            "use its output alone to suggest a specific trade or claim one side benefits -- that's "
            "still a data_gaps entry (reason: out_of_scope_capability), not something to conclude "
            "from roster composition."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "owner_display_name": {
                    "type": "string",
                    "description": (
                        "Optional -- omit to get every team in the league at once; give a specific "
                        "team's owner display name to look at just that one team's composition."
                    ),
                }
            },
        },
    },
    {
        "name": "get_player_signals",
        "description": (
            "Structured, name-resolved lookup of a real NFL player's computed matchup signals "
            "(recent efficiency trend, red zone role share, target share, opponent defense "
            "tendency, implied team total, aDOT/RYOE/CPOE as available). ALWAYS use this for any "
            "question about a specific named player -- never assume which real player a name "
            "refers to yourself. This resolves the name against the real player list first: if "
            "more than one real player plausibly matches (e.g. two players share a surname), it "
            "returns every candidate instead of picking one -- when that happens, ask for the "
            "player's team or position to disambiguate rather than guessing which one was meant."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"player_name": {"type": "string"}},
            "required": ["player_name"],
        },
    },
    {
        "name": "rank_players",
        "description": (
            "Deterministic head-to-head ranking of two or more named players by the same composite "
            "opportunity score the Feed's start/sit report uses -- THE start/sit verdict. REQUIRED for "
            "any question comparing specific named players (start X or Y, who's the better play, rank "
            "my RBs, X vs Y): call it once with every player being compared, and make your stated "
            "verdict its `recommended` player. It resolves each name against the real player list "
            "exactly like get_player_signals (an ambiguous or unknown name comes back in `unranked` "
            "with why, never guessed), scores each player on their current-season signals -- or, when "
            "a player has none, their stale prior-season fallback, marked stale: true on that entry -- "
            "and returns `verdict`: 'clear' (a single top player, in `recommended`), 'tied' (the top "
            "scores are indistinguishable -- `recommended` is null, `tied_at_top` lists them), or "
            "'insufficient_data' (fewer than two of the named players had any usable signal -- "
            "`recommended` is null). A verdict computed on stale entries is still the verdict (it's what "
            "the Feed shows too) -- say plainly that it rests on the named prior season's numbers, but do "
            "not replace it with your own sense of either player's track record. Every entry's "
            "signals_summary cites the actual numbers; use them to explain the verdict."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": "The players being compared, as named in the question (2 or more).",
                }
            },
            "required": ["player_names"],
        },
    },
    {
        "name": "get_team_record",
        "description": (
            "Structured lookup of a fantasy team's win/loss/tie record in this league (exact -- "
            "Sleeper computes this itself from completed matchups, this doesn't recompute it). "
            "Omit owner_display_name for the user's own team; give it to ask about another "
            "team's record."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "owner_display_name": {
                    "type": "string",
                    "description": "Optional -- omit for the user's own team.",
                }
            },
        },
    },
    {
        "name": "get_current_matchup",
        "description": (
            "Structured lookup of who a fantasy team plays this week (exact Sleeper data). "
            "Omit owner_display_name for the user's own team; give it to ask about another "
            "team's matchup. If this week's matchup data hasn't been ingested locally yet, "
            "returns a note saying so instead of guessing an opponent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "owner_display_name": {
                    "type": "string",
                    "description": "Optional -- omit for the user's own team.",
                }
            },
        },
    },
    {
        "name": "search_league_info",
        "description": (
            "Semantic search over this league's general information -- settings, matchup "
            "scores, transactions. Do NOT use this for a question about a specific named "
            "player's performance or matchup signals (use get_player_signals for that) -- "
            "semantic search over player text cannot reliably tell two same-surname players "
            "apart."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "submit_recommendation",
        "description": (
            "Conclude with your final recommendation. Call this exactly once, last, to end the "
            "conversation -- never answer in plain text instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "recommendation": {
                    "type": "string",
                    "description": "The concrete recommendation, e.g. which player to start.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "The explanation -- cite the specific signals/facts that drove this.",
                },
                "player_id": {
                    "type": "string",
                    "description": "nflverse player_id of the recommended player, if one specific player is being recommended.",
                },
                "data_gaps": {
                    "type": "array",
                    "description": (
                        "Structured record of anything you couldn't fully ground -- a named player "
                        "get_player_signals resolved but had zero signal data for (has_signals: false, not "
                        "just stale), or a part of the question no available tool/signal could answer at all "
                        "(e.g. trade strategy, which assets to package, or who to target -- no tool here "
                        "compares rosters or values assets across teams). Never fill an out-of-scope part "
                        "with plausible-sounding advice instead of recording the gap here: every specific "
                        "claim in recommendation/reasoning must be backed by an actual tool call you made "
                        "this turn, not general fantasy-football knowledge. Leave this empty only when the "
                        "whole answer is fully grounded -- never invent an entry, and never let one part of "
                        "a compound question you can't answer stop you from answering the part(s) you can."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "player_name": {
                                "type": "string",
                                "description": "The player this gap is about, if any -- omit for a question-level gap.",
                            },
                            "reason": {
                                "type": "string",
                                "enum": ["no_signal_data", "out_of_scope_capability"],
                                "description": (
                                    "no_signal_data: get_player_signals resolved the player's identity but "
                                    "returned has_signals: false (nothing computed for them at all -- distinct "
                                    "from stale, which means prior-season data exists). "
                                    "out_of_scope_capability: no tool/signal exists yet for this part of the "
                                    "question."
                                ),
                            },
                            "detail": {
                                "type": "string",
                                "description": "Plain-language explanation of the specific gap.",
                            },
                        },
                        "required": ["reason", "detail"],
                    },
                },
            },
            "required": ["recommendation", "reasoning"],
        },
    },
]


@dataclass
class RecommendContext:
    raw_dir: Path
    persist_dir: Path
    season: int
    as_of_week: int
    player_idx: Any  # polars.DataFrame, from player_index.build_player_index()
    # The verified league this context answers for (Phase 5.1) -- always
    # set by recommend()/generate_report(); optional only so tests that
    # exercise one tool handler directly can still build a context by
    # hand. The tool handlers themselves read Sleeper data via raw_dir
    # and never need scoring settings (see league.py's docstring for
    # why: nothing in the tools is scoring-dependent).
    league: LeagueConfig | None = None
    # Which roster in that league is "mine" (Phase 5.2) -- passed down to
    # src/rag/lookup.py's my_* functions. None means "fall back to the
    # MY_ROSTER_ID environment variable" (the single-league CLI
    # convention), so a CLI user's behavior is unchanged; the API server
    # always sets it from the session, never from its own environment.
    roster_id: str | None = None
    # Where the computed signals tables live (the parquet files
    # src/signals/matchup_signals.py writes) -- what the rank_players tool
    # ranks from, loaded lazily once per context on first use so a chat
    # turn that never compares players never reads them. Same default as
    # generate_report()'s signals_dir, so Chat and the Feed rank from the
    # same files by construction.
    signals_dir: Path = SIGNALS_DIR
    _signal_tables: ranking.SignalTables | None = field(default=None, repr=False)

    def signal_tables(self) -> ranking.SignalTables:
        if self._signal_tables is None:
            self._signal_tables = ranking.SignalTables.load(self.signals_dir, self.season, self.as_of_week)
        return self._signal_tables


@dataclass
class RecommendResult:
    recommendation: str
    reasoning: str | None
    player_id: str | None
    tool_calls: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    error: str | None = None
    # Which league this answer was grounded in (Phase 5.1) -- echoed so a
    # multi-league caller (Phase 5.2's API) can never mistake one
    # league's answer for another's.
    league_id: str | None = None
    # Which roster the "my"-flavored tools answered for (Phase 5.2) --
    # the explicit roster_id if one was passed, else None (meaning the
    # MY_ROSTER_ID environment fallback applied, or no tool needed it).
    roster_id: str | None = None
    # Structured version of any gap named in `reasoning`'s prose (a
    # no-signal-data player, or a part of the question no tool could
    # answer) -- see submit_recommendation's tool schema. Always a list,
    # never omitted/None: [] means "the answer was fully grounded," a
    # deliberate, present-but-empty shape (matching tool_calls/messages
    # already always being lists) so a caller can check `if
    # result["data_gaps"]:` without a None-guard, and a future Phase 5 UI
    # can render it directly without special-casing "field missing."
    data_gaps: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "recommendation": self.recommendation,
            "reasoning": self.reasoning,
            "player_id": self.player_id,
            "data_gaps": self.data_gaps,
            "tool_calls": self.tool_calls,
            "messages": self.messages,
            "error": self.error,
            "league_id": self.league_id,
            "roster_id": self.roster_id,
        }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())["data"]


def _infer_season_and_week(raw_dir: Path) -> tuple[int, int]:
    """Fall back to Sleeper's own current-week state (the same source
    src/ingest/sleeper.py's run() uses) when season/as_of_week aren't
    given explicitly."""
    state_path = raw_dir / "nfl_state.json"
    if not state_path.exists():
        raise RuntimeError(
            "season/as_of_week weren't given and "
            f"{state_path} doesn't exist -- pass them explicitly, or run "
            "`python -m src.ingest.sleeper` first."
        )
    state = _load_json(state_path)
    season = int(state["season"])
    week = int(state.get("display_week") or state.get("week") or 1)
    return season, week


def _build_system_prompt(league: dict, scoring_settings: dict, season: int, as_of_week: int) -> str:
    scoring_summary = ", ".join(f"{k}={v}" for k, v in sorted(scoring_settings.items()) if v) or "not available"
    return (
        f'You are Ask Madden, a fantasy football assistant for the Sleeper league "{league.get("name")}" '
        f"({season} season, week {as_of_week}). This league's scoring settings: {scoring_summary}.\n\n"
        "Give recommendations grounded in the specific facts and computed matchup signals you retrieve "
        "via tools -- never a bare opinion, and never a guess about which real player a name refers to. "
        "For any question naming a specific player, call get_player_signals to resolve their identity and "
        "pull their signals before saying anything substantive about them. If get_player_signals reports "
        "the name as ambiguous, name the specific candidates in your reasoning and say which one you need "
        "clarified rather than picking one yourself. Use get_my_roster / find_owner for roster-ownership "
        "questions, get_league_rosters for roster-COMPOSITION questions across teams (which teams are "
        "deep/shallow at a position -- never trade value, see below), get_team_record / get_current_matchup "
        "for standings/schedule questions (all exact Sleeper data), and search_league_info only for general "
        "league questions that aren't about one named player's performance or one team's record/matchup.\n\n"
        "If get_player_signals returns stale: true, the current season has no computed signals yet (e.g. "
        "it hasn't started) and you're seeing a prior-season reference instead -- say so explicitly and "
        "plainly in your reasoning (name which season the numbers are actually from) rather than presenting "
        "them as this season's performance.\n\n"
        "If get_player_signals returns has_signals: false, nothing at all has been computed for that player "
        "(not even a stale prior-season fallback) -- do not fill that gap with your own general knowledge "
        "about them (e.g. 'a promising rookie' or 'unproven'). Say explicitly in your reasoning that no "
        "computed signal data exists for them, and record a data_gaps entry (reason: no_signal_data, "
        "player_name set) in submit_recommendation instead of reasoning about them generically.\n\n"
        "A question can have an answerable part and a part nothing here can answer yet -- e.g. 'what's my "
        "weakest position, and who should I trade for to fix it?' is answerable for the weakest-position "
        "half (get_my_roster + get_player_signals) AND, now, partially answerable for the trade-partner "
        "half: get_league_rosters can tell you which teams have surplus/depth at that position (real "
        "roster composition, e.g. 'Team Y rosters 4 RBs to your weak TE need' is a fact you can state if "
        "get_league_rosters actually shows it). What it still can't tell you is whether a trade is fair, "
        "what either side's players are worth, or what to actually offer -- this project doesn't compute "
        "season-long player value or trade valuation yet. Never let the still-unanswerable part (valuation) "
        "cause you to give up on the whole question, and never let the newly-answerable part (composition) "
        "go unused just because valuation is missing: answer every part you can ground -- including "
        "composition, now that get_league_rosters exists -- then record a data_gaps entry (reason: "
        "out_of_scope_capability) specifically for the valuation/fairness piece, rather than submitting 'I "
        "don't have enough information' or silently dropping the composition half.\n\n"
        "Critical, and stricter than just 'record a gap': never paper over the still-missing valuation "
        "piece with plausible-sounding advice instead of admitting the gap. Every specific claim in "
        "recommendation/reasoning -- which team has surplus at a position, a player to target from another "
        "team, anything about another team's roster composition -- must be backed by an actual "
        "get_league_rosters (or find_owner/get_player_signals) call you made THIS turn, never by your own "
        "general fantasy-football knowledge. get_league_rosters tells you WHAT a team has, not what it's "
        "worth or what a fair trade looks like -- there is still no tool for trade value, fairness, or 'what "
        "should I offer'. If a question calls for that, you have no tool that can ground an answer: do not "
        "improvise one, and do not dress up a composition fact (e.g. 'they have extra RBs') as if it proves "
        "a trade would be fair or good. Say explicitly that you can identify composition (and do so, "
        "concretely, when get_league_rosters supports it) but can't recommend actual trade value or "
        "strategy, and record that specific gap as a data_gaps entry (reason: out_of_scope_capability) -- "
        "never as trade advice.\n\n"
        "Head-to-head and start/sit comparisons between specific named players ('start X or Y?', 'who's the "
        "better play, X or Y', 'rank my WRs', 'X vs Y this week') have exactly one verdict here: rank_players' "
        "output. For any such question, call rank_players once with every player being compared (it resolves "
        "names itself, so you don't also need get_player_signals for identity -- call get_player_signals "
        "too only if you want extra detail beyond the numbers rank_players already cites), then make your "
        "recommendation and player_id its `recommended` player, and explain that verdict using the "
        "signals_summary numbers it returned. rank_players is the same deterministic ranking the Feed's "
        "start/sit report shows, so your verdict and the Feed's can't disagree if you use it. What you must "
        "never do: reach a start/sit verdict from your own general knowledge of either player -- reputation, "
        "'consistently elite performance history', last year's fantasy finish, or anything else you didn't "
        "retrieve this turn. That applies with full force when rank_players' entries are stale: true -- a "
        "verdict on last season's numbers is still the verdict (the Feed shows the same one); say plainly "
        "which season it rests on, but do not override it with your own read of the players. If rank_players "
        "returns verdict 'tied', say the ranking can't separate them and name the tied players -- do not pick "
        "one. If it returns 'insufficient_data', say the comparison can't be grounded (name who lacked "
        "data, with a data_gaps entry, reason no_signal_data, for each player it listed as unranked with no "
        "signals) -- do not pick the one who happened to have data, and do not pick from general knowledge. "
        "If a name came back unranked as ambiguous, ask which player was meant, exactly as for "
        "get_player_signals.\n\n"
        "Always end by calling submit_recommendation exactly once with a concrete recommendation and the "
        "reasoning that led to it, citing the specific signals you retrieved -- this league's scoring "
        "settings above should inform which stats matter (e.g. reception volume matters more here if "
        "rec > 0). Include a data_gaps entry for every no_signal_data player and every out_of_scope_capability "
        "gap per above -- leave data_gaps empty only when the full answer is genuinely grounded. If the tools "
        "genuinely don't have what's needed to answer any part of the question, say so plainly in "
        "submit_recommendation rather than guessing or repeating the same tool call."
    )


def _tool_get_my_roster(tool_input: dict, ctx: RecommendContext) -> dict:
    position = tool_input.get("position")
    try:
        players = (
            lookup.my_players_by_position(position, ctx.raw_dir, roster_id=ctx.roster_id)
            if position
            else lookup.my_players(ctx.raw_dir, roster_id=ctx.roster_id)
        )
    except RuntimeError as e:
        return {"error": str(e)}
    return {
        "players": [
            {
                "player_id": p.get("player_id"),
                "name": p.get("full_name"),
                "position": p.get("position"),
                "team": p.get("team"),
            }
            for p in players
        ]
    }


def _tool_find_owner(tool_input: dict, ctx: RecommendContext) -> dict:
    team = lookup.find_team_by_player(tool_input["player_name"], ctx.raw_dir)
    if team is None:
        return {"owner": None, "note": "No team in this league rosters a player matching that name."}
    return {
        "owner_team_name": team.get("team_name"),
        "owner_display_name": team.get("display_name"),
        "roster_id": team.get("roster_id"),
    }


def _tool_get_league_rosters(tool_input: dict, ctx: RecommendContext) -> dict:
    owner = tool_input.get("owner_display_name")
    if owner:
        roster = lookup.team_roster_for_owner(owner, ctx.raw_dir)
        if roster is None:
            return {"error": f"No team found for owner {owner!r} in this league."}
        return {"teams": [roster]}
    return {"teams": lookup.all_team_rosters(ctx.raw_dir)}


def _tool_get_player_signals(tool_input: dict, ctx: RecommendContext) -> dict:
    result = player_index.resolve_player(tool_input["player_name"], ctx.player_idx)

    if result.match_type == "none":
        return {"resolved": False, "note": f"No current NFL player matches {tool_input['player_name']!r}."}

    if result.match_type == "ambiguous":
        return {
            "resolved": False,
            "ambiguous": True,
            "candidates": [
                {"player_id": c.player_id, "name": c.player_name, "position": c.position, "team": c.team}
                for c in result.candidates
            ],
            "note": (
                "More than one current player matches this name -- ask which one was meant "
                "(or use team/position to disambiguate) instead of guessing."
            ),
        }

    match = result.candidates[0]
    chunk, stale = retrieve.query_player_signal_with_fallback(
        match.player_id, season=ctx.season, as_of_week=ctx.as_of_week, persist_dir=ctx.persist_dir
    )
    base = {
        "resolved": True,
        "player_id": match.player_id,
        "player_name": match.player_name,
        "position": match.position,
        "team": match.team,
    }
    if chunk is None:
        # Identity resolved but genuinely nothing computed for them, ever
        # (not even a stale prior-season fallback -- e.g. a rookie with no
        # snaps yet). has_signals: false is the explicit, structured
        # signal the system prompt tells the model to turn into a
        # data_gaps entry (reason: no_signal_data) instead of silently
        # reasoning about this player from its own background knowledge.
        base["has_signals"] = False
        base["note"] = "Resolved the player but no computed signals are embedded yet for this season/week."
        return base
    base["has_signals"] = True
    base["stale"] = stale
    if stale:
        source_season = chunk["metadata"].get("season")
        source_week = chunk["metadata"].get("week")
        base["source_season"] = source_season
        base["source_as_of_week"] = source_week
        base["signals"] = (
            f"[STALE -- no {ctx.season} signal data yet; showing {source_season} season-end reference "
            f"instead, do not present this as current] {chunk['text']}"
        )
    else:
        base["signals"] = chunk["text"]
    return base


def _tool_rank_players(tool_input: dict, ctx: RecommendContext) -> dict:
    """Resolve each named player exactly as get_player_signals does
    (structured exact-then-fuzzy name resolution, never a guess), attach
    the same signal row generate_report() would use for them
    (current-season, else stale prior-season fallback), and rank them
    with ranking.rank_candidates() -- the identical call the Feed's
    start_sit report makes. The one thing this adds on top of that shared
    result is `same_position`: the start_sit report only ever compares
    players within one roster position, and the score isn't
    position-normalized, so a cross-position comparison (an RB vs. a WR
    for a flex spot) is flagged as such rather than silently treated as
    equivalent to the Feed's within-position verdict."""
    names = tool_input.get("player_names") or []
    if len(names) < 2:
        return {"error": "rank_players needs at least two player names to compare."}

    tables = ctx.signal_tables()
    candidates: list[dict] = []
    unresolved: list[dict] = []
    for name in names:
        result = player_index.resolve_player(name, ctx.player_idx)
        if result.match_type == "none":
            unresolved.append({"name": name, "reason": "unresolved", "note": f"No current NFL player matches {name!r}."})
            continue
        if result.match_type == "ambiguous":
            unresolved.append(
                {
                    "name": name,
                    "reason": "ambiguous",
                    "candidates": [
                        {"player_id": c.player_id, "name": c.player_name, "position": c.position, "team": c.team}
                        for c in result.candidates
                    ],
                    "note": "More than one current player matches this name -- ask which one was meant instead of guessing.",
                }
            )
            continue
        match = result.candidates[0]
        candidates.append(
            {
                "player_id": match.player_id,
                "name": match.player_name,
                "position": match.position,
                "team": match.team,
                "row": tables.row_for(match.player_id),
            }
        )

    ranked = ranking.rank_candidates(candidates)
    # A resolved player with no usable signal at all is the has_signals:
    # false case -- the same reason code the system prompt maps to a
    # data_gaps entry, so the model can't mistake "unranked" for "ranked
    # last".
    unranked = [{**entry, "reason": "no_signal_data", "has_signals": False} for entry in ranked["unranked"]]
    positions = {c["position"] for c in candidates if c.get("position")}
    return {
        "season": ctx.season,
        "as_of_week": ctx.as_of_week,
        "verdict": ranked["verdict"],
        "recommended": ranked["recommended"],
        "ranked": ranked["ranked"],
        "tied_at_top": ranked["tied_at_top"],
        "unranked": unranked + unresolved,
        "same_position": len(positions) <= 1,
        "score_description": ranked["score_description"],
        "note": (
            "This is the same deterministic ranking the Feed's start/sit report shows for these players."
            if len(positions) <= 1
            else (
                "These players play different positions. The Feed's start/sit report only compares players "
                "within one position and this score is not position-normalized, so treat this as the same "
                "raw opportunity comparison the Feed's waiver report uses across positions, not as a "
                "within-position start/sit verdict."
            )
        ),
    }


def _tool_get_team_record(tool_input: dict, ctx: RecommendContext) -> dict:
    owner = tool_input.get("owner_display_name")
    try:
        record = (
            lookup.team_record_for_owner(owner, ctx.raw_dir)
            if owner
            else lookup.my_team_record(ctx.raw_dir, roster_id=ctx.roster_id)
        )
    except RuntimeError as e:
        return {"error": str(e)}
    if record is None:
        return {"error": f"No team found for owner {owner!r} in this league."}
    return record


def _tool_get_current_matchup(tool_input: dict, ctx: RecommendContext) -> dict:
    owner = tool_input.get("owner_display_name")
    try:
        matchup = (
            lookup.current_matchup_for_owner(owner, ctx.as_of_week, ctx.raw_dir)
            if owner
            else lookup.my_current_matchup(ctx.as_of_week, ctx.raw_dir, roster_id=ctx.roster_id)
        )
    except RuntimeError as e:
        return {"error": str(e)}
    if matchup is None:
        return {
            "note": (
                f"No matchup data for week {ctx.as_of_week} has been ingested locally yet -- "
                f"run `python -m src.ingest.sleeper --week {ctx.as_of_week}` first."
            )
        }
    return matchup


def _tool_search_league_info(tool_input: dict, ctx: RecommendContext) -> dict:
    results = retrieve.query(tool_input["query"], n_results=5, persist_dir=ctx.persist_dir)
    return {"results": [{"text": r["text"], "type": r["metadata"].get("type")} for r in results]}


_DISPATCH = {
    "get_my_roster": _tool_get_my_roster,
    "find_owner": _tool_find_owner,
    "get_league_rosters": _tool_get_league_rosters,
    "get_player_signals": _tool_get_player_signals,
    "rank_players": _tool_rank_players,
    "get_team_record": _tool_get_team_record,
    "get_current_matchup": _tool_get_current_matchup,
    "search_league_info": _tool_search_league_info,
}


def dispatch_tool(name: str, tool_input: dict, ctx: RecommendContext) -> dict:
    """Run one tool call and return its structured result. submit_recommendation
    is handled by the orchestration loop directly, never here -- it ends
    the conversation rather than producing a tool_result to feed back."""
    handler = _DISPATCH.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}
    return handler(tool_input, ctx)


def _verdict_contradiction(submit_input: dict, tool_calls: list[dict]) -> str | None:
    """The code-checkable half of the "ground your verdict in rank_players"
    rule: given what the model is about to submit and every tool call
    made so far in this recommend() call, return a message describing the
    contradiction if the submitted player_id is one of the players a
    rank_players call compared but is NOT what that ranking supports --
    a different player than its `recommended`, or any pick at all when
    its verdict was 'tied' or 'insufficient_data'. None when there's
    nothing to object to (no rank_players call, no player_id, or a
    player_id outside the compared set -- e.g. a compound question whose
    recommendation is about someone else entirely). This can't catch a
    prose-only contradiction (a recommendation string naming the wrong
    player with player_id unset) -- that part stays with the system
    prompt and real-model validation, same as Phase 3.7."""
    player_id = submit_input.get("player_id")
    if not player_id:
        return None
    for call in tool_calls:
        if call.get("name") != "rank_players":
            continue
        result = call.get("result") or {}
        if "verdict" not in result:
            continue
        compared = {e.get("player_id") for e in result.get("ranked", []) + result.get("unranked", [])}
        if player_id not in compared:
            continue
        verdict = result["verdict"]
        recommended = result.get("recommended") or {}
        if verdict == "clear" and recommended.get("player_id") != player_id:
            return (
                f"Your player_id ({player_id}) contradicts rank_players, which ranked "
                f"{recommended.get('name')} ({recommended.get('player_id')}) first among the players you "
                "compared. The start/sit verdict must be rank_players' recommended player -- resubmit with "
                "that player (you may explain the ranking's numbers, but not override it)."
            )
        if verdict == "tied":
            names = ", ".join(e.get("name") or "?" for e in result.get("tied_at_top", []))
            return (
                f"Your player_id ({player_id}) picks one of the players rank_players reported as tied "
                f"({names}). Resubmit without a player_id, saying the ranking cannot separate them."
            )
        if verdict == "insufficient_data":
            return (
                f"Your player_id ({player_id}) picks one of the compared players even though rank_players "
                "reported insufficient_data (fewer than two of them had any usable signal). Resubmit without "
                "a player_id, saying the comparison can't be grounded and recording a data_gaps entry for "
                "each player with no signal data."
            )
    return None


def recommend(
    question: str,
    league_id: str,
    messages: list[dict] | None = None,
    raw_dir: Path = RAW_DIR,
    persist_dir: Path = CHROMA_DIR,
    season: int | None = None,
    as_of_week: int | None = None,
    client: "anthropic.Anthropic | None" = None,
    model: str = DEFAULT_MODEL,
    max_turns: int = MAX_TOOL_TURNS,
    roster_id: str | None = None,
    signals_dir: Path = SIGNALS_DIR,
) -> dict:
    """Answer question using retrieved facts + computed signals via a
    Claude tool-use agent, returning
    {"recommendation", "reasoning", "player_id", "data_gaps", "tool_calls",
    "messages", "error", "league_id", "roster_id"}.

    roster_id: which roster in league_id is "mine" for the my-flavored
    tools (Phase 5.2). None falls back to the MY_ROSTER_ID environment
    variable inside src/rag/lookup.py -- the single-league CLI/.env
    convention, unchanged. The API server always passes the session's
    roster_id explicitly and never relies on the fallback.

    league_id: the Sleeper league to answer for -- required (Phase 5.1).
    Resolved via src/reasoning/league.py's load_league(): the league's
    real scoring_settings come from the ingested league.json in raw_dir,
    and a raw_dir that holds a *different* league raises
    LeagueMismatchError before any model call rather than silently
    answering from the wrong roster/scoring. raw_dir/persist_dir remain
    the on-disk location of that league's data (per-league storage
    layout is Phase 5.2's job, not this parameter's).

    messages: prior conversation history (as previously returned in a
    result's "messages"), for a multi-turn conversation -- e.g. the
    model asks a clarifying question, the caller answers it in the next
    recommend() call. Leave this None (the default) for a single
    independent question; this is what evals/run_eval.py and
    evals/run_decision_eval.py must do, since carrying state across
    supposedly-independent eval questions would leak context between
    them -- neither passes messages, so neither is affected by this
    parameter existing.

    signals_dir: where the computed signals parquet tables live -- what
    the rank_players tool ranks from. Defaults to the same directory
    generate_report() reads, so a Chat verdict and a Feed verdict are
    computed from the same files.

    client defaults to a real anthropic.Anthropic() (reads
    ANTHROPIC_API_KEY from the environment) but can be injected -- tests
    pass a fake client so the orchestration loop is verified without
    hitting the network.

    Never raises on a question the tools couldn't resolve (e.g. no
    matching data, or the model can't converge within max_turns) --
    returns a result explaining that instead, with error set to a short
    code identifying why, so a caller (the CLI, an eval harness) always
    gets a well-formed response to work with rather than an unhandled
    exception.

    Loads .env itself (see the comment just below) so this works whether
    called via the CLI or imported directly -- e.g.
    `from src.reasoning.recommend import recommend`, the same way Phase
    5's API layer will call it.
    """
    # load_dotenv() only sets a variable if it isn't already in the
    # environment, so this is safe/correct in production too: real env
    # vars (Phase 5.2's deployment target) always win, and this is a
    # cheap no-op if no .env file exists (the normal production case).
    # This used to only happen in main() (this module's CLI entry
    # point), which meant recommend() silently depended on being called
    # via `python -m src.reasoning.recommend` -- calling it directly
    # (e.g. `from src.reasoning.recommend import recommend`, the same
    # way Phase 5's API layer will) skipped it entirely and failed with
    # an Anthropic auth error instead of a clear "ANTHROPIC_API_KEY not
    # set" message, since ANTHROPIC_API_KEY was never loaded from .env.
    # Phase 5.2: once per process, not once per call -- see load_dotenv_once().
    load_dotenv_once()

    # Phase 5.1: the league is an explicit, verified input. This is the
    # single read of the league's real Sleeper scoring_settings -- they
    # go into the system prompt below, verbatim, exactly as Phase 3
    # already did; the only difference is that the league they belong
    # to is now named by the caller and checked against what's on disk.
    league = load_league(league_id, raw_dir=raw_dir, persist_dir=persist_dir)
    scoring_settings = league.scoring_settings

    if season is None or as_of_week is None:
        inferred_season, inferred_week = _infer_season_and_week(raw_dir)
        season = season if season is not None else inferred_season
        as_of_week = as_of_week if as_of_week is not None else inferred_week

    ctx = RecommendContext(
        raw_dir=raw_dir,
        persist_dir=persist_dir,
        season=season,
        as_of_week=as_of_week,
        player_idx=player_index.build_player_index(season),
        league=league,
        roster_id=str(roster_id) if roster_id is not None else None,
        signals_dir=signals_dir,
    )

    client = client or anthropic.Anthropic()
    system_prompt = _build_system_prompt({"name": league.name}, scoring_settings, season, as_of_week)
    messages = list(messages) if messages else []
    messages.append({"role": "user", "content": question})
    tool_calls: list[dict] = []

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if not tool_uses:
            # The model answered without calling submit_recommendation --
            # e.g. a clarifying question. Fall back to whatever text it
            # gave rather than looping forever waiting for a tool call
            # that will never come; messages is still valid to continue
            # from (the last turn is a plain assistant text turn, no
            # pending tool_use to resolve).
            text = "".join(block.text for block in response.content if block.type == "text")
            return RecommendResult(
                text, None, None, tool_calls, messages, league_id=league.league_id, roster_id=ctx.roster_id
            ).to_dict()

        submit = next((b for b in tool_uses if b.name == "submit_recommendation"), None)
        if submit is not None:
            # Resolve every tool_use in this turn -- including
            # submit_recommendation itself, with a synthetic result --
            # so `messages` stays API-valid if the caller continues the
            # conversation with another question afterward (the API
            # rejects a new user turn while a prior tool_use is
            # unresolved).
            #
            # Any other tools in the same turn run first, so a
            # rank_players call made alongside the submit still counts
            # when checking the submitted verdict against it below.
            tool_results = []
            for block in tool_uses:
                if block is submit:
                    continue
                result = dispatch_tool(block.name, block.input, ctx)
                tool_calls.append({"name": block.name, "input": block.input, "result": result})
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})

            contradiction = _verdict_contradiction(submit.input, tool_calls)
            if contradiction is not None:
                # The submitted verdict contradicts the deterministic
                # ranking the model itself retrieved. Don't pass it
                # through: hand the contradiction back as the
                # submit_recommendation tool's result (is_error) and let
                # the model resubmit. Recorded in tool_calls so a caller
                # can see it happened. Bounded by max_turns like every
                # other turn -- a model that keeps contradicting the
                # ranking ends in the graceful max_turns_exceeded result,
                # never in a wrong verdict presented as grounded.
                tool_calls.append(
                    {"name": "submit_recommendation", "input": submit.input, "result": {"error": contradiction}}
                )
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": submit.id, "content": contradiction, "is_error": True}
                )
                messages.append({"role": "user", "content": tool_results})
                continue

            tool_results.append(
                {"type": "tool_result", "tool_use_id": submit.id, "content": "Recommendation recorded."}
            )
            messages.append({"role": "user", "content": tool_results})

            return RecommendResult(
                recommendation=submit.input["recommendation"],
                reasoning=submit.input.get("reasoning"),
                player_id=submit.input.get("player_id"),
                data_gaps=submit.input.get("data_gaps") or [],
                tool_calls=tool_calls,
                messages=messages,
                league_id=league.league_id,
                roster_id=ctx.roster_id,
            ).to_dict()

        tool_results = []
        for block in tool_uses:
            result = dispatch_tool(block.name, block.input, ctx)
            tool_calls.append({"name": block.name, "input": block.input, "result": result})
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
        messages.append({"role": "user", "content": tool_results})

    return RecommendResult(
        recommendation="I don't have enough information to answer that.",
        reasoning=(
            f"Tried {max_turns} tool-use turns without reaching a final answer -- the "
            "available tools didn't have enough data to resolve this question."
        ),
        player_id=None,
        tool_calls=tool_calls,
        messages=messages,
        error="max_turns_exceeded",
        league_id=league.league_id,
        roster_id=ctx.roster_id,
    ).to_dict()


def _print_result(result: dict) -> None:
    print(result["recommendation"])
    if result["reasoning"]:
        print(f"\nReasoning: {result['reasoning']}")
    if result.get("data_gaps"):
        print("\nData gaps:")
        for gap in result["data_gaps"]:
            who = f" ({gap['player_name']})" if gap.get("player_name") else ""
            print(f"  - [{gap.get('reason', 'unknown')}]{who} {gap.get('detail', '')}")


def _run_repl(league_id: str, season: int | None, as_of_week: int | None, roster_id: str | None = None) -> None:
    """Multi-turn REPL: each turn's response feeds the next call's
    `messages`, so a clarifying question the agent asks can be answered
    in the same conversation instead of starting over."""
    print("Ask Madden -- interactive mode (type 'exit' to quit)")
    messages: list[dict] | None = None
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        result = recommend(
            question, league_id, messages=messages, season=season, as_of_week=as_of_week, roster_id=roster_id
        )
        messages = result["messages"]
        _print_result(result)
        print()


def main() -> None:
    # recommend()/generate_report() load .env themselves (so a direct
    # import gets the same behavior as this CLI), but the CLI's own
    # --league-id default below reads SLEEPER_LEAGUE_ID at argparse time,
    # before either of them runs -- so load it here too. load_dotenv()
    # never overrides a real environment variable, so calling it in both
    # places is harmless (and, since Phase 5.2, a no-op the second time).
    load_dotenv_once()
    parser = argparse.ArgumentParser(description="Ask Madden: retrieval + signals -> Claude recommendation")
    parser.add_argument("question", nargs="?", help="a single question; omit this and pass --interactive instead")
    parser.add_argument(
        "--league-id",
        default=os.environ.get("SLEEPER_LEAGUE_ID"),
        help=(
            "the Sleeper league to answer for (Phase 5.1: required -- defaults to SLEEPER_LEAGUE_ID from "
            "the environment/.env, the same variable `python -m src.ingest.sleeper` uses)"
        ),
    )
    parser.add_argument(
        "--roster-id",
        default=None,
        help=(
            "which roster in that league is yours (Phase 5.2) -- defaults to MY_ROSTER_ID from the "
            "environment/.env, exactly as before"
        ),
    )
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--as-of-week", type=int, default=None)
    parser.add_argument(
        "--interactive", action="store_true", help="start a multi-turn REPL instead of asking one question"
    )
    parser.add_argument(
        "--report",
        choices=["start_sit", "drop", "waiver_pickups"],
        default=None,
        help=(
            "generate a structured report (start/sit, drop, or waiver-wire pickups) instead of "
            "answering a single question or starting a REPL -- see src/reasoning/report.py"
        ),
    )
    args = parser.parse_args()

    if not args.league_id:
        raise SystemExit(
            "a Sleeper league ID is required: pass --league-id <id>, or set SLEEPER_LEAGUE_ID in .env "
            "(see .env.example)"
        )

    if args.report:
        # Imported here, not at module level: report.py imports this
        # module (to reuse dispatch_tool/RecommendContext), so importing
        # it back at the top of this file would be a circular import.
        from src.reasoning.report import generate_report, _print_report

        _print_report(
            generate_report(
                args.report, args.league_id, season=args.season, as_of_week=args.as_of_week, roster_id=args.roster_id
            )
        )
        return

    if args.interactive:
        _run_repl(args.league_id, season=args.season, as_of_week=args.as_of_week, roster_id=args.roster_id)
        return

    if not args.question:
        raise SystemExit("a question is required unless --interactive or --report is given")

    result = recommend(
        args.question, args.league_id, season=args.season, as_of_week=args.as_of_week, roster_id=args.roster_id
    )
    _print_result(result)


if __name__ == "__main__":
    main()
