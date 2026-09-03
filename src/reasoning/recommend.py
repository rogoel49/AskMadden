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

**Scope note**: recommend() operates on whatever league is already
ingested into data/raw/sleeper/ (like lookup.py and embed.py already
do) -- it does not take a league_id and re-fetch a different league.
Multi-league parameterization is explicitly Phase 5's job
(src/api/, per PROJECT_SPEC.md); adding it here early would blur that
phase boundary for no benefit yet.

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

DEFAULT_MODEL = os.environ.get("ASKMADDEN_MODEL", "claude-sonnet-4-5-20250929")
MAX_TOOL_TURNS = 8

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
            "player is rostered in this league at all."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"player_name": {"type": "string"}},
            "required": ["player_name"],
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


@dataclass
class RecommendResult:
    recommendation: str
    reasoning: str | None
    player_id: str | None
    tool_calls: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "recommendation": self.recommendation,
            "reasoning": self.reasoning,
            "player_id": self.player_id,
            "tool_calls": self.tool_calls,
            "messages": self.messages,
            "error": self.error,
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
        "questions, get_team_record / get_current_matchup for standings/schedule questions (all exact "
        "Sleeper data), and search_league_info only for general league questions that aren't about one "
        "named player's performance or one team's record/matchup.\n\n"
        "Always end by calling submit_recommendation exactly once with a concrete recommendation and the "
        "reasoning that led to it, citing the specific signals you retrieved -- this league's scoring "
        "settings above should inform which stats matter (e.g. reception volume matters more here if "
        "rec > 0). If the tools genuinely don't have what's needed to answer (e.g. a question about data "
        "that hasn't been ingested), say so plainly in submit_recommendation rather than guessing or "
        "repeating the same tool call."
    )


def _tool_get_my_roster(tool_input: dict, ctx: RecommendContext) -> dict:
    position = tool_input.get("position")
    try:
        players = (
            lookup.my_players_by_position(position, ctx.raw_dir) if position else lookup.my_players(ctx.raw_dir)
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
    chunk = retrieve.query_player_signal(
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
        base["note"] = "Resolved the player but no computed signals are embedded yet for this season/week."
        return base
    base["signals"] = chunk["text"]
    return base


def _tool_get_team_record(tool_input: dict, ctx: RecommendContext) -> dict:
    owner = tool_input.get("owner_display_name")
    try:
        record = lookup.team_record_for_owner(owner, ctx.raw_dir) if owner else lookup.my_team_record(ctx.raw_dir)
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
            else lookup.my_current_matchup(ctx.as_of_week, ctx.raw_dir)
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
    "get_player_signals": _tool_get_player_signals,
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


def recommend(
    question: str,
    messages: list[dict] | None = None,
    raw_dir: Path = RAW_DIR,
    persist_dir: Path = CHROMA_DIR,
    season: int | None = None,
    as_of_week: int | None = None,
    client: "anthropic.Anthropic | None" = None,
    model: str = DEFAULT_MODEL,
    max_turns: int = MAX_TOOL_TURNS,
) -> dict:
    """Answer question using retrieved facts + computed signals via a
    Claude tool-use agent, returning
    {"recommendation", "reasoning", "player_id", "tool_calls", "messages", "error"}.

    messages: prior conversation history (as previously returned in a
    result's "messages"), for a multi-turn conversation -- e.g. the
    model asks a clarifying question, the caller answers it in the next
    recommend() call. Leave this None (the default) for a single
    independent question; this is what evals/run_eval.py and
    evals/run_decision_eval.py must do, since carrying state across
    supposedly-independent eval questions would leak context between
    them -- neither passes messages, so neither is affected by this
    parameter existing.

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
    """
    league_path = raw_dir / "league.json"
    if not league_path.exists():
        raise RuntimeError(f"{league_path} doesn't exist -- run `python -m src.ingest.sleeper` first.")
    league = _load_json(league_path)
    scoring_settings = league.get("scoring_settings", {})

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
    )

    client = client or anthropic.Anthropic()
    system_prompt = _build_system_prompt(league, scoring_settings, season, as_of_week)
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
            return RecommendResult(text, None, None, tool_calls, messages).to_dict()

        submit = next((b for b in tool_uses if b.name == "submit_recommendation"), None)
        if submit is not None:
            # Resolve every tool_use in this turn -- including
            # submit_recommendation itself, with a synthetic result --
            # so `messages` stays API-valid if the caller continues the
            # conversation with another question afterward (the API
            # rejects a new user turn while a prior tool_use is
            # unresolved).
            tool_results = []
            for block in tool_uses:
                if block is submit:
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": "Recommendation recorded."}
                    )
                    continue
                result = dispatch_tool(block.name, block.input, ctx)
                tool_calls.append({"name": block.name, "input": block.input, "result": result})
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
            messages.append({"role": "user", "content": tool_results})

            return RecommendResult(
                recommendation=submit.input["recommendation"],
                reasoning=submit.input.get("reasoning"),
                player_id=submit.input.get("player_id"),
                tool_calls=tool_calls,
                messages=messages,
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
    ).to_dict()


def _print_result(result: dict) -> None:
    print(result["recommendation"])
    if result["reasoning"]:
        print(f"\nReasoning: {result['reasoning']}")


def _run_repl(season: int | None, as_of_week: int | None) -> None:
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
        result = recommend(question, messages=messages, season=season, as_of_week=as_of_week)
        messages = result["messages"]
        _print_result(result)
        print()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ask Madden: retrieval + signals -> Claude recommendation")
    parser.add_argument("question", nargs="?", help="a single question; omit this and pass --interactive instead")
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

    if args.report:
        # Imported here, not at module level: report.py imports this
        # module (to reuse dispatch_tool/RecommendContext), so importing
        # it back at the top of this file would be a circular import.
        from src.reasoning.report import generate_report, _print_report

        _print_report(generate_report(args.report, season=args.season, as_of_week=args.as_of_week))
        return

    if args.interactive:
        _run_repl(season=args.season, as_of_week=args.as_of_week)
        return

    if not args.question:
        raise SystemExit("a question is required unless --interactive or --report is given")

    result = recommend(args.question, season=args.season, as_of_week=args.as_of_week)
    _print_result(result)


if __name__ == "__main__":
    main()
