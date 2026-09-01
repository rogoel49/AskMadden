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

    def to_dict(self) -> dict:
        return {
            "recommendation": self.recommendation,
            "reasoning": self.reasoning,
            "player_id": self.player_id,
            "tool_calls": self.tool_calls,
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
        "questions (exact Sleeper data) and search_league_info only for general league questions that "
        "aren't about one named player's performance.\n\n"
        "Always end by calling submit_recommendation exactly once with a concrete recommendation and the "
        "reasoning that led to it, citing the specific signals you retrieved -- this league's scoring "
        "settings above should inform which stats matter (e.g. reception volume matters more here if "
        "rec > 0)."
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


def _tool_search_league_info(tool_input: dict, ctx: RecommendContext) -> dict:
    results = retrieve.query(tool_input["query"], n_results=5, persist_dir=ctx.persist_dir)
    return {"results": [{"text": r["text"], "type": r["metadata"].get("type")} for r in results]}


_DISPATCH = {
    "get_my_roster": _tool_get_my_roster,
    "find_owner": _tool_find_owner,
    "get_player_signals": _tool_get_player_signals,
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
    {"recommendation", "reasoning", "player_id", "tool_calls"}.

    client defaults to a real anthropic.Anthropic() (reads
    ANTHROPIC_API_KEY from the environment) but can be injected -- tests
    pass a fake client so the orchestration loop is verified without
    hitting the network.
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
    messages: list[dict] = [{"role": "user", "content": question}]
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
            # The model answered without calling submit_recommendation.
            # Fall back to whatever text it gave rather than looping
            # forever waiting for a tool call that will never come.
            text = "".join(block.text for block in response.content if block.type == "text")
            return RecommendResult(text, None, None, tool_calls).to_dict()

        submit = next((b for b in tool_uses if b.name == "submit_recommendation"), None)
        if submit is not None:
            return RecommendResult(
                recommendation=submit.input["recommendation"],
                reasoning=submit.input.get("reasoning"),
                player_id=submit.input.get("player_id"),
                tool_calls=tool_calls,
            ).to_dict()

        tool_results = []
        for block in tool_uses:
            result = dispatch_tool(block.name, block.input, ctx)
            tool_calls.append({"name": block.name, "input": block.input, "result": result})
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"recommend() exceeded max_turns={max_turns} without reaching submit_recommendation")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ask Madden: retrieval + signals -> Claude recommendation")
    parser.add_argument("question")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--as-of-week", type=int, default=None)
    args = parser.parse_args()

    result = recommend(args.question, season=args.season, as_of_week=args.as_of_week)
    print(result["recommendation"])
    if result["reasoning"]:
        print(f"\nReasoning: {result['reasoning']}")


if __name__ == "__main__":
    main()
