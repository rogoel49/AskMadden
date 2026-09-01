"""Player identity resolution: maps a natural-language player name
mention to a specific nflverse gsis_id via structured (exact/fuzzy)
lookup against nflverse's own player reference table -- never via
embedding proximity.

Why this lives here, not in src/reasoning/: it's a league-agnostic fix.
"Which real NFL player does this name refer to" has nothing to do with
any one fantasy league's roster or scoring settings (that per-league
join still happens in src/reasoning/recommend.py, per CLAUDE.md's key
architectural principle) -- it's the same class of problem cli.py
already solved for "my"-flavored identity questions (routed to
structured lookup instead of semantic search), generalized here to any
named-player mention and made available to every src/rag/retrieve.py
caller, not just recommend.py.

Bug this fixes: retrieve.py's query() is pure semantic embedding search
over chunk sentences. Two real NFL players who share a surname (e.g.
Christian McCaffrey vs. his brother Luke McCaffrey; the several dozen
active NFL players surnamed Williams) can outrank each other in
embedding space, so asking about one by name can silently return
signals for the wrong person, or fail to surface the intended player's
chunk at all. Exact name matching against the real player list, with
fuzzy matches returned as explicit candidates (never silently guessed)
when there's no exact match, closes that gap.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass

import polars as pl

from src.ingest import nflverse

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
# Tuned against the real 2024 player list during development: 0.75 let
# "McCaffrey" fuzzy-match an unrelated "Nate McCrary" (token ratio
# exactly 0.75) alongside the two real McCaffreys. 0.8 excludes that
# false positive while still catching real typos (e.g. "Willaims" ->
# "Williams" is 0.875). This is a simple heuristic, not a tuned model --
# revisit if real usage turns up more false positives or false negatives.
FUZZY_CUTOFF = 0.8


@dataclass
class PlayerMatch:
    player_id: str
    player_name: str
    position: str
    team: str | None


@dataclass
class ResolveResult:
    """match_type is "exact" (exactly one plausible player), "ambiguous"
    (more than one real, recent player matches -- either they share the
    literal full name, or several distinct players are similarly close
    fuzzy matches), or "none" (no plausible match at all). Only "exact"
    should ever be used without confirming with whoever asked --
    "ambiguous" must be surfaced as a question, never silently resolved
    to whichever candidate sorts first."""

    match_type: str
    candidates: list[PlayerMatch]


def build_player_index(season: int, players: pl.DataFrame | None = None) -> pl.DataFrame:
    """The name-resolution reference table: skill-position players
    (QB/RB/WR/TE -- the fantasy-relevant positions, matching
    evals/build_ground_truth.py's scope) whose nflverse gsis_id is in the
    modern "00-XXXXXXX" format (the ID space pbp/matchup_signals.py
    uses) and who were active in `season` or the season before it.

    That recency filter matters: nflverse's player database goes back to
    1999, and its "status" field doesn't reliably mean "currently
    rostered" (e.g. Ed McCaffrey, retired since 2003, is still recorded
    as "ACT") -- without filtering by last_season, the index would carry
    decades of retired namesakes as false ambiguity for every common
    surname.
    """
    if players is None:
        players = nflverse.fetch_players()
    return players.filter(
        pl.col("gsis_id").str.starts_with("00-")
        & pl.col("position").is_in(SKILL_POSITIONS)
        & pl.col("last_season").is_not_null()
        & (pl.col("last_season") >= season - 1)
    ).select(
        pl.col("gsis_id").alias("player_id"),
        pl.col("display_name").alias("player_name"),
        pl.col("position"),
        pl.col("latest_team").alias("team"),
    )


def _normalize(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _tokens(name: str) -> list[str]:
    return _normalize(name).split()


def _to_match(row: dict) -> PlayerMatch:
    return PlayerMatch(
        player_id=row["player_id"], player_name=row["player_name"], position=row["position"], team=row["team"]
    )


def _tokens_match(query_tokens: list[str], candidate_tokens: list[str]) -> bool:
    """True if every query token corresponds to some candidate token,
    exactly or as a close (typo-tolerant) match -- compared token-to-
    token, not as whole strings. Whole-string ratio matching has a real
    length bias that reproduces the exact bug this module exists to fix:
    a bare surname like "McCaffrey" scores *higher* by difflib's ratio
    against the shorter "Luke McCaffrey" than the longer "Christian
    McCaffrey", so it would silently "resolve" to the wrong brother
    instead of coming back ambiguous. Comparing token-to-token removes
    that bias -- "mccaffrey" matches the surname token in both names
    equally, correctly leaving it ambiguous."""
    for query_token in query_tokens:
        if not any(
            query_token == candidate_token
            or difflib.SequenceMatcher(None, query_token, candidate_token).ratio() >= FUZZY_CUTOFF
            for candidate_token in candidate_tokens
        ):
            return False
    return True


def resolve_player(name_query: str, index: pl.DataFrame) -> ResolveResult:
    """Resolve name_query to a player via structured lookup: exact
    full-name match first, then token-level fuzzy matching as a fallback
    -- never embedding similarity. Ambiguity (an exact name shared by
    more than one real player, a bare surname shared by several, or
    several plausible fuzzy candidates) is returned explicitly rather
    than picking one; callers (recommend.py's tool dispatch) must
    surface that to the model/user instead of guessing.
    """
    rows = index.to_dicts()
    by_normalized: dict[str, list[dict]] = {}
    for row in rows:
        by_normalized.setdefault(_normalize(row["player_name"]), []).append(row)

    needle = _normalize(name_query)
    exact = by_normalized.get(needle, [])
    if len(exact) == 1:
        return ResolveResult("exact", [_to_match(exact[0])])
    if len(exact) > 1:
        return ResolveResult("ambiguous", [_to_match(r) for r in exact])

    query_tokens = needle.split()
    matches = [row for row in rows if _tokens_match(query_tokens, _tokens(row["player_name"]))]
    if not matches:
        return ResolveResult("none", [])
    if len(matches) == 1:
        return ResolveResult("exact", [_to_match(matches[0])])
    return ResolveResult("ambiguous", [_to_match(r) for r in matches])
