import polars as pl

from src.rag import player_index as pi

# Real nflverse player records, verified live against nflreadpy's
# load_players() during development (see PR description) -- these are
# the exact identities behind the reported bug: asking about "Christian
# McCaffrey" returned his brother Luke's signal chunk instead, and
# "Caleb Williams" never surfaced at all among the many NFL Williamses.
_PLAYERS = pl.DataFrame(
    [
        {"gsis_id": "00-0033280", "display_name": "Christian McCaffrey", "position": "RB", "latest_team": "SF", "last_season": 2026},
        {"gsis_id": "00-0039355", "display_name": "Luke McCaffrey", "position": "WR", "latest_team": "WAS", "last_season": 2026},
        {"gsis_id": "00-0010650", "display_name": "Ed McCaffrey", "position": "WR", "latest_team": "DEN", "last_season": 2003},
        {"gsis_id": "00-0032937", "display_name": "Max McCaffrey", "position": "WR", "latest_team": "SF", "last_season": 2018},
        {"gsis_id": "00-0039918", "display_name": "Caleb Williams", "position": "QB", "latest_team": "CHI", "last_season": 2026},
        {"gsis_id": "00-0023440", "display_name": "Cadillac Williams", "position": "RB", "latest_team": "LA", "last_season": 2011},
        # Two real "Mike Williams" WRs in nflverse's database -- only one
        # is recent (2025); the other's last active season was 2014. This
        # is the real case build_player_index's recency filter exists for:
        # without it, a common name+position pair would be falsely
        # ambiguous forever, even when only one is actually still around.
        {"gsis_id": "00-0033536", "display_name": "Mike Williams", "position": "WR", "latest_team": "LAC", "last_season": 2025},
        {"gsis_id": "00-0027702", "display_name": "Mike Williams", "position": "WR", "latest_team": "BUF", "last_season": 2014},
        # Legacy (non "00-XXXXXXX") gsis_id format -- pre-modern-era
        # records that build_player_index must exclude since pbp/signals
        # never use this ID space.
        {"gsis_id": "WIL142164", "display_name": "Al Williams", "position": "WR", "latest_team": "LAC", "last_season": 1987},
    ]
)


def test_build_player_index_filters_to_recent_modern_id_players():
    index = pi.build_player_index(season=2024, players=_PLAYERS)
    names = set(index["player_name"].to_list())

    assert "Christian McCaffrey" in names
    assert "Luke McCaffrey" in names
    assert "Caleb Williams" in names
    # Ed and Max McCaffrey, and old Mike Williams, are stale (last_season
    # well before 2023) -- excluded so they can't manufacture false
    # ambiguity for a current question.
    assert "Ed McCaffrey" not in names
    assert "Max McCaffrey" not in names
    assert index.filter(pl.col("player_id") == "00-0027702").is_empty()
    # Legacy-format id excluded regardless of recency.
    assert index.filter(pl.col("player_id") == "WIL142164").is_empty()
    # The recent Mike Williams survives.
    assert index.filter(pl.col("player_id") == "00-0033536").height == 1


def test_resolve_player_christian_mccaffrey_does_not_return_his_brother():
    index = pi.build_player_index(season=2024, players=_PLAYERS)

    result = pi.resolve_player("Christian McCaffrey", index)

    assert result.match_type == "exact"
    assert len(result.candidates) == 1
    assert result.candidates[0].player_id == "00-0033280"
    assert result.candidates[0].team == "SF"


def test_resolve_player_luke_mccaffrey_is_not_confused_with_christian():
    index = pi.build_player_index(season=2024, players=_PLAYERS)

    result = pi.resolve_player("Luke McCaffrey", index)

    assert result.match_type == "exact"
    assert result.candidates[0].player_id == "00-0039355"
    assert result.candidates[0].player_id != "00-0033280"


def test_resolve_player_caleb_williams_is_found_and_unambiguous():
    index = pi.build_player_index(season=2024, players=_PLAYERS)

    result = pi.resolve_player("Caleb Williams", index)

    assert result.match_type == "exact"
    assert len(result.candidates) == 1
    assert result.candidates[0].player_id == "00-0039918"
    assert result.candidates[0].position == "QB"


def test_resolve_player_mike_williams_resolves_to_the_recent_one():
    # Before recency filtering there are two real Mike Williams WRs;
    # build_player_index already dropped the stale one, so this should
    # be a clean exact match, not "ambiguous".
    index = pi.build_player_index(season=2024, players=_PLAYERS)

    result = pi.resolve_player("Mike Williams", index)

    assert result.match_type == "exact"
    assert result.candidates[0].player_id == "00-0033536"


def test_resolve_player_returns_ambiguous_for_a_genuinely_shared_name():
    # Synthetic case: two real, both-current players sharing a literal
    # full name (this does happen in the NFL -- shared names aren't
    # unique to McCaffrey/Williams) must never be silently resolved to
    # whichever one sorts first.
    players = pl.DataFrame(
        [
            {"gsis_id": "00-0011111", "display_name": "Sam Example", "position": "RB", "latest_team": "SF", "last_season": 2026},
            {"gsis_id": "00-0022222", "display_name": "Sam Example", "position": "WR", "latest_team": "DAL", "last_season": 2026},
        ]
    )
    index = pi.build_player_index(season=2024, players=players)

    result = pi.resolve_player("Sam Example", index)

    assert result.match_type == "ambiguous"
    assert {c.player_id for c in result.candidates} == {"00-0011111", "00-0022222"}


def test_resolve_player_no_match_returns_none_not_a_guess():
    index = pi.build_player_index(season=2024, players=_PLAYERS)

    result = pi.resolve_player("Zzzzyx Nobody", index)

    assert result.match_type == "none"
    assert result.candidates == []


def test_resolve_player_fuzzy_matches_a_close_typo():
    index = pi.build_player_index(season=2024, players=_PLAYERS)

    result = pi.resolve_player("Caleb Willaims", index)  # transposed letters

    assert result.match_type == "exact"
    assert result.candidates[0].player_id == "00-0039918"


def test_resolve_player_bare_surname_does_not_guess_between_mccaffreys():
    # "McCaffrey" alone shouldn't confidently resolve to either brother --
    # both are still-current players sharing that surname, so this must
    # come back ambiguous, never a single silent guess.
    index = pi.build_player_index(season=2024, players=_PLAYERS)

    result = pi.resolve_player("McCaffrey", index)

    assert result.match_type == "ambiguous"
    assert {c.player_id for c in result.candidates} == {"00-0033280", "00-0039355"}
