"""Matchup signals: derived, league-agnostic features computed from
nflverse play-by-play, NGS, and game-script data. See PROJECT_SPEC.md's
signals table for the full list and rationale; this module implements
everything except the Phase 4 stretch (derived coverage classification,
needs Big Data Bowl tracking data) and CROE in the narrow literal sense
(NGS doesn't publish that exact stat -- see src/ingest/ngs.py's docstring
for the proxy used here).

**As-of-week filtering is mandatory** (see CLAUDE.md's non-negotiable
methodology rules): every signal below that trails performance (defense
run-funnel, red zone share, efficiency trend, target share) is computed
from play-by-play with week < as_of_week only -- never the target week
itself or later, or a backtest would leak the outcome it's trying to
predict. Game script / implied total is the one exception: that's the
*pregame* line for as_of_week itself, which is legitimately known before
kickoff.

This module has no knowledge of any specific Sleeper league, roster, or
scoring format -- signals are computed once per season/week and shared
across every league, per CLAUDE.md's key architectural principle. The
per-league join (whose roster a player is on, what that league's scoring
settings are worth) happens in recommend.py (Phase 3) / src/api/
(Phase 5), not here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src.ingest import nflverse, ngs, odds

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "signals"
REG_SEASON_TYPE = "REG"


def _history(pbp: pl.DataFrame, as_of_week: int) -> pl.DataFrame:
    """Regular-season plays strictly before as_of_week -- the only plays
    any trailing signal below is allowed to see."""
    return pbp.filter((pl.col("season_type") == REG_SEASON_TYPE) & (pl.col("week") < as_of_week))


def defense_run_funnel_rate(pbp: pl.DataFrame, as_of_week: int) -> pl.DataFrame:
    """Per defense, share of yards allowed that came on the ground vs.
    through the air, through as_of_week, and how that compares to the
    league average (positive run_funnel_vs_avg = funnels opponents to the
    run more than average -- a positive signal for facing RBs)."""
    hist = _history(pbp, as_of_week)
    plays = hist.filter((pl.col("rush") == 1) | (pl.col("pass") == 1))
    agg = plays.group_by("defteam").agg(
        rush_yards_allowed=pl.col("yards_gained").filter(pl.col("rush") == 1).sum(),
        pass_yards_allowed=pl.col("yards_gained").filter(pl.col("pass") == 1).sum(),
    )
    agg = agg.with_columns(
        run_funnel_rate=pl.col("rush_yards_allowed")
        / (pl.col("rush_yards_allowed") + pl.col("pass_yards_allowed"))
    )
    league_avg = agg["run_funnel_rate"].mean()
    return agg.rename({"defteam": "team"}).with_columns(
        (pl.col("run_funnel_rate") - league_avg).alias("run_funnel_rate_vs_avg")
    )


def red_zone_role_share(pbp: pl.DataFrame, as_of_week: int) -> pl.DataFrame:
    """Per player, share of their team's red zone (yardline_100 <= 20)
    rush attempts + targets through as_of_week -- a TD-equity proxy."""
    hist = _history(pbp, as_of_week)
    canonical = _current_player_reference(hist)
    rz = hist.filter(pl.col("yardline_100") <= 20)
    rushes = rz.filter((pl.col("rush") == 1) & pl.col("rusher_player_id").is_not_null()).select(
        pl.col("posteam").alias("team"),
        pl.col("rusher_player_id").alias("player_id"),
        pl.col("rusher_player_name").alias("player_name"),
    )
    targets = rz.filter((pl.col("pass") == 1) & pl.col("receiver_player_id").is_not_null()).select(
        pl.col("posteam").alias("team"),
        pl.col("receiver_player_id").alias("player_id"),
        pl.col("receiver_player_name").alias("player_name"),
    )
    touches = pl.concat([rushes, targets])
    if touches.is_empty():
        return touches.with_columns(
            pl.lit(0).alias("red_zone_touches"),
            pl.lit(0).alias("team_red_zone_plays"),
            pl.lit(None, dtype=pl.Float64).alias("red_zone_share"),
        )

    # team_red_zone_plays is every red zone touch for that team this
    # season, by whoever was on the field for it -- computed from the
    # full, unrestricted touches so a traded player's pre-trade plays
    # still count toward their old team's total. Only the player-level
    # numerator is restricted to each player's current-team identity.
    per_team = touches.group_by("team").agg(team_red_zone_plays=pl.len())
    player_touches = _restrict_to_current_team(touches, canonical)
    per_player = player_touches.group_by(["player_id", "player_name", "team"]).agg(
        red_zone_touches=pl.len()
    )
    return per_player.join(per_team, on="team").with_columns(
        (pl.col("red_zone_touches") / pl.col("team_red_zone_plays")).alias("red_zone_share")
    )


def _current_player_reference(hist: pl.DataFrame) -> pl.DataFrame:
    """Each player's most-recently-used team and name in hist, as
    [player_id, team, player_name]. Resolves two real-world quirks in
    nflverse pbp that would otherwise multiply a player's rows in every
    downstream group_by(player_id, ...): mid-season trades (the same
    player_id appears under multiple teams, e.g. Davante Adams' LV->NYJ
    move) and inconsistent name formatting for the same player within a
    season (e.g. "Di.Johnson" vs "Dio.Johnson" for Diontae Johnson).
    Ties within the same week keep an arbitrary but deterministic pick;
    that only matters for same-week name-variant noise, not team
    identity. See _restrict_to_current_team."""
    rush = hist.filter(pl.col("rusher_player_id").is_not_null()).select(
        pl.col("rusher_player_id").alias("player_id"),
        pl.col("posteam").alias("team"),
        pl.col("rusher_player_name").alias("player_name"),
        pl.col("week"),
    )
    targets = hist.filter(pl.col("receiver_player_id").is_not_null()).select(
        pl.col("receiver_player_id").alias("player_id"),
        pl.col("posteam").alias("team"),
        pl.col("receiver_player_name").alias("player_name"),
        pl.col("week"),
    )
    passes = hist.filter(pl.col("passer_player_id").is_not_null()).select(
        pl.col("passer_player_id").alias("player_id"),
        pl.col("posteam").alias("team"),
        pl.col("passer_player_name").alias("player_name"),
        pl.col("week"),
    )
    appearances = pl.concat([rush, targets, passes])
    if appearances.is_empty():
        return appearances.drop("week")
    return (
        appearances.sort("week")
        .group_by("player_id", maintain_order=True)
        .last()
        .select("player_id", "team", "player_name")
    )


def _restrict_to_current_team(rows: pl.DataFrame, canonical: pl.DataFrame) -> pl.DataFrame:
    """Keep only the rows that happened on each player's current (most
    recent) team, and replace whatever name variant appears on that row
    with the canonical one. This intentionally *drops* a traded player's
    pre-trade rows rather than relabeling them onto their new team --
    relabeling would silently pollute the new team's own totals (e.g. a
    red-zone-play or target denominator) with plays that never actually
    happened there. `rows` must have [player_id, team, player_name, ...];
    any other columns pass through unchanged."""
    return rows.drop("player_name").join(canonical, on=["player_id", "team"], how="inner")


def _role_epa_plays(hist: pl.DataFrame) -> pl.DataFrame:
    """Every play attributed to the player whose role it measures: rush
    attempts to the rusher, targets to the receiver, pass attempts to the
    passer. A play appears at most once per role, so a QB's scrambles and
    throws both count without double-counting any single play."""
    rush_rows = hist.filter((pl.col("rush") == 1) & pl.col("rusher_player_id").is_not_null()).select(
        pl.col("posteam").alias("team"),
        pl.col("rusher_player_id").alias("player_id"),
        pl.col("rusher_player_name").alias("player_name"),
        pl.col("week"),
        pl.col("epa"),
    )
    target_rows = hist.filter((pl.col("pass") == 1) & pl.col("receiver_player_id").is_not_null()).select(
        pl.col("posteam").alias("team"),
        pl.col("receiver_player_id").alias("player_id"),
        pl.col("receiver_player_name").alias("player_name"),
        pl.col("week"),
        pl.col("epa"),
    )
    pass_rows = hist.filter((pl.col("pass") == 1) & pl.col("passer_player_id").is_not_null()).select(
        pl.col("posteam").alias("team"),
        pl.col("passer_player_id").alias("player_id"),
        pl.col("passer_player_name").alias("player_name"),
        pl.col("week"),
        pl.col("epa"),
    )
    return pl.concat([rush_rows, target_rows, pass_rows])


def recent_efficiency_trend(pbp: pl.DataFrame, as_of_week: int, trailing_games: int = 3) -> pl.DataFrame:
    """Per player, EPA/play over the last `trailing_games` weeks vs. their
    season-to-date average (both strictly before as_of_week). Positive
    epa_trend = trending up; null when a player has no plays in the
    trailing window (e.g. they only played early in the season)."""
    hist = _history(pbp, as_of_week)
    canonical = _current_player_reference(hist)
    # Restricting to each player's current-team stint here is also the
    # semantically right call, not just a dedup fix: a player's efficiency
    # trend should reflect his current offense, not averaged in with
    # whatever he did for a team he's no longer on.
    role_plays = _restrict_to_current_team(_role_epa_plays(hist), canonical)
    season = role_plays.group_by(["player_id", "player_name", "team"]).agg(
        season_epa_per_play=pl.col("epa").mean(), season_plays=pl.len()
    )
    trailing_window = role_plays.filter(pl.col("week") >= as_of_week - trailing_games)
    trailing = trailing_window.group_by(["player_id", "player_name", "team"]).agg(
        trailing_epa_per_play=pl.col("epa").mean(), trailing_plays=pl.len()
    )
    trend = season.join(trailing, on=["player_id", "player_name", "team"], how="left")
    return trend.with_columns(
        (pl.col("trailing_epa_per_play") - pl.col("season_epa_per_play")).alias("epa_trend")
    )


def _opponent_map(schedules: pl.DataFrame, as_of_week: int) -> pl.DataFrame:
    """Each team's opponent in as_of_week, as [team, opponent]. A team on
    a bye that week simply has no row here, so joins against this map
    correctly leave that team's opponent null rather than stale."""
    sched_week = schedules.filter(pl.col("week") == as_of_week)
    return pl.concat(
        [
            sched_week.select(pl.col("home_team").alias("team"), pl.col("away_team").alias("opponent")),
            sched_week.select(pl.col("away_team").alias("team"), pl.col("home_team").alias("opponent")),
        ]
    )


def opponent_adjusted_target_share(pbp: pl.DataFrame, schedules: pl.DataFrame, as_of_week: int) -> pl.DataFrame:
    """Per player, target share (of their team's targets, through
    as_of_week) reweighted by their as_of_week opponent's pass defense:
    target_share_adjusted = target_share * (1 + 0.1 * opponent's pass EPA
    allowed z-score). A positive z-score means the opponent allows
    more efficient pass plays than average (an easier matchup), so it
    scales the share up; a tougher-than-average defense scales it down.
    The 0.1 weight is a deliberately modest, simple reweighting -- not a
    fitted model -- so raw target_share is also returned for callers who
    want it unadjusted. Players on a bye in as_of_week get a null
    opponent and null adjusted share (there's no matchup to adjust for).
    """
    hist = _history(pbp, as_of_week)
    canonical = _current_player_reference(hist)
    targets = hist.filter((pl.col("pass") == 1) & pl.col("receiver_player_id").is_not_null())

    # team_targets is every target thrown by that team this season,
    # regardless of who caught it -- computed from the full, unrestricted
    # targets so a traded player's pre-trade targets still count toward
    # their old team's total. Only the player-level numerator below is
    # restricted to each player's current-team identity.
    per_team = targets.group_by("posteam").agg(team_targets=pl.len()).rename({"posteam": "team"})

    player_targets = targets.select(
        pl.col("receiver_player_id").alias("player_id"),
        pl.col("posteam").alias("team"),
        pl.col("receiver_player_name").alias("player_name"),
    )
    player_targets = _restrict_to_current_team(player_targets, canonical)
    per_player = player_targets.group_by(["player_id", "player_name", "team"]).agg(targets=pl.len())
    per_player = per_player.join(per_team, on="team").with_columns(
        (pl.col("targets") / pl.col("team_targets")).alias("target_share")
    )

    pass_def = hist.filter(pl.col("pass") == 1).group_by("defteam").agg(
        opponent_pass_epa_allowed=pl.col("epa").mean()
    ).rename({"defteam": "team"})
    league_mean = pass_def["opponent_pass_epa_allowed"].mean()
    league_std = pass_def["opponent_pass_epa_allowed"].std()
    pass_def = pass_def.with_columns(
        ((pl.col("opponent_pass_epa_allowed") - league_mean) / league_std).alias(
            "opponent_pass_epa_allowed_z"
        )
    )

    per_player = per_player.join(_opponent_map(schedules, as_of_week), on="team", how="left")
    per_player = per_player.join(
        pass_def.rename({"team": "opponent"}), on="opponent", how="left"
    )
    return per_player.with_columns(
        (pl.col("target_share") * (1 + 0.1 * pl.col("opponent_pass_epa_allowed_z"))).alias(
            "target_share_adjusted"
        )
    )


def _ngs_trailing_average(df: pl.DataFrame, as_of_week: int, value_cols: list[str]) -> pl.DataFrame:
    """Season-to-date (weeks < as_of_week) average of value_cols per
    player, keyed by NGS's player_gsis_id -- the same ID format pbp uses
    for rusher/receiver/passer_player_id, so this joins directly onto the
    pbp-derived tables above without name matching."""
    hist = df.filter(pl.col("week") < as_of_week)
    return hist.group_by("player_gsis_id").agg(
        [pl.col(c).mean().alias(c) for c in value_cols]
    ).rename({"player_gsis_id": "player_id"})


def build_signals_table(
    season: int,
    as_of_week: int,
    pbp: pl.DataFrame,
    schedules: pl.DataFrame,
    ngs_receiving: pl.DataFrame | None = None,
    ngs_rushing: pl.DataFrame | None = None,
    ngs_passing: pl.DataFrame | None = None,
    trailing_games: int = 3,
) -> list[dict]:
    """Combine every signal above into one row per player, as of
    as_of_week. This is the table src/rag's signal-chunk builder turns
    into per-player retrievable text (see TODO.md's Phase 2 chunk
    granularity rule) and what Phase 3's recommend.py will eventually
    query directly. Rows are keyed by player_id (nflverse gsis_id), not
    Sleeper's player_id -- resolving between the two ID spaces is part of
    the per-league join, not this league-agnostic layer.
    """
    trend = recent_efficiency_trend(pbp, as_of_week, trailing_games)
    rz = red_zone_role_share(pbp, as_of_week)
    targets = opponent_adjusted_target_share(pbp, schedules, as_of_week)
    defense = defense_run_funnel_rate(pbp, as_of_week)
    implied_totals = odds.implied_team_totals(schedules).filter(pl.col("week") == as_of_week)

    table = trend.select("player_id", "player_name", "team", "season_plays", "epa_trend")
    table = table.join(
        rz.select("player_id", "red_zone_touches", "red_zone_share"), on="player_id", how="left"
    )
    # Opponent comes from the player's own team + schedule, not from the
    # targets join below -- a pure rusher has no targets row at all, and
    # would otherwise never get an opponent (or opponent defense signal)
    # attached despite being exactly who run-funnel rate matters most for.
    table = table.join(_opponent_map(schedules, as_of_week), on="team", how="left")
    table = table.join(
        targets.select("player_id", "target_share", "target_share_adjusted"),
        on="player_id",
        how="left",
    )
    table = table.join(
        defense.rename({"team": "opponent"}).select("opponent", "run_funnel_rate_vs_avg"),
        on="opponent",
        how="left",
    )
    table = table.join(
        implied_totals.select(pl.col("team"), pl.col("implied_total")), on="team", how="left"
    )

    if ngs_receiving is not None:
        table = table.join(
            _ngs_trailing_average(
                ngs_receiving,
                as_of_week,
                ["avg_intended_air_yards", "avg_separation", "avg_yac_above_expectation", "catch_percentage"],
            ).rename(
                {
                    "avg_intended_air_yards": "adot",
                    "avg_yac_above_expectation": "croe_proxy_yac_oe",
                    "avg_separation": "croe_proxy_separation",
                }
            ),
            on="player_id",
            how="left",
        )
    if ngs_rushing is not None:
        table = table.join(
            _ngs_trailing_average(ngs_rushing, as_of_week, ["rush_yards_over_expected_per_att"]).rename(
                {"rush_yards_over_expected_per_att": "ryoe_per_att"}
            ),
            on="player_id",
            how="left",
        )
    if ngs_passing is not None:
        table = table.join(
            _ngs_trailing_average(
                ngs_passing, as_of_week, ["completion_percentage_above_expectation"]
            ).rename({"completion_percentage_above_expectation": "cpoe"}),
            on="player_id",
            how="left",
        )

    table = table.with_columns(pl.lit(season).alias("season"), pl.lit(as_of_week).alias("as_of_week"))
    return table.to_dicts()


def save_signals_table(rows: list[dict], season: int, as_of_week: int, out_dir: Path = PROCESSED_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"signals_{season}_week{as_of_week}.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute matchup signals for one season/as-of-week")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--as-of-week", type=int, required=True)
    args = parser.parse_args()

    pbp = nflverse.fetch_pbp(args.season)
    schedules = nflverse.fetch_schedules(args.season)
    ngs_receiving = ngs.fetch_ngs(args.season, "receiving")
    ngs_rushing = ngs.fetch_ngs(args.season, "rushing")
    ngs_passing = ngs.fetch_ngs(args.season, "passing")

    rows = build_signals_table(
        args.season,
        args.as_of_week,
        pbp,
        schedules,
        ngs_receiving=ngs_receiving,
        ngs_rushing=ngs_rushing,
        ngs_passing=ngs_passing,
    )
    path = save_signals_table(rows, args.season, args.as_of_week)
    print(f"wrote {len(rows)} signal rows -> {path}")


if __name__ == "__main__":
    main()
