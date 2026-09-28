"""Fit and test the availability table (src/signals/play_rates.json):
P(played | designation, practice status, position), from nflverse's
injury reports joined to snap counts ("played" = at least one offensive
snap that week).

Fit seasons are pooled counts; the test season is held out and reports
(a) calibration per group -- predicted rate vs. what actually happened,
(b) Brier score against two naive baselines (every Questionable player
plays; Questionable = 50/50), and (c) a lineup test: for every
Questionable/Doubtful player with a season-to-date half-PPR average,
paired against a healthy player at the same position with the closest
average, does "average x play probability" pick the higher actual scorer
more often than the raw average does?

Writes src/signals/play_rates.json (the table the product reads) and
evals/results/<date>_play_rates.json (everything measured).

    python -m evals.fit_play_rates --fit 2020 2021 2022 2023 2024 --test 2025
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from src.signals import availability as av

RESULTS_DIR = Path(__file__).resolve().parent / "results"
POSITIONS = list(av.SKILL_POSITIONS)


def labeled_reports(seasons: list[int]) -> pl.DataFrame:
    """Injury-report rows for skill positions with `played` attached."""
    inj = pl.concat([av.fetch_injury_reports(s) for s in seasons])
    snaps = (
        nfl.load_snap_counts(seasons)
        .filter(pl.col("game_type") == "REG")
        .select(["pfr_player_id", pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64), "offense_snaps"])
    )
    ids = nfl.load_players().select(["gsis_id", "pfr_id"]).drop_nulls().unique(subset=["gsis_id"])
    return (
        inj.join(ids, on="gsis_id", how="left")
        .join(snaps, left_on=["pfr_id", "season", "week"], right_on=["pfr_player_id", "season", "week"], how="left")
        .with_columns(
            (pl.col("offense_snaps").fill_null(0) > 0).alias("played"),
            pl.col("practice_status").map_elements(av.practice_bucket, return_dtype=pl.Utf8).alias("bucket"),
        )
        .filter(pl.col("report_status").is_in(list(av.PROBABILISTIC_STATUSES)))
    )


def fit(df: pl.DataFrame) -> dict:
    rates: dict[str, dict] = {}
    for status in av.PROBABILISTIC_STATUSES:
        sub = df.filter(pl.col("report_status") == status)
        for bucket in ("full", "limited", "dnp", "unknown"):
            b = sub.filter(pl.col("bucket") == bucket)
            if b.height:
                rates[av.rate_key(status, bucket, None)] = {"n": b.height, "rate": round(b["played"].mean(), 3)}
            for pos in POSITIONS:
                bp = b.filter(pl.col("position") == pos)
                if bp.height:
                    rates[av.rate_key(status, bucket, pos)] = {"n": bp.height, "rate": round(bp["played"].mean(), 3)}
    return rates


def predict(df: pl.DataFrame, rates: dict) -> pl.DataFrame:
    table = {"rates": rates}
    return df.with_columns(
        pl.struct(["report_status", "practice_status", "position"])
        .map_elements(lambda s: av.play_probability(s["report_status"], s["practice_status"], s["position"], table),
                      return_dtype=pl.Float64)
        .alias("p")
    ).filter(pl.col("p").is_not_null())


def brier(df: pl.DataFrame, col: str) -> float:
    return round(((df[col] - df["played"].cast(pl.Float64)) ** 2).mean(), 4)


def calibration(df: pl.DataFrame) -> list[dict]:
    return [
        {"status": r["report_status"], "practice": r["bucket"], "n": r["n"], "predicted": round(r["p"], 3),
         "actual": round(r["actual"], 3)}
        for r in df.group_by(["report_status", "bucket"]).agg(pl.len().alias("n"), pl.col("p").mean(), pl.col("played").mean().alias("actual"))
        .sort(["report_status", "n"], descending=[False, True]).iter_rows(named=True)
    ]


def half_ppr() -> pl.Expr:
    return (
        0.04 * pl.col("passing_yards") + 4 * pl.col("passing_tds") - 2 * pl.col("passing_interceptions")
        + 0.1 * pl.col("rushing_yards") + 6 * pl.col("rushing_tds")
        + 0.5 * pl.col("receptions") + 0.1 * pl.col("receiving_yards") + 6 * pl.col("receiving_tds")
        - 2 * (pl.col("rushing_fumbles_lost") + pl.col("receiving_fumbles_lost") + pl.col("sack_fumbles_lost"))
    )


def season_points(season: int) -> pl.DataFrame:
    return (
        nfl.load_player_stats([season])
        .filter((pl.col("season_type") == "REG") & pl.col("position").is_in(POSITIONS))
        .with_columns(half_ppr().alias("pts"))
        .select(["player_id", "position", pl.col("week").cast(pl.Int64), "pts"])
    )


def output_factors(seasons: list[int], labeled: pl.DataFrame, min_prior_games: int = 3) -> dict:
    """Per designation (and practice bucket, when big enough): among
    designated players who DID play, total actual points / total
    season-to-date average -- the share of his usual output a player
    delivers through an injury. A second discount on top of the play
    rate: a Questionable player who suits up is usually limited."""
    rows = []
    for season in seasons:
        stats = season_points(season)
        for week in sorted(stats["week"].unique().to_list()):
            if week < min_prior_games + 1:
                continue
            prior = stats.filter(pl.col("week") < week).group_by("player_id").agg(pl.col("pts").mean().alias("avg"), pl.len().alias("games")).filter(pl.col("games") >= min_prior_games)
            actual = {r["player_id"]: r["pts"] for r in stats.filter(pl.col("week") == week).iter_rows(named=True)}
            avg = {r["player_id"]: r["avg"] for r in prior.iter_rows(named=True)}
            for r in labeled.filter((pl.col("season") == season) & (pl.col("week") == week) & pl.col("played")).iter_rows(named=True):
                if r["gsis_id"] in avg and avg[r["gsis_id"]] > 0:
                    rows.append({"status": r["report_status"], "bucket": r["bucket"], "avg": avg[r["gsis_id"]], "actual": actual.get(r["gsis_id"], 0.0)})
    df = pl.DataFrame(rows)
    out: dict[str, dict] = {}
    for status in av.PROBABILISTIC_STATUSES:
        sub = df.filter(pl.col("status") == status)
        if sub.height:
            out[av.rate_key(status, "unknown", None)] = {"n": sub.height, "factor": round(sub["actual"].sum() / sub["avg"].sum(), 3)}
        for bucket in ("full", "limited", "dnp"):
            b = sub.filter(pl.col("bucket") == bucket)
            if b.height:
                out[av.rate_key(status, bucket, None)] = {"n": b.height, "factor": round(b["actual"].sum() / b["avg"].sum(), 3)}
    return out


def lineup_test(test_season: int, predicted: pl.DataFrame, factors: dict, min_prior_games: int = 3) -> dict:
    """Pairs a designated player against the healthy same-position player
    with the closest season-to-date average, and scores which one actually
    outscored the other (a player who did not play scores 0)."""
    stats = season_points(test_season)
    weeks = sorted(stats["week"].unique().to_list())
    designated = {(r["gsis_id"], r["week"]): r for r in predicted.iter_rows(named=True)}
    all_report_ids = {(r["gsis_id"], r["week"]) for r in predicted.iter_rows(named=True)}
    pairs = []
    for week in weeks:
        if week < min_prior_games + 1:
            continue
        prior = stats.filter(pl.col("week") < week).group_by(["player_id", "position"]).agg(
            pl.col("pts").mean().alias("avg"), pl.len().alias("games")
        ).filter(pl.col("games") >= min_prior_games)
        actual = {r["player_id"]: r["pts"] for r in stats.filter(pl.col("week") == week).iter_rows(named=True)}
        rows = prior.to_dicts()
        healthy_by_pos: dict[str, list[dict]] = {}
        for r in rows:
            if (r["player_id"], week) not in all_report_ids and r["player_id"] in actual:
                healthy_by_pos.setdefault(r["position"], []).append(r)
        for r in rows:
            d = designated.get((r["player_id"], week))
            if d is None:
                continue
            pool = healthy_by_pos.get(r["position"]) or []
            if not pool:
                continue
            # Healthy partners from 40% to 120% of the designated player's
            # average (closest to each of five rungs), so the test covers
            # "clearly worse" through "slightly better" alternatives rather
            # than only the coin-flip pair a nearest-neighbour would pick.
            for ratio in (0.4, 0.6, 0.8, 1.0, 1.2):
                h = min(pool, key=lambda x: abs(x["avg"] - r["avg"] * ratio))
                k = av.output_factor(d["report_status"], d["practice_status"], {"factors": factors})
                pairs.append({
                    "week": week, "status": d["report_status"], "bucket": d["bucket"], "p": d["p"], "k": k if k is not None else 1.0, "ratio": ratio,
                    "q_avg": r["avg"], "h_avg": h["avg"], "q_pts": actual.get(r["player_id"], 0.0), "h_pts": actual[h["player_id"]],
                })
    pdf = pl.DataFrame(pairs)
    if pdf.is_empty():
        return {"pairs": 0}
    pdf = pdf.with_columns(
        (pl.col("q_pts") > pl.col("h_pts")).alias("q_won"),
        (pl.col("q_avg") >= pl.col("h_avg")).alias("raw_picks_q"),
        (pl.col("q_avg") * pl.col("p") >= pl.col("h_avg")).alias("adj_picks_q"),
        (pl.col("q_avg") * pl.col("p") * pl.col("k") >= pl.col("h_avg")).alias("adj2_picks_q"),
    ).with_columns(
        (pl.col("raw_picks_q") == pl.col("q_won")).alias("raw_right"),
        (pl.col("adj_picks_q") == pl.col("q_won")).alias("adj_right"),
        (pl.col("adj2_picks_q") == pl.col("q_won")).alias("adj2_right"),
        (~pl.col("q_won")).alias("healthy_right"),  # the "never start a designated player" rule
    )
    def groups(keys):
        return [
            {**{k: r[k] for k in keys}, "pairs": r["n"], "raw_accuracy": round(r["raw"], 3), "adjusted_accuracy": round(r["adj"], 3),
             "adjusted_with_output_factor_accuracy": round(r["adj2"], 3), "always_healthy_accuracy": round(r["hea"], 3)}
            for r in pdf.group_by(keys).agg(pl.len().alias("n"), pl.col("raw_right").mean().alias("raw"), pl.col("adj_right").mean().alias("adj"),
                                            pl.col("adj2_right").mean().alias("adj2"), pl.col("healthy_right").mean().alias("hea"))
            .sort(keys).iter_rows(named=True)
        ]
    by = groups(["status", "bucket"])
    by_ratio = groups(["ratio"])
    return {
        "pairs": pdf.height,
        "description": "designated player vs the healthy same-position player with the closest season-to-date half-PPR average; "
                       "raw = pick the higher average, adjusted = pick the higher average x play probability, "
                       "adjusted_with_output_factor = average x play probability x share-of-usual-output when playing",
        "raw_accuracy": round(pdf["raw_right"].mean(), 3),
        "adjusted_accuracy": round(pdf["adj_right"].mean(), 3),
        "adjusted_with_output_factor_accuracy": round(pdf["adj2_right"].mean(), 3),
        "always_healthy_accuracy": round(pdf["healthy_right"].mean(), 3),
        "designated_actually_outscored": round(pdf["q_won"].mean(), 3),
        "by_group": by,
        "by_healthy_avg_ratio": by_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024])
    parser.add_argument("--test", type=int, default=2025)
    parser.add_argument("--write", action="store_true", help="write src/signals/play_rates.json, fitted on --fit plus --test")
    args = parser.parse_args()

    train = labeled_reports(args.fit)
    rates = fit(train)
    factors = output_factors(args.fit, train)
    test = predict(labeled_reports([args.test]), rates)
    naive_all_play = test.with_columns(pl.when(pl.col("report_status") == "Questionable").then(1.0).otherwise(0.0).alias("p1"))
    naive_half = test.with_columns(pl.when(pl.col("report_status") == "Questionable").then(0.5).otherwise(0.0).alias("p1"))
    pooled_rates = {av.rate_key(st, b, None): rates.get(av.rate_key(st, "unknown", None), {"n": 0, "rate": 0.0}) for st in av.PROBABILISTIC_STATUSES for b in ("full", "limited", "dnp", "unknown")}
    for st in av.PROBABILISTIC_STATUSES:  # pooled = every practice bucket of a designation shares its overall rate
        sub = train.filter(pl.col("report_status") == st)
        for b in ("full", "limited", "dnp", "unknown"):
            pooled_rates[av.rate_key(st, b, None)] = {"n": sub.height, "rate": round(sub["played"].mean(), 3)}
    pooled = predict(labeled_reports([args.test]), pooled_rates).rename({"p": "p1"})
    out = {
        "date": str(date.today()),
        "fit_seasons": args.fit, "test_season": args.test,
        "fit_rows": train.height, "test_rows": test.height,
        "test_brier": brier(test, "p"),
        "baseline_brier_every_questionable_plays": brier(naive_all_play, "p1"),
        "baseline_brier_questionable_is_50_50": brier(naive_half, "p1"),
        "pooled_per_designation_brier": brier(pooled, "p1"),
        "pooled_per_designation_rates": {k: v for k, v in pooled_rates.items() if k.endswith("|unknown|*")},
        "test_calibration": calibration(test),
        "output_factors": factors,
        "lineup_test": lineup_test(args.test, test, factors),
        "rates": rates,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{date.today()}_play_rates.json"
    result_path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "rates"}, indent=2))
    print("->", result_path)
    if args.write:
        everything = pl.concat([train, labeled_reports([args.test])])
        shipped = fit(everything)
        shipped_factors = output_factors(args.fit + [args.test], everything)
        av.PLAY_RATES_PATH.write_text(json.dumps(
            {"fitted_on": args.fit + [args.test], "holdout_test": {"season": args.test, "brier": out["test_brier"],
             "results_file": result_path.name}, "played_means": "at least one offensive snap that week",
             "source": "nflverse injuries + snap_counts", "rates": shipped, "factors": shipped_factors,
             "factors_mean": "among designated players who played: actual half-PPR points / season-to-date average"}, indent=2))
        print("->", av.PLAY_RATES_PATH)


if __name__ == "__main__":
    main()
