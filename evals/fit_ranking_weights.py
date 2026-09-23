"""Fit the ranking score's weights on real history instead of guessing them.

Phase 6 / strategy 2 (2026-09-21). src/reasoning/ranking.py's opportunity
score was a hand-set linear combination ("simple, documented, not
fitted"), and the first real decision eval scored 52% on 400 dilemmas.
This script builds the same features the product has at query time --
strictly as-of each week -- for every regular-season week of the train
seasons, forms same-position pairs of players who both played that week,
labels each pair by who actually scored more (nflverse box scores under
one fixed scoring format), and fits a logistic model on the feature
DIFFERENCES with no intercept. A no-intercept pairwise logistic model IS
a linear score: P(a > b) = sigmoid(w . (f_a - f_b)), so the fitted w are
directly the weights for score(x) = w . f(x). It is then scored on
held-out seasons against baselines (the old hand weights, points-per-game
alone, last season's points-per-game alone).

Features per player (missing -> 0, exactly as the product treats a
signal that isn't computed):
  ppg                      points per game this season, weeks < as-of  (Phase 6 proxy)
  ppg_prior                points per game over all of last season
  target_share             opponent-adjusted when available, else raw
  red_zone_share
  epa_trend                only when trustworthy (>= MIN_TREND_PLAYS plays), else 0
  implied_total            team's implied total this week
  cpoe                     passers only (NGS completion % over expected)
  run_funnel_rate_vs_avg   this week's opponent defense lean

Usage:
    python -m evals.fit_ranking_weights                 # train 2023, test 2024
    python -m evals.fit_ranking_weights --train 2022 2023 --test 2024

Writes evals/results/<date>_ranking_fit.json (weights, metrics, config)
and prints the weights to paste into ranking.py. No model call, no cost.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from src.ingest import nflverse, ngs
from src.reasoning import ranking
from src.signals import matchup_signals
from src.reasoning import points_proxy as pp
from src.signals import player_stats as ps

RESULTS_DIR = Path(__file__).resolve().parent / "results"
# One fixed format for fitting. Weights are unit-free multipliers on
# per-game points etc.; each league still computes its own ppg from its
# own scoring at query time.
FIT_SCORING = {"rec": 0.5, "pass_td": 4, "pass_yd": 0.04, "pass_int": -1, "pass_2pt": 2, "rush_yd": 0.1, "rush_td": 6,
               "rush_2pt": 2, "rec_yd": 0.1, "rec_td": 6, "rec_2pt": 2, "fum_lost": -2}
BASE_FEATURES = ["ppg", "ppg_prior", "target_share", "red_zone_share", "epa_trend", "implied_total", "cpoe", "run_funnel_rate_vs_avg"]
# Step 3 candidates (2026-09-22), each accepted only if held-out accuracy moves on BOTH test seasons:
#   opp_pos_allowed_rel  this week's opponent's fantasy points allowed per game to the player's position, as-of,
#                        relative to the league average for that position (allowed/avg - 1); 0 when unknown
#   ppg_last3            points per game over the previous 3 weeks (recency), as-of
#   opp_epa_allowed_rel  this week's opponent defense's EPA allowed per play on the plays that matter for the
#                        position (rush plays for RBs, pass plays for QB/WR/TE), as-of, minus the league mean --
#                        the in-season-available cousin of a coverage/scheme signal (positive = softer defense)
CANDIDATE_FEATURES = ["opp_pos_allowed_rel", "ppg_last3", "opp_epa_allowed_rel"]
FEATURES = list(BASE_FEATURES)
# --blend K replaces ppg/ppg_prior with one shrunk estimate: (games*ppg + K*ppg_prior)/(games+K) -- K prior-season
# "pseudo-games", so one big week early in the season doesn't read like a season's average. With no prior season the
# raw ppg is used; with no games yet, the prior alone.
BLEND_K: float | None = None
MIN_POINTS_BOTH = 5.0     # both players fantasy-relevant that week -- same spirit as build_decision_questions' 8
PAIRS_PER_GROUP = 300     # per (season, week, position), seeded


def blended_ppg(proxy: dict | None, prior: dict | None, k: float) -> float | None:
    ppg, games = (proxy or {}).get("ppg"), (proxy or {}).get("games_played") or 0
    prior_ppg = (prior or {}).get("ppg")
    if ppg is None and prior_ppg is None:
        return None
    if prior_ppg is None:
        return ppg
    if ppg is None:
        return prior_ppg
    return (games * ppg + k * prior_ppg) / (games + k)


def features_for(row: dict | None, proxy: dict | None, prior: dict | None, extra: dict | None = None) -> list[float]:
    row = row or {}
    extra = extra or {}
    if BLEND_K is not None:
        proxy = {"ppg": blended_ppg(proxy, prior, BLEND_K)}
        prior = {"ppg": None}
    share = row.get("target_share_adjusted")
    share = share if share is not None else row.get("target_share")
    trend = row.get("epa_trend") if (row.get("epa_trend") is not None and ranking.trend_is_trustworthy(row)) else None
    vals = {
        "ppg": (proxy or {}).get("ppg"),
        "ppg_prior": (prior or {}).get("ppg"),
        "target_share": share,
        "red_zone_share": row.get("red_zone_share"),
        "epa_trend": trend,
        "implied_total": row.get("implied_total"),
        "cpoe": row.get("cpoe"),
        "run_funnel_rate_vs_avg": row.get("run_funnel_rate_vs_avg"),
        "opp_pos_allowed_rel": extra.get("opp_pos_allowed_rel"),
        "ppg_last3": extra.get("ppg_last3"),
        "opp_epa_allowed_rel": extra.get("opp_epa_allowed_rel"),
    }
    return [float(vals[f]) if vals[f] is not None else 0.0 for f in FEATURES]


def defense_epa_allowed(pbp: pl.DataFrame, season: int, as_of_week: int) -> dict[tuple[str, str], float]:
    """Per (defense, "rush"|"pass"): mean EPA allowed per play on regular-
    season plays before as_of_week, plus the league mean keyed ("*", kind)."""
    plays = pbp.filter((pl.col("season") == season) & (pl.col("week") < as_of_week) & (pl.col("season_type") == "REG")
                       & pl.col("epa").is_not_null() & pl.col("defteam").is_not_null())
    out: dict[tuple[str, str], float] = {}
    for kind, col in (("rush", "rush_attempt"), ("pass", "pass_attempt")):
        sub = plays.filter(pl.col(col) == 1)
        if sub.is_empty():
            continue
        for row in sub.group_by("defteam").agg(pl.col("epa").mean().alias("epa")).to_dicts():
            out[(row["defteam"], kind)] = float(row["epa"])
        out[("*", kind)] = float(sub["epa"].mean())
    return out


def build_pairs(season: int, weeks: range, seed: int = 0) -> list[dict]:
    """All the as-of features + outcomes for one season, as pairs."""
    pbp = nflverse.fetch_pbp(season)
    schedules = nflverse.fetch_schedules(season)
    rec, ru, pa = (ngs.fetch_ngs(season, k) for k in ("receiving", "rushing", "passing"))
    stats = ps.fetch_weekly_stats(season)
    stats_prev = ps.fetch_weekly_stats(season - 1)
    prior = pp.season_points_proxy(stats_prev, FIT_SCORING)
    positions = {r["player_id"]: r["position"] for r in stats.select("player_id", "position").unique().to_dicts()}
    weekly = pp.points_by_week(stats, FIT_SCORING)
    rng = random.Random(seed)
    pairs = []
    for week in weeks:
        rows = {r["player_id"]: r for r in matchup_signals.build_signals_table(
            season, week, pbp, schedules, ngs_receiving=rec, ngs_rushing=ru, ngs_passing=pa)}
        proxy = pp.season_points_proxy(stats, FIT_SCORING, as_of_week=week)
        allowed = pp.points_allowed_by_position(stats, FIT_SCORING, as_of_week=week)
        last3 = pp.trailing_ppg(stats, FIT_SCORING, as_of_week=week)
        epa_allowed = defense_epa_allowed(pbp, season, week)

        def extra_for(pid: str) -> dict:
            row = rows.get(pid) or {}
            pos = positions.get(pid)
            opp = row.get("opponent")
            rel = None
            if opp and pos and (opp, pos) in allowed and ("*", pos) in allowed and allowed[("*", pos)]["allowed_pg"]:
                rel = allowed[(opp, pos)]["allowed_pg"] / allowed[("*", pos)]["allowed_pg"] - 1.0
            epa_rel = None
            if opp and pos:
                kind = "rush" if pos == "RB" else "pass"
                d = epa_allowed.get((opp, kind)); m = epa_allowed.get(("*", kind))
                if d is not None and m is not None:
                    epa_rel = d - m
            return {"opp_pos_allowed_rel": rel, "ppg_last3": (last3.get(pid) or {}).get("ppg_last"), "opp_epa_allowed_rel": epa_rel}
        actual = {r["player_id"]: r["points"] for r in weekly.filter(pl.col("week") == week).to_dicts()}
        by_pos: dict[str, list[str]] = {}
        for pid, pts in actual.items():
            if pts >= MIN_POINTS_BOTH and (pid in rows or pid in proxy or pid in prior):
                by_pos.setdefault(positions.get(pid, "?"), []).append(pid)
        for pos, pids in by_pos.items():
            if pos not in ps.SKILL_POSITIONS or len(pids) < 2:
                continue
            all_pairs = [(a, b) for i, a in enumerate(pids) for b in pids[i + 1:] if actual[a] != actual[b]]
            rng.shuffle(all_pairs)
            for a, b in all_pairs[:PAIRS_PER_GROUP]:
                fa = features_for(rows.get(a), proxy.get(a), prior.get(a), extra_for(a))
                fb = features_for(rows.get(b), proxy.get(b), prior.get(b), extra_for(b))
                pairs.append({"season": season, "week": week, "position": pos, "x": [p - q for p, q in zip(fa, fb)],
                              "y": 1 if actual[a] > actual[b] else 0, "gap": abs(actual[a] - actual[b]),
                              "fa": fa, "fb": fb})
    return pairs


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 3000, lr: float = 0.5) -> np.ndarray:
    """No-intercept logistic regression on standardized features (plain
    gradient descent -- numpy only, no sklearn dependency)."""
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = X / sd
    w = np.zeros(X.shape[1])
    n = len(y)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Xs @ w)))
        grad = Xs.T @ (p - y) / n + l2 * w
        w -= lr * grad
    return w / sd  # back to raw units: score = w_raw . f


def accuracy(pairs: list[dict], score) -> dict:
    """score(f) -> float; a pair is right when score(fa) > score(fb) iff a won."""
    total = correct = 0
    by_pos: dict[str, list[int]] = {}
    by_gap: dict[str, list[int]] = {}
    by_week: dict[str, list[int]] = {}
    for p in pairs:
        sa, sb = score(p["fa"]), score(p["fb"])
        if sa == sb:
            continue  # a tie is not a call either way
        ok = int((sa > sb) == (p["y"] == 1))
        total += 1; correct += ok
        by_pos.setdefault(p["position"], [0, 0]); by_pos[p["position"]][0] += ok; by_pos[p["position"]][1] += 1
        bucket = "<5" if p["gap"] < 5 else "5-10" if p["gap"] < 10 else "10+"
        by_gap.setdefault(bucket, [0, 0]); by_gap[bucket][0] += ok; by_gap[bucket][1] += 1
        wb = "weeks 2-5" if p["week"] <= 5 else "weeks 6-10" if p["week"] <= 10 else "weeks 11-18"
        by_week.setdefault(wb, [0, 0]); by_week[wb][0] += ok; by_week[wb][1] += 1
    return {"accuracy": round(correct / total, 4) if total else None, "scored": total, "undecided": len(pairs) - total,
            "by_position": {k: {"correct": v[0], "scored": v[1], "accuracy": round(v[0] / v[1], 3)} for k, v in sorted(by_pos.items())},
            "by_gap": {k: {"correct": v[0], "scored": v[1], "accuracy": round(v[0] / v[1], 3)} for k, v in by_gap.items()},
            "by_week": {k: {"correct": v[0], "scored": v[1], "accuracy": round(v[0] / v[1], 3)} for k, v in sorted(by_week.items())}}


# The hand-set weights ranking.py used until 2026-09-21 ("simple, documented, not fitted"), frozen here so the
# baseline stays the baseline after ranking.py adopts the fit.
HAND_WEIGHTS_PRE_FIT = {"epa_trend": 2.0, "red_zone_share": 3.0, "target_share": 2.0, "cpoe": 0.05, "implied_total_passers_only": 0.04}


def hand_weight_score(f: list[float]) -> float:
    """The pre-fit ranking.py score, on the same features (ppg terms absent)."""
    d = dict(zip(FEATURES, f)); w = HAND_WEIGHTS_PRE_FIT
    return (w["epa_trend"] * d["epa_trend"] + w["red_zone_share"] * d["red_zone_share"] + w["target_share"] * d["target_share"]
            + (w["cpoe"] * d["cpoe"] + w["implied_total_passers_only"] * d["implied_total"] if d["cpoe"] else 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="fit ranking weights on real history")
    parser.add_argument("--train", type=int, nargs="+", default=[2023])
    parser.add_argument("--test", type=int, nargs="+", default=[2024])
    parser.add_argument("--weeks", type=int, nargs=2, default=[2, 18], help="as-of weeks to use (inclusive)")
    parser.add_argument("--blend", type=float, default=None, help="shrink ppg toward last season with this many pseudo-games (see BLEND_K)")
    parser.add_argument("--with", dest="extra", nargs="*", default=[], choices=CANDIDATE_FEATURES, help="candidate features to add (Step 3)")
    parser.add_argument("--per-position", action="store_true", help="fit one weight vector per position instead of one shared")
    args = parser.parse_args()
    global BLEND_K, FEATURES
    BLEND_K = args.blend
    FEATURES = list(BASE_FEATURES) + list(args.extra)
    weeks = range(args.weeks[0], args.weeks[1] + 1)

    train = [p for s in args.train for p in build_pairs(s, weeks)]
    test = [p for s in args.test for p in build_pairs(s, weeks)]
    X = np.array([p["x"] for p in train]); y = np.array([p["y"] for p in train], dtype=float)
    # symmetrize so the no-intercept model sees both orderings
    X = np.vstack([X, -X]); y = np.concatenate([y, 1 - y])
    w = fit_logistic(X, y)
    weights = {f: round(float(v), 4) for f, v in zip(FEATURES, w)}

    if args.per_position:
        # one vector per position; a pair is always same-position, so score() can key off it
        by_pos: dict[str, np.ndarray] = {}
        for pos in sorted({p["position"] for p in train}):
            sub = [p for p in train if p["position"] == pos]
            Xp = np.array([p["x"] for p in sub]); yp = np.array([p["y"] for p in sub], dtype=float)
            by_pos[pos] = fit_logistic(np.vstack([Xp, -Xp]), np.concatenate([yp, 1 - yp]))
        weights = {pos: {f: round(float(v), 4) for f, v in zip(FEATURES, vec)} for pos, vec in by_pos.items()}
        for p in train + test:
            p["_w"] = by_pos.get(p["position"], w)
        # accuracy() scores fa/fb through one function; stash the pair's vector on the features via a closure per pair
        def acc_per_pos(pairs):
            total = correct = 0; by_gap = {}; by_week = {}; by_p = {}
            for p in pairs:
                sa, sb = float(np.dot(p["_w"], p["fa"])), float(np.dot(p["_w"], p["fb"]))
                if sa == sb: continue
                ok = int((sa > sb) == (p["y"] == 1)); total += 1; correct += ok
                by_p.setdefault(p["position"], [0, 0]); by_p[p["position"]][0] += ok; by_p[p["position"]][1] += 1
                g = "<5" if p["gap"] < 5 else "5-10" if p["gap"] < 10 else "10+"; by_gap.setdefault(g, [0, 0]); by_gap[g][0] += ok; by_gap[g][1] += 1
                wb = "weeks 2-5" if p["week"] <= 5 else "weeks 6-10" if p["week"] <= 10 else "weeks 11-18"; by_week.setdefault(wb, [0, 0]); by_week[wb][0] += ok; by_week[wb][1] += 1
            fmt = lambda d: {k: {"correct": v[0], "scored": v[1], "accuracy": round(v[0] / v[1], 3)} for k, v in d.items()}  # noqa: E731
            return {"accuracy": round(correct / total, 4) if total else None, "scored": total, "undecided": len(pairs) - total,
                    "by_position": fmt(by_p), "by_gap": fmt(by_gap), "by_week": fmt(by_week)}
        report = {"date": date.today().isoformat(), "train_seasons": args.train, "test_seasons": args.test, "per_position": True,
                  "weeks": [args.weeks[0], args.weeks[1]], "features": FEATURES, "blend_k": BLEND_K, "train_pairs": len(train), "test_pairs": len(test),
                  "weights": weights, "test": {"fitted_per_position": acc_per_pos(test), "fitted_shared": accuracy(test, lambda f: float(np.dot(w, f)))},
                  "train_fit": {"fitted_per_position": acc_per_pos(train)}}
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{report['date']}_ranking_fit_per_position.json"
        out.write_text(json.dumps(report, indent=2)); print(json.dumps({k: v for k, v in report.items() if k != "features"}, indent=2)); print(f"wrote {out}")
        return

    fitted = lambda f: float(np.dot(w, f))  # noqa: E731
    ppg_only = lambda f: f[FEATURES.index("ppg")]  # noqa: E731
    prior_only = lambda f: f[FEATURES.index("ppg_prior")]  # noqa: E731
    report = {
        "date": date.today().isoformat(), "train_seasons": args.train, "test_seasons": args.test,
        "weeks": [args.weeks[0], args.weeks[1]], "features": FEATURES, "fit_scoring": FIT_SCORING,
        "min_points_both": MIN_POINTS_BOTH, "pairs_per_group": PAIRS_PER_GROUP, "blend_k": BLEND_K,
        "train_pairs": len(train), "test_pairs": len(test), "weights": weights,
        "test": {"fitted": accuracy(test, fitted), "hand_weights_pre_fit": accuracy(test, hand_weight_score),
                 "ppg_only": accuracy(test, ppg_only), "prior_season_ppg_only": accuracy(test, prior_only)},
        "train_fit": {"fitted": accuracy(train, fitted)},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = ('' if BLEND_K is None else f'_blend{BLEND_K:g}') + (('_with_' + '_'.join(args.extra)) if args.extra else '')
    out = RESULTS_DIR / f"{report['date']}_ranking_fit{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "features"}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
