"""Retrospective lineup eval: what the product's start/sit ranking would have
started for every real roster in an ingested league, week by week, against
what the manager actually started and what hindsight says was best
(2026-10-14, after the Ferguson-over-Likely miss).

This is the fourth eval type, distinct from the three in CLAUDE.md: not
retrieval, not synthetic dilemmas from ground_truth.jsonl, but the real
rosters real people set this season, scored by Sleeper's own `players_points`
(the league's scoring, kickers and defenses included). It answers "are we
good?" for the season so far: for every (league, roster, week) it builds the
lineup `report.py`'s start_sit would have recommended with only the data
available before that week's games, and compares three totals --

    manager : the starters Sleeper recorded (locked lineups, final points)
    ours    : the ranking's lineup, filled the way _start_sit_report fills it
    optimal : the best lineup in hindsight

-- plus every individual start/sit call the ranking made (a recommended
starter vs. each benched alternative at the same slot), whether it was right,
and how confident it said it was.

As-of discipline (CLAUDE.md's non-negotiable rule): the signals table is the
one computed as of that week, points per game count only weeks strictly
before it, and the injury report is that week's official NFL report (Out =
unavailable, Questionable/Doubtful = discounted, exactly as the product does).
Two honest gaps vs. the live product: Sleeper's own injury designation and
depth chart are snapshots the ingest overwrites each cycle, so neither is
available for a past week -- a player Sleeper listed Out or IR who is absent
from that week's NFL report is a candidate here (he usually ranks last on no
current-season data), and the backup-QB rule cannot fire. The lineup-filling
is a mirror of report._start_sit_report (dedicated slots first, flex from the
leftovers, rank_candidates each time) rather than a call to it, because that
function reads the CURRENT roster through get_my_roster; the ranking itself
(ranking.rank_candidates over SignalTables.row_for) is the real one.

Inputs (all fetched by the caller, nothing here hits the network except the
bye-week map):
  --sleeper-dir   <dir>/<league_id>/{league.json, matchups_week_N.json} as
                  Sleeper returns them (src/ingest/sleeper.fetch_*), fetched
                  AFTER the weeks finished so starters/points are final
  --players-json  a Sleeper players table ({"data": {...}} as the ingest
                  writes it, or the raw dict) for names/positions
  --processed-dir a data/processed layout: signals/, player_stats/,
                  unit_stats/, injuries/ with every week's table
  --weeks 1-5

Usage:
  python -m evals.retrospective_lineups --sleeper-dir /tmp/sleeper \
      --players-json data/raw/leagues/<id>/sleeper/players.json \
      --processed-dir data/processed --weeks 1-5 \
      --out evals/results/2026-10-14_retrospective_lineups.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.ingest import nflverse
from src.rag import lookup, player_index
from src.reasoning import ranking
from src.reasoning.ranking import SignalTables, rank_candidates

UNAVAILABLE_REPORT_STATUSES = {"Out"}
SKILL = ("QB", "RB", "WR", "TE", "K")


def parse_weeks(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(w) for w in spec.split(",")]


def load_json(path: Path):
    data = json.loads(path.read_text())
    return data["data"] if isinstance(data, dict) and "data" in data and "fetched_at" in data else data


class Resolver:
    """Sleeper id -> (nflverse id, name, position, team), by the product's own
    name resolution (player_index.resolve_player); DEF units by team."""

    def __init__(self, players: dict, season: int):
        self.players = players
        self.idx = player_index.build_player_index(season)
        self.cache: dict[str, dict | None] = {}
        self.unresolved: Counter = Counter()

    def resolve(self, sleeper_id: str) -> dict | None:
        if sleeper_id in self.cache:
            return self.cache[sleeper_id]
        info = self.players.get(sleeper_id) or {}
        out = None
        if sleeper_id.isalpha():  # a team defense: "TB"
            out = {"player_id": sleeper_id, "name": f"{sleeper_id} D/ST", "position": "DEF", "team": sleeper_id}
        elif info.get("position") in SKILL and info.get("full_name"):
            res = player_index.resolve_player(info["full_name"], self.idx)
            cands = res.candidates
            if res.match_type == "ambiguous" and info.get("team"):
                cands = [c for c in cands if c.team == info["team"]]
            if len(cands) == 1:
                c = cands[0]
                out = {"player_id": c.player_id, "name": c.player_name, "position": info["position"], "team": info.get("team") or c.team}
            else:
                self.unresolved[f"{info['full_name']} ({info['position']}, {res.match_type})"] += 1
        self.cache[sleeper_id] = out
        return out


def slot_counts(roster_positions: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for slot in lookup.starter_slots(roster_positions):
        counts[slot] = counts.get(slot, 0) + 1
    return counts


def fill_by_ranking(candidates: list[dict], slots: dict[str, int]) -> tuple[list[dict], list[dict], int]:
    """Mirror of report._start_sit_report's slot filling. Returns
    (started candidates, decisions, open_slots). A decision is one
    comparison the ranking made: a starter vs. an alternative it benched at
    the same slot group, with the adjacent-pair confidence when the two are
    adjacent in the ranking."""
    by_position: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_position[c["position"]].append(c)
    flex_slots = [(s, n) for s, n in slots.items() if s in lookup.FLEX_ELIGIBILITY]
    flex_eligible = {p for s, _ in flex_slots for p in lookup.FLEX_ELIGIBILITY[s]}
    by_id = {c["player_id"]: c for c in candidates}
    started: list[dict] = []
    decisions: list[dict] = []
    flex_pool: list[dict] = []
    open_slots = 0

    def _take(group: str, pool: list[dict], n: int) -> list[dict]:
        if not pool:
            return []
        r = rank_candidates(pool)
        ranked = r["ranked"]
        starters, alts = ranked[:n], ranked[n:]
        for i, s in enumerate(starters):
            for j, a in enumerate(alts, start=len(starters)):
                decisions.append({
                    "slot": group, "starter": s["name"], "starter_id": s["player_id"], "alternative": a["name"],
                    "alternative_id": a["player_id"], "starter_score": s["opportunity_score"], "alt_score": a["opportunity_score"],
                    "adjacent": (j == i + 1),
                    "win_probability": s["win_probability_vs_next"] if j == i + 1 else None,
                    "confidence_label": s["confidence_label"] if j == i + 1 else None,
                    "starter_stale": s["stale"], "alt_stale": a["stale"],
                    "starter_play_probability": s.get("play_probability"), "alt_play_probability": a.get("play_probability"),
                })
        return [by_id[s["player_id"]] for s in starters], [by_id[a["player_id"]] for a in alts]

    for position in sorted(by_position):
        n = slots.get(position, 0)
        pool = by_position[position]
        if n == 0:
            if position in flex_eligible:
                flex_pool.extend(pool)
            continue
        starters, alts = _take(position, pool, n)
        started.extend(starters)
        open_slots += max(0, n - len(starters))
        if position in flex_eligible:
            flex_pool.extend(alts)
    # Most restrictive flex first (REC_FLEX/WRRB_FLEX before FLEX before SUPER_FLEX), as nested
    # eligibility makes that the right order; the report iterates roster order, same result in practice.
    for slot, n in sorted(flex_slots, key=lambda sn: len(lookup.FLEX_ELIGIBILITY[sn[0]])):
        pool = [c for c in flex_pool if c["position"] in lookup.FLEX_ELIGIBILITY[slot]]
        starters, _ = _take(slot, pool, n)
        chosen = {s["player_id"] for s in starters}
        flex_pool = [c for c in flex_pool if c["player_id"] not in chosen]
        started.extend(starters)
        open_slots += max(0, n - len(starters))
    return started, decisions, open_slots


def optimal_points(candidates: list[dict], slots: dict[str, int]) -> float:
    """Hindsight-best lineup: dedicated slots take the top scorers at each
    position, then flex slots (most restrictive first) take the best eligible
    leftovers -- exact for nested flex eligibility, which is every Sleeper
    flex type here."""
    by_position: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_position[c["position"]].append(c)
    total = 0.0
    leftovers: list[dict] = []
    for position, pool in by_position.items():
        pool = sorted(pool, key=lambda c: c["actual"], reverse=True)
        n = slots.get(position, 0)
        total += sum(c["actual"] for c in pool[:n])
        leftovers.extend(pool[n:])
    flex_slots = sorted(((s, n) for s, n in slots.items() if s in lookup.FLEX_ELIGIBILITY),
                        key=lambda sn: len(lookup.FLEX_ELIGIBILITY[sn[0]]))
    for slot, n in flex_slots:
        eligible = sorted((c for c in leftovers if c["position"] in lookup.FLEX_ELIGIBILITY[slot]),
                          key=lambda c: c["actual"], reverse=True)
        take = eligible[:n]
        total += sum(c["actual"] for c in take)
        taken = {c["player_id"] for c in take}
        leftovers = [c for c in leftovers if c["player_id"] not in taken]
    return round(total, 2)


def run(sleeper_dir: Path, players_json: Path, processed_dir: Path, weeks: list[int], season: int = 2026) -> dict:
    players = load_json(players_json)
    resolver = Resolver(players, season)
    byes = nflverse.bye_weeks(season)
    rosters_out: list[dict] = []
    decisions_out: list[dict] = []
    leagues_out = []
    for league_dir in sorted(p for p in sleeper_dir.iterdir() if p.is_dir()):
        league = load_json(league_dir / "league.json")
        scoring = league["scoring_settings"]
        slots = slot_counts(league["roster_positions"])
        leagues_out.append({"league_id": league["league_id"], "name": league["name"], "slots": slots})
        for week in weeks:
            path = league_dir / f"matchups_week_{week}.json"
            if not path.exists():
                continue
            tables = SignalTables.load(processed_dir / "signals", season, week, scoring,
                                       stats_dir=processed_dir / "player_stats", injuries_dir=processed_dir / "injuries")
            for team in load_json(path):
                pts = team.get("players_points") or {}
                cands, skipped = [], []
                for sid in team.get("players") or []:
                    r = resolver.resolve(sid)
                    actual = float(pts.get(sid) or 0.0)
                    if r is None:
                        skipped.append({"sleeper_id": sid, "actual": actual, "name": (players.get(sid) or {}).get("full_name")})
                        continue
                    row = tables.row_for(r["player_id"])
                    avail = tables.availability.get(r["player_id"]) or {}
                    if avail.get("report_status") in UNAVAILABLE_REPORT_STATUSES:
                        skipped.append({**r, "actual": actual, "why": "Out on the NFL report"})
                        continue
                    if ranking.on_bye(row, season, week, r["team"], byes):
                        skipped.append({**r, "actual": actual, "why": "bye"})
                        continue
                    cands.append({**r, "row": row, "injury_status": None, "depth_chart_order": None, "sleeper_id": sid, "actual": actual})
                started, decisions, open_slots = fill_by_ranking(cands, slots)
                ours = round(sum(c["actual"] for c in started), 2)
                manager_ids = [s for s in (team.get("starters") or []) if s not in (None, "", "0")]
                manager = round(sum(float(pts.get(s) or 0.0) for s in manager_ids), 2)
                all_for_opt = cands + [s for s in skipped if "player_id" in s]  # hindsight may start anyone who scored
                optimal = optimal_points(all_for_opt, slots)
                actual_by_id = {c["player_id"]: c["actual"] for c in cands}
                started_ids = {c["player_id"] for c in started}
                manager_gsis = {resolver.cache.get(s, {}) and resolver.cache[s]["player_id"] for s in manager_ids if resolver.cache.get(s)}
                for d in decisions:
                    d.update({
                        "league": league["name"], "week": week, "roster_id": team["roster_id"],
                        "starter_points": actual_by_id.get(d["starter_id"]), "alt_points": actual_by_id.get(d["alternative_id"]),
                        "manager_started_starter": d["starter_id"] in manager_gsis,
                        "manager_started_alt": d["alternative_id"] in manager_gsis,
                    })
                    d["correct"] = (d["starter_points"] or 0) >= (d["alt_points"] or 0)
                    d["margin"] = round((d["starter_points"] or 0) - (d["alt_points"] or 0), 2)
                decisions_out.extend(decisions)
                rosters_out.append({
                    "league": league["name"], "league_id": league["league_id"], "week": week, "roster_id": team["roster_id"],
                    "manager_points": manager, "our_points": ours, "optimal_points": optimal,
                    "manager_slots_filled": len(manager_ids), "our_slots_filled": len(started), "open_slots": open_slots,
                    "stale_starters": sum(1 for c in started if c["row"] and c["row"].get("stale")),
                    "n_candidates": len(cands), "skipped": [{k: v for k, v in s.items() if k != "row"} for s in skipped],
                    "our_lineup": [{"name": c["name"], "position": c["position"], "actual": c["actual"]} for c in started],
                    "manager_lineup": [{"name": (resolver.cache.get(s) or {}).get("name") or (players.get(s) or {}).get("full_name") or s,
                                        "actual": float(pts.get(s) or 0.0)} for s in manager_ids],
                    "our_minus_manager": round(ours - manager, 2),
                    "disagreements": sorted({c["name"] for c in started if c["player_id"] not in manager_gsis}),
                })
    return {"leagues": leagues_out, "rosters": rosters_out, "decisions": decisions_out,
            "unresolved_players": dict(resolver.unresolved)}


def summarize(result: dict) -> dict:
    rosters, decisions = result["rosters"], result["decisions"]

    def _block(rs: list[dict], ds: list[dict]) -> dict:
        n = len(rs)
        if not n:
            return {}
        ours_better = sum(1 for r in rs if r["our_points"] > r["manager_points"])
        ties = sum(1 for r in rs if r["our_points"] == r["manager_points"])
        return {
            "rosters": n,
            "manager_avg": round(sum(r["manager_points"] for r in rs) / n, 2),
            "ours_avg": round(sum(r["our_points"] for r in rs) / n, 2),
            "optimal_avg": round(sum(r["optimal_points"] for r in rs) / n, 2),
            "ours_minus_manager_avg": round(sum(r["our_minus_manager"] for r in rs) / n, 2),
            "ours_beat_manager": ours_better, "ties": ties, "manager_beat_ours": n - ours_better - ties,
            "manager_share_of_optimal": round(sum(r["manager_points"] for r in rs) / max(1e-9, sum(r["optimal_points"] for r in rs)), 3),
            "ours_share_of_optimal": round(sum(r["our_points"] for r in rs) / max(1e-9, sum(r["optimal_points"] for r in rs)), 3),
            "decisions": len(ds),
            "decision_accuracy": round(sum(d["correct"] for d in ds) / len(ds), 3) if ds else None,
            "adjacent_decisions": sum(1 for d in ds if d["adjacent"]),
            "adjacent_accuracy": round(sum(d["correct"] for d in ds if d["adjacent"]) / max(1, sum(1 for d in ds if d["adjacent"])), 3),
        }

    out = {"overall": _block(rosters, decisions), "by_week": {}, "by_league": {}, "by_confidence_label": {}, "by_slot": {},
           "manager_disagreed": {}, "designated_starters": {}}
    for wk in sorted({r["week"] for r in rosters}):
        out["by_week"][wk] = _block([r for r in rosters if r["week"] == wk], [d for d in decisions if d["week"] == wk])
    for lg in sorted({r["league"] for r in rosters}):
        out["by_league"][lg] = _block([r for r in rosters if r["league"] == lg], [d for d in decisions if d["league"] == lg])
    adj = [d for d in decisions if d["adjacent"] and d["win_probability"] is not None]
    for label in ("coin flip", "lean", "clear", "strong"):
        ds = [d for d in adj if d["confidence_label"] == label]
        if ds:
            out["by_confidence_label"][label] = {
                "n": len(ds), "accuracy": round(sum(d["correct"] for d in ds) / len(ds), 3),
                "mean_stated_probability": round(sum(d["win_probability"] for d in ds) / len(ds), 3),
            }
    for slot in sorted({d["slot"] for d in decisions}):
        ds = [d for d in decisions if d["slot"] == slot]
        out["by_slot"][slot] = {"n": len(ds), "accuracy": round(sum(d["correct"] for d in ds) / len(ds), 3)}
    # Where the manager made the opposite call (started our alternative, benched our starter): who was right?
    dis = [d for d in decisions if d["manager_started_alt"] and not d["manager_started_starter"]]
    out["manager_disagreed"] = {"n": len(dis), "ranking_was_right": sum(d["correct"] for d in dis),
                                "manager_was_right": sum(1 for d in dis if not d["correct"])}
    desig = [d for d in decisions if d["starter_play_probability"] is not None or d["alt_play_probability"] is not None]
    out["designated_starters"] = {
        "n": len(desig),
        "ranking_started_designated": sum(1 for d in desig if d["starter_play_probability"] is not None),
        "ranking_benched_designated": sum(1 for d in desig if d["alt_play_probability"] is not None and d["starter_play_probability"] is None),
        "benched_designated_outscored_starter": sum(1 for d in desig if d["alt_play_probability"] is not None
                                                    and d["starter_play_probability"] is None and not d["correct"]),
    }
    out["worst_calls"] = sorted(decisions, key=lambda d: d["margin"])[:15]
    out["unresolved_players"] = result["unresolved_players"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sleeper-dir", type=Path, required=True)
    ap.add_argument("--players-json", type=Path, required=True)
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--weeks", default="1-5")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    result = run(args.sleeper_dir, args.players_json, args.processed_dir, parse_weeks(args.weeks), args.season)
    summary = summarize(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, **result}, indent=1, default=str))
        print("wrote", args.out)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("worst_calls",)}, indent=1, default=str))
    print("\nWorst calls (our starter vs. the alternative we benched, by points margin):")
    for d in summary["worst_calls"]:
        print(f"  {d['league'][:12]:12s} wk{d['week']} r{d['roster_id']:<2} {d['slot']:10s} started {d['starter']} ({d['starter_points']}) "
              f"over {d['alternative']} ({d['alt_points']}) margin {d['margin']} conf={d['win_probability']} "
              f"{'[manager disagreed]' if d['manager_started_alt'] and not d['manager_started_starter'] else ''}")


if __name__ == "__main__":
    main()
