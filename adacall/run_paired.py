"""Run AdaCall's paired experiment on the adacall_germany or adacall_campus study.

  germany: no spillover. EV purchase bonus for one federal state (Bayern) or nationwide (all);
           outcomes: share of buyers choosing an all-electric brand, EV index of chosen brands.
  campus:  group spillover. Canteen subsidy for k dorms (e.g. 5) or campus-wide (all);
           outcomes: canteen share, mean satisfaction.

For every replication:
  1. control arm, cold run, namespace ``r<rep>/ctrl``;
  2. one paired treatment branch per target: SocioVerse2 replays steps < t* from the control run,
     then each agent reuses the control reply when its prompt is identical;
  3. for the targets in --indep-targets, an independent treatment arm (cold run, own namespace).

Outcomes are taken at the last step and averaged over the post-fork steps. Replies go through one
PairedCache (``<out>/cache.jsonl``); a rerun with the same cache makes no API calls. Results:
``<out>/summary.json`` and ``<out>/runs.csv``; ``cjk_replies`` counts replies containing CJK text.

Usage:
  python run_paired.py --study germany --llm-kind standin --out runs/de_standin     # free
  python run_paired.py --study campus --llm-kind openai --model deepseek-flash --out runs/cd_llm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SV2 = HERE.parent / "SocioVerse2"
sys.path.insert(0, str(SV2))
sys.path.insert(0, str(HERE))

import duckdb  # noqa: E402

import studies.adacall_campus as cd  # noqa: E402
import studies.adacall_germany as de  # noqa: E402
from cache import PairedCache  # noqa: E402
from run_od import _no_calls, load_env  # noqa: E402
from socioverse.engine import build_simulator  # noqa: E402
from socioverse.schemas.simulation import WarmStartSpec  # noqa: E402
from studies.adacall_germany.model import is_cjk  # noqa: E402

# study -> bundle factory, outcomes, defaults (steps, t*, targets, independent targets)
STUDIES = {
    "germany": {"module": de, "outcomes": ["share_pure_ev", "ev_index"], "steps": 10, "t_star": 3,
                "targets": ["Bayern", "all"], "indep": ["Bayern"]},
    "campus": {"module": cd, "outcomes": ["share_canteen", "mean_satisfaction"], "steps": 6,
               "t_star": 2, "targets": ["5", "all"], "indep": ["5"]},
}
OUTCOMES: list[str] = []


def _target(study: str, target: str | None):
    if target is None or study != "campus" or target == "all":
        return target
    return int(target)


def outcomes(db: Path, t_star: int) -> dict:
    con = duckdb.connect(str(db), read_only=True)
    try:
        df = con.execute("SELECT * FROM metrics ORDER BY step").fetchdf()
        sources = dict(con.execute("SELECT action_source, count(*) FROM panel "
                                   "WHERE action_source IS NOT NULL GROUP BY 1").fetchall())
    finally:
        con.close()
    post = df[df["step"] >= t_star]
    out = {"sources": sources}
    for m in OUTCOMES:
        out[m] = float(df[m].iloc[-1])
        out[f"{m}_post"] = round(float(post[m].mean()), 5)
    return out


def run_arm(cache: PairedCache, args, *, target, ns: str, reuse_ns: str | None, db: Path,
            warm_from: Path | None) -> dict:
    decision_args = {"llm_kind": args.llm_kind, "model": args.model,
                     "temperature": args.temperature, "max_tokens": args.max_tokens,
                     "max_workers": args.workers, "rpm": args.rpm}
    kw = ({"month_offset": args.month_offset} if args.study == "germany"
          else {"cohort_round": args.cohort_round})
    _, env, pop, sim = STUDIES[args.study]["module"].make_bundles(
        n_agents=args.n_agents, n_steps=args.steps, t_star=args.t_star,
        target=_target(args.study, target), decision_args=decision_args, **kw)
    if warm_from is not None:
        sim.warm_start = WarmStartSpec(source_version=ns.rsplit("/", 1)[0] + "/ctrl",
                                       source_trajectory=str(warm_from),
                                       resume_from=args.t_star - 1)
    db.parent.mkdir(parents=True, exist_ok=True)
    s = build_simulator(env_bundle=env, pop_bundle=pop, sim_config=sim, store_path=db)
    backend = _no_calls if args.replay_only else s.decision.backend
    s.decision.client = cache.client(backend, ns=ns, reuse_ns=reuse_ns)
    paid0, reused0 = cache.paid[ns], cache.reused[ns]
    s.run()
    out = outcomes(db, args.t_star)
    out.update(ns=ns, paid=cache.paid[ns] - paid0, reused=cache.reused[ns] - reused0,
               n_targeted=s.env.snapshot().get("n_exposed", s.env.snapshot().get("n_subsidised")))
    return out


def summarize(rows: list[dict], n_agents: int, steps: int, t_star: int) -> list[dict]:
    post, full = n_agents * (steps - t_star + 1), n_agents * steps
    cells: dict = {}
    for r in rows:
        if r["arm"] != "ctrl":
            cells.setdefault((r["target"], r["arm"]), []).append(r)
    out = []
    for (target, arm), rs in sorted(cells.items()):
        cell = {"target": target, "arm": arm, "n_reps": len(rs),
                "n_targeted": rs[0]["n_targeted"],
                "paid_vs_postfork": round(st.mean(r["paid"] for r in rs) / post, 4),
                "paid_vs_independent": round(st.mean(r["paid"] for r in rs) / full, 4)}
        for m in OUTCOMES:
            for k in (f"delta_{m}", f"delta_{m}_post"):
                d = [r[k] for r in rs]
                cell[f"{k}_mean"] = round(st.mean(d), 5)
                cell[f"{k}_sd"] = round(st.stdev(d), 5) if len(d) > 1 else None
        out.append(cell)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=sorted(STUDIES), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm-kind", choices=["standin", "openai"], default="standin")
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--rpm", type=int, default=None, help="cap on API calls per minute")
    ap.add_argument("--n-agents", type=int, default=100)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--t-star", type=int, default=None)
    ap.add_argument("--month-offset", type=int, default=12,
                    help="germany: step 0 = this month (0 = 2024-01)")
    ap.add_argument("--cohort-round", type=float, default=0.1,
                    help="campus: rounding step of the archetype shares shown in prompts")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--targets", nargs="+", default=None)
    ap.add_argument("--indep-targets", nargs="*", default=None)
    ap.add_argument("--replay-only", action="store_true",
                    help="serve every reply from the cache and fail on a miss (no API calls)")
    args = ap.parse_args()
    spec = STUDIES[args.study]
    for k, d in (("steps", "steps"), ("t_star", "t_star"), ("targets", "targets"),
                 ("indep_targets", "indep")):
        if getattr(args, k) is None:
            setattr(args, k, spec[d])
    OUTCOMES[:] = spec["outcomes"]
    load_env(SV2 / ".env")
    if args.llm_kind == "openai":
        args.model = args.model or os.environ.get("SV_LLM_MODEL", "deepseek-flash")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = PairedCache(out / "cache.jsonl", model=args.model or args.llm_kind,
                        temperature=args.temperature, max_tokens=args.max_tokens)
    rows, t0 = [], time.time()
    for rep in range(args.reps):
        base = f"r{rep}"
        ctrl_db = out / base / "ctrl" / "study.duckdb"
        c = run_arm(cache, args, target=None, ns=f"{base}/ctrl", reuse_ns=None, db=ctrl_db,
                    warm_from=None)
        rows.append({"rep": rep, "target": None, "arm": "ctrl", **c})
        for target in args.targets:
            arms = [("paired", f"{base}/paired_{target}", f"{base}/ctrl", ctrl_db)]
            if target in args.indep_targets:
                arms.append(("indep", f"{base}/indep_{target}", None, None))
            for arm, ns, reuse, warm in arms:
                db = out / base / f"{arm}_{target}" / "study.duckdb"
                r = run_arm(cache, args, target=target, ns=ns, reuse_ns=reuse, db=db,
                            warm_from=warm)
                for m in OUTCOMES:
                    r[f"delta_{m}"] = round(r[m] - c[m], 5)
                    r[f"delta_{m}_post"] = round(r[f"{m}_post"] - c[f"{m}_post"], 5)
                rows.append({"rep": rep, "target": target, "arm": arm, **r})
                o = OUTCOMES[0]
                print(f"[{time.time() - t0:6.0f}s] r{rep} {target:6s} {arm:6s} "
                      f"d_{o}={r[f'delta_{o}']:+.3f} d_{o}_post={r[f'delta_{o}_post']:+.4f} "
                      f"paid={r['paid']:4d} "
                      f"reused={r['reused']:4d} api_calls={cache.api_calls}", flush=True)
    summary = {
        "config": dict(vars(args)),
        "api_calls_this_process": cache.api_calls,
        "total_paid_calls": sum(r["paid"] for r in rows),
        "cjk_replies": sum(is_cjk(v) for v in cache.entries.values()),
        "wall_seconds": round(time.time() - t0, 1),
        "cells": summarize(rows, args.n_agents, args.steps, args.t_star),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    fields = ["rep", "target", "arm", "n_targeted", "paid", "reused", "ns"] + [
        f"{p}{m}{s}" for m in OUTCOMES for p in ("", "delta_") for s in ("", "_post")]
    with (out / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    for cell in summary["cells"]:
        print(cell)
    print(f"api calls this process: {cache.api_calls}; total paid calls: "
          f"{summary['total_paid_calls']}; CJK replies: {summary['cjk_replies']}")


if __name__ == "__main__":
    main()
