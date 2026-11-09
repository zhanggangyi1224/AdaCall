"""Run AdaCall's paired experiment on the adacall_od study.

For every (topology, replication):
  1. control arm, cold run, namespace ``<topo>/r<rep>/ctrl``;
  2. one paired treatment branch per reach: SocioVerse2 replays steps < t* from the control run
     (WarmStartSpec), then from t* each agent reuses the control reply when its prompt is
     identical, so only diverged agents are paid;
  3. for the reaches in --indep-reaches, an independent treatment arm (cold run, own
     namespace), the conventional baseline.

All replies go through one PairedCache (``<out>/cache.jsonl``); a rerun with the same cache makes
no API calls and reproduces every number. Results: ``<out>/summary.json`` and ``<out>/runs.csv``.

Usage:
  python run_od.py --llm-kind standin --out runs/standin                  # free
  python run_od.py --llm-kind openai --model deepseek-flash --out runs/llm
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

import studies.adacall_od as od  # noqa: E402
from cache import PairedCache  # noqa: E402
from socioverse.engine import build_simulator  # noqa: E402
from socioverse.schemas.simulation import WarmStartSpec  # noqa: E402


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def final_metrics(db: Path) -> dict:
    con = duckdb.connect(str(db), read_only=True)
    try:
        row = con.execute("SELECT * FROM metrics ORDER BY step DESC LIMIT 1").fetchdf().iloc[0]
        sources = dict(con.execute("SELECT action_source, count(*) FROM panel "
                                   "WHERE action_source IS NOT NULL GROUP BY 1").fetchall())
    finally:
        con.close()
    return {"mean_opinion": float(row["mean_opinion"]), "opinion_std": float(row["opinion_std"]),
            "frac_above_0_5": float(row["frac_above_0_5"]), "sources": sources}


def _no_calls(prompt: str, **ctx) -> str:
    raise RuntimeError(f"cache miss in --replay-only mode: {ctx.get('ns')} "
                       f"{ctx.get('agent_id')} step {ctx.get('step')}")


def run_arm(cache: PairedCache, args, *, topology: str, reach: float | None, ns: str,
            reuse_ns: str | None, db: Path, warm_from: Path | None, eps: float = 0.0,
            audit: float = 0.0, backend=None) -> dict:
    decision_args = {"llm_kind": args.llm_kind, "model": args.model,
                     "temperature": args.temperature, "max_tokens": args.max_tokens,
                     "max_workers": args.workers, "rpm": args.rpm,
                     "prompt_variant": getattr(args, "prompt_variant", "base"),
                     "exposure": getattr(args, "exposure", "both")}
    _, env, pop, sim = od.make_bundles(n_agents=args.n_agents, topology=topology,
                                       n_steps=args.steps, t_star=args.t_star, reach=reach,
                                       campaign=getattr(args, "campaign", od.bundles.CAMPAIGN),
                                       decision_args=decision_args)
    if warm_from is not None:
        sim.warm_start = WarmStartSpec(source_version=ns.rsplit("/", 1)[0] + "/ctrl",
                                       source_trajectory=str(warm_from),
                                       resume_from=args.t_star - 1)
    db.parent.mkdir(parents=True, exist_ok=True)
    s = build_simulator(env_bundle=env, pop_bundle=pop, sim_config=sim, store_path=db)
    if backend is None:
        backend = _no_calls if args.replay_only else s.decision.backend
    s.decision.client = cache.client(backend, ns=ns, reuse_ns=reuse_ns, eps=eps, audit=audit)
    paid0, reused0 = cache.paid[ns], cache.reused[ns]
    s.run()
    out = final_metrics(db)
    out.update(ns=ns, paid=cache.paid[ns] - paid0, reused=cache.reused[ns] - reused0)
    return out


def summarize(rows: list[dict], n_agents: int, steps: int, t_star: int) -> list[dict]:
    post = n_agents * (steps - t_star + 1)          # treatment decisions after the fork
    full = n_agents * steps                         # what an independent treatment arm pays
    cells = {}
    for r in rows:
        if r["arm"] == "ctrl":
            continue
        cells.setdefault((r["topology"], r["reach"], r["arm"]), []).append(r)
    out = []
    for (topo, reach, arm), rs in sorted(cells.items()):
        d = [r["delta"] for r in rs]
        out.append({"topology": topo, "reach": reach, "arm": arm, "n_reps": len(rs),
                    "delta_mean": round(st.mean(d), 5),
                    "delta_sd": round(st.stdev(d), 5) if len(d) > 1 else None,
                    "sign_positive_share": round(sum(x > 0 for x in d) / len(d), 3),
                    "paid_vs_postfork": round(st.mean(r["paid"] for r in rs) / post, 4),
                    "paid_vs_independent": round(st.mean(r["paid"] for r in rs) / full, 4)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/standin")
    ap.add_argument("--llm-kind", choices=["standin", "openai"], default="standin")
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--rpm", type=int, default=None, help="cap on API calls per minute")
    ap.add_argument("--n-agents", type=int, default=100)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--t-star", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--topologies", nargs="+", default=["ring", "small_world"])
    ap.add_argument("--reaches", nargs="+", type=float, default=[0.01, 0.05, 1.0])
    ap.add_argument("--indep-reaches", nargs="+", type=float, default=[0.01, 0.05])
    ap.add_argument("--replay-only", action="store_true",
                    help="serve every reply from the cache and fail on a miss (no API calls)")
    args = ap.parse_args()
    load_env(SV2 / ".env")
    if args.llm_kind == "openai":
        args.model = args.model or os.environ.get("SV_LLM_MODEL", "deepseek-flash")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = PairedCache(out / "cache.jsonl", model=args.model or args.llm_kind,
                        temperature=args.temperature, max_tokens=args.max_tokens)
    rows, t0 = [], time.time()
    for topo in args.topologies:
        for rep in range(args.reps):
            base = f"{topo}/r{rep}"
            ctrl_db = out / topo / f"r{rep}" / "ctrl" / "study.duckdb"
            c = run_arm(cache, args, topology=topo, reach=None, ns=f"{base}/ctrl",
                        reuse_ns=None, db=ctrl_db, warm_from=None)
            rows.append({"topology": topo, "rep": rep, "reach": None, "arm": "ctrl", "delta": 0.0, **c})
            for reach in args.reaches:
                arms = [("paired", f"{base}/paired{reach}", f"{base}/ctrl", ctrl_db)]
                if reach in args.indep_reaches:
                    arms.append(("indep", f"{base}/indep{reach}", None, None))
                for arm, ns, reuse, warm in arms:
                    db = out / topo / f"r{rep}" / f"{arm}{reach}" / "study.duckdb"
                    r = run_arm(cache, args, topology=topo, reach=reach, ns=ns, reuse_ns=reuse,
                                db=db, warm_from=warm)
                    r["delta"] = round(r["mean_opinion"] - c["mean_opinion"], 5)
                    rows.append({"topology": topo, "rep": rep, "reach": reach, "arm": arm, **r})
                    print(f"[{time.time() - t0:6.0f}s] {topo:11s} r{rep} reach={reach:<5} {arm:6s} "
                          f"delta={r['delta']:+.4f} paid={r['paid']:4d} reused={r['reused']:4d} "
                          f"api_calls={cache.api_calls}", flush=True)
    summary = {
        "config": {k: v for k, v in vars(args).items()},
        "api_calls_this_process": cache.api_calls,
        "total_paid_calls": sum(r["paid"] for r in rows),
        "wall_seconds": round(time.time() - t0, 1),
        "cells": summarize(rows, args.n_agents, args.steps, args.t_star),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "runs.csv").open("w", newline="") as f:
        fields = ["topology", "rep", "reach", "arm", "delta", "mean_opinion", "opinion_std",
                  "frac_above_0_5", "paid", "reused", "ns"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    for c in summary["cells"]:
        print(c)
    print(f"api calls this process: {cache.api_calls}; total paid calls: {summary['total_paid_calls']}")


if __name__ == "__main__":
    main()
