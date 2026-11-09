"""C3 conclusion-reproducibility audit of the opinion-diffusion result (reach 5%).

The audited conclusion is the paired estimate of the campaign effect in ``--source`` (default the
20-replication DeepSeek run). Each check reruns the paired design under one change and asks
whether the conclusion survives:
  placebo     the campaign message is replaced by an unrelated one (negative control); the
              effect should vanish. Controls are replayed from the source, so only the treated
              agents and their neighbours are paid.
  temp0       temperature 0 instead of 0.7 (new control and paired arms).
  paraphrase  the prompt is reworded and reordered with the same information (new arms).
A check passes when its 95% CI excludes 0 with the source sign (the placebo: when it includes 0).

Decomposition (no pass/fail; reuses the source controls like the placebo). A treated agent's
prompt carries two channels, media pressure 1 and the message line:
  pressure_only        media pressure 1, no message line
  message_only         the campaign message, media pressure 0
  placebo_message_only the placebo message, media pressure 0

Usage: python run_audit.py --llm-kind openai --model deepseek-flash --out runs/audit_llm
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scipy import stats  # noqa: E402

from cache import PairedCache  # noqa: E402
from run_od import SV2, load_env, run_arm  # noqa: E402
from studies.adacall_od.bundles import CAMPAIGN, PLACEBO  # noqa: E402

CHECKS = {  # name -> (changes to the run arguments, reuse the source controls?)
    "placebo": ({"campaign": PLACEBO}, True),
    "temp0": ({"temperature": 0.0}, False),
    "paraphrase": ({"prompt_variant": "paraphrase"}, False),
    "pressure_only": ({"exposure": "pressure"}, True),
    "message_only": ({"exposure": "message"}, True),
    "placebo_message_only": ({"campaign": PLACEBO, "exposure": "message"}, True),
}
DECOMPOSITION = {"pressure_only", "message_only", "placebo_message_only"}
# Paired contrasts per replication r (same control). "campaign" is the source paired arm.
CONTRASTS = [("campaign", "placebo"),                    # content, with media pressure
             ("message_only", "placebo_message_only"),   # content, message channel alone
             ("pressure_only", "message_only"),          # pressure channel vs message channel
             ("campaign", "pressure_only"), ("campaign", "message_only")]


def ci(d: list[float]) -> tuple[float, float]:
    m, se = st.mean(d), st.stdev(d) / len(d) ** 0.5
    h = float(stats.t.ppf(0.975, len(d) - 1)) * se
    return round(m - h, 5), round(m + h, 5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="runs/llm_phase1")
    ap.add_argument("--out", default="runs/audit_llm")
    ap.add_argument("--llm-kind", choices=["standin", "openai"], default="openai")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rpm", type=int, default=None)
    ap.add_argument("--n-agents", type=int, default=100)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--t-star", type=int, default=2)
    ap.add_argument("--topology", default="small_world")
    ap.add_argument("--reach", type=float, default=0.05)
    ap.add_argument("--checks", nargs="+", default=list(CHECKS))
    ap.add_argument("--reps", nargs="+", type=int, default=[10, 3, 3],
                    help="replications per check, in the order of --checks")
    ap.add_argument("--replay-only", action="store_true")
    ap.add_argument("--max-api-calls", type=int, default=None,
                    help="stop before a replication once this process has made this many calls")
    args = ap.parse_args()
    load_env(SV2 / ".env")
    model = args.model if args.llm_kind == "openai" else args.llm_kind
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = json.loads((HERE / args.source / "summary.json").read_text())
    ref = next(c for c in src["cells"] if c["topology"] == args.topology
               and c["reach"] == args.reach and c["arm"] == "paired")
    caches: dict[float, PairedCache] = {}
    rows, t0 = [], time.time()
    for name, reps in zip(args.checks, args.reps):
        change, reuse_source = CHECKS[name]
        a = copy.copy(args)
        a.campaign = CAMPAIGN
        for k, v in change.items():
            setattr(a, k, v)
        if a.temperature not in caches:
            caches[a.temperature] = PairedCache(out / f"cache_t{a.temperature}.jsonl", model=model,
                                                temperature=a.temperature,
                                                max_tokens=args.max_tokens)
            if a.temperature == args.temperature:      # the source controls share these settings
                c = caches[a.temperature]
                for line in (HERE / args.source / "cache.jsonl").read_text().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        c.entries.setdefault(r["key"], r["reply"])
        cache = caches[a.temperature]
        for rep in range(reps):
            made = sum(x.api_calls for x in caches.values())
            if args.max_api_calls is not None and made >= args.max_api_calls:
                sys.exit(f"stopped: {made} API calls reached --max-api-calls; rerun resumes")
            base = f"{args.topology}/r{rep}"
            cns = f"{base}/ctrl" if reuse_source else f"{base}/{name}/ctrl"
            cdb = out / name / f"r{rep}" / "ctrl" / "study.duckdb"
            c = run_arm(cache, a, topology=args.topology, reach=None, ns=cns, reuse_ns=None,
                        db=cdb, warm_from=None)
            r = run_arm(cache, a, topology=args.topology, reach=args.reach,
                        ns=f"{base}/{name}/paired{args.reach}", reuse_ns=cns,
                        db=out / name / f"r{rep}" / "paired" / "study.duckdb", warm_from=cdb)
            r.update(check=name, rep=rep, delta=round(r["mean_opinion"] - c["mean_opinion"], 5),
                     ctrl_paid=c["paid"])
            rows.append(r)
            print(f"[{time.time() - t0:6.0f}s] {name:10s} r{rep} delta={r['delta']:+.4f} "
                  f"paid={r['paid']:4d} ctrl_paid={c['paid']:4d} "
                  f"api_calls={sum(x.api_calls for x in caches.values())}", flush=True)
    cells = []
    for name in args.checks:
        d = [r["delta"] for r in rows if r["check"] == name]
        lo, hi = ci(d)
        want_zero = name == "placebo"
        same_sign = (lo > 0) if ref["delta_mean"] > 0 else (hi < 0)
        cells.append({"check": name, "n_reps": len(d), "delta_mean": round(st.mean(d), 5),
                      "delta_sd": round(st.stdev(d), 5), "ci95": [lo, hi],
                      "passes": None if name in DECOMPOSITION
                      else (lo <= 0 <= hi) if want_zero else same_sign})
    deltas = {(r["check"], r["rep"]): r["delta"] for r in rows}
    with open(HERE / args.source / "runs.csv") as f:
        for r in csv.DictReader(f):
            if (r["topology"] == args.topology and r["arm"] == "paired" and r["reach"]
                    and float(r["reach"]) == args.reach):
                deltas[("campaign", int(r["rep"]))] = float(r["delta"])
    contrasts = []
    for x, y in CONTRASTS:
        reps = sorted(i for (c, i) in deltas if c == x and (y, i) in deltas)
        if len(reps) >= 2:
            d = [deltas[(x, i)] - deltas[(y, i)] for i in reps]
            contrasts.append({"contrast": f"{x} - {y}", "n_reps": len(d),
                              "mean": round(st.mean(d), 5), "ci95": list(ci(d))})
    summary = {"config": dict(vars(args)),
               "reference": {"source": args.source, **ref},
               "api_calls_this_process": sum(x.api_calls for x in caches.values()),
               "wall_seconds": round(time.time() - t0, 1), "cells": cells,
               "contrasts": contrasts,
               "runs": [{k: r[k] for k in ("check", "rep", "delta", "paid", "ctrl_paid")}
                        for r in rows]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("reference:", ref)
    for cell in cells + contrasts:
        print(cell)


if __name__ == "__main__":
    main()
