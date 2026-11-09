"""C2 sweep: reuse tolerance eps vs. calls, bias and MSE of the treatment effect (opinion diffusion).

For each (topology, replication) the control arm and the exact paired arm (eps = 0) are replayed
from the committed run in ``--source`` (no API calls); each eps > 0 arm is a paired branch whose
agents also reuse the control reply when their prompt differs from the control prompt only in
numbers within eps. A treatment prompt that is byte-identical to one the exact paired arm already
sent is served from that arm's reply (an exact draw, so no bias is added; it still counts as paid
in the cost, but it reaches no provider: ``served_from_exact`` in runs.csv and
``served_from_exact.txt`` list these replies so the spend accounting can exclude them). A seeded
fraction ``--audit`` of the tolerant reuses is also paid under the exact
prompt, which estimates the per-decision error tolerance adds without a full independent run.

bias(eps) is the mean over replications of delta(eps) - delta(0), which shares the control arm and
every exact draw, so it is estimated with little noise. With budget B (in calls) and c(eps) calls
per replication, MSE(eps) = bias^2 + var(delta(eps)) * c(eps) / B.

Usage:
  python run_eps.py --llm-kind standin --source runs/standin40 --out runs/eps_standin --reps 20
  python run_eps.py --llm-kind openai --model deepseek-flash --out runs/eps_llm --reps 10
  python run_eps.py --model deepseek-flash --out runs/eps_llm_holdout --reps 10 --rep-start 10 \
      --eps 0.0 0.01 0.03 --max-api-calls 4500      # preregistered held-out replications r10-r19
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cache import PairedCache  # noqa: E402
from run_od import SV2, _no_calls, load_env, run_arm  # noqa: E402
from studies.adacall_od.model import parse_reply  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="runs/llm_phase1", help="run whose cache seeds ctrl/eps=0")
    ap.add_argument("--out", default="runs/eps_llm")
    ap.add_argument("--llm-kind", choices=["standin", "openai"], default="openai")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rpm", type=int, default=None)
    ap.add_argument("--n-agents", type=int, default=100)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--t-star", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--rep-start", type=int, default=0,
                    help="first replication index (held-out runs use the source run's later reps)")
    ap.add_argument("--max-api-calls", type=int, default=None,
                    help="stop before the next replication once this many provider calls were made")
    ap.add_argument("--topologies", nargs="+", default=["small_world", "ring"])
    ap.add_argument("--reach", type=float, default=0.05)
    ap.add_argument("--eps", nargs="+", type=float, default=[0.0, 0.01, 0.03, 0.1])
    ap.add_argument("--audit", type=float, default=0.1)
    ap.add_argument("--replay-only", action="store_true")
    args = ap.parse_args()
    load_env(SV2 / ".env")
    model = args.model if args.llm_kind == "openai" else args.llm_kind
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = PairedCache(out / "cache.jsonl", model=model, temperature=args.temperature,
                        max_tokens=args.max_tokens)
    seeded = 0                                    # source replies are read, never rewritten
    for line in (HERE / args.source / "cache.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            cache.entries.setdefault(r["key"], r["reply"])
            seeded += 1
    rows, t0, served_keys = [], time.time(), []
    for topo in args.topologies:
        for rep in range(args.rep_start, args.rep_start + args.reps):
            if args.max_api_calls is not None and REAL_CALLS[0] >= args.max_api_calls:
                print(f"stopping before {topo} r{rep}: {REAL_CALLS[0]} provider calls >= cap",
                      flush=True)
                break
            base = f"{topo}/r{rep}"
            ctrl_db = out / topo / f"r{rep}" / "ctrl" / "study.duckdb"
            c = run_arm(cache, args, topology=topo, reach=None, ns=f"{base}/ctrl", reuse_ns=None,
                        db=ctrl_db, warm_from=None)
            exact_ns = f"{base}/paired{args.reach}"
            for eps in args.eps:
                ns = exact_ns if eps == 0 else f"{exact_ns}_e{eps}"
                real = _real_backend(args)
                backend = _no_calls if args.replay_only else _via_exact(cache, exact_ns, real)
                r = run_arm(cache, args, topology=topo, reach=args.reach, ns=ns,
                            reuse_ns=f"{base}/ctrl", db=out / topo / f"r{rep}" / f"e{eps}" /
                            "study.duckdb", warm_from=ctrl_db, eps=eps, audit=args.audit,
                            backend=backend)
                served = [] if eps == 0 else served_from_exact(cache, ns, exact_ns)
                served_keys += served
                r.update(topology=topo, rep=rep, eps=eps, reused_eps=cache.reused_eps[ns],
                         served_from_exact=len(served),
                         delta=round(r["mean_opinion"] - c["mean_opinion"], 5))
                rows.append(r)
                print(f"[{time.time() - t0:6.0f}s] {topo:11s} r{rep} eps={eps:<5} "
                      f"delta={r['delta']:+.4f} paid={r['paid']:4d} reused={r['reused']:4d} "
                      f"(tolerant {r['reused_eps']:4d}, served {len(served):3d}) "
                      f"api_calls={REAL_CALLS[0]}", flush=True)
    summary = {"config": dict(vars(args)), "seeded_replies": seeded,
               "api_calls_this_process": REAL_CALLS[0], "wall_seconds": round(time.time() - t0, 1),
               "served_from_exact_arm": len(served_keys),
               "cells": summarize(rows, args), "audit": audit_summary(cache)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "served_from_exact.txt").write_text("".join(k + "\n" for k in served_keys))
    with (out / "runs.csv").open("w", newline="") as f:
        fields = ["topology", "rep", "eps", "delta", "mean_opinion", "paid", "reused", "reused_eps",
                  "served_from_exact", "ns"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    for cell in summary["cells"]:
        print(cell)
    print("audit:", summary["audit"])
    print(f"api calls this process: {REAL_CALLS[0]}")


REAL_CALLS = [0]                                  # calls that reached the provider


def _real_backend(args):
    from studies.adacall_od.model import OpenAIBackend, StandInBackend
    return (OpenAIBackend(args.model, args.temperature, args.max_tokens, rpm=args.rpm)
            if args.llm_kind == "openai" else StandInBackend())


def _via_exact(cache: PairedCache, exact_ns: str, real):
    """Serve a prompt the exact paired arm already sent from that arm's reply; else call."""
    def call(prompt: str, *, ns: str = "", agent_id: str = "", step: int = 0, **ctx) -> str:
        hit = cache._lookup(cache.key(exact_ns, agent_id, step, prompt))
        if hit is not None:
            return hit
        with cache._lock:
            REAL_CALLS[0] += 1
        return real(prompt, ns=ns, agent_id=agent_id, step=step, **ctx)
    return call


def served_from_exact(cache: PairedCache, ns: str, exact_ns: str) -> list[str]:
    """Cache keys of this arm's (and its audits') replies whose prompt the exact paired arm had
    already sent: ``_via_exact`` served them from that arm, so no provider call was made."""
    keys = []
    for (n, agent_id, step), prompt in list(cache.prompts.items()):
        if n not in (ns, f"{ns}/audit"):
            continue
        own = cache.key(n, agent_id, step, prompt)
        if cache._lookup(own) is not None and \
                cache._lookup(cache.key(exact_ns, agent_id, step, prompt)) is not None:
            keys.append(own)                      # paid by this arm, served by the exact arm
    return keys


def summarize(rows: list[dict], args) -> list[dict]:
    post = args.n_agents * (args.steps - args.t_star + 1)
    ctrl = args.n_agents * args.steps
    out = []
    for topo in args.topologies:
        base = {r["rep"]: r["delta"] for r in rows if r["topology"] == topo and r["eps"] == 0}
        for eps in args.eps:
            rs = [r for r in rows if r["topology"] == topo and r["eps"] == eps]
            d = [r["delta"] for r in rs]
            diff = [r["delta"] - base[r["rep"]] for r in rs if r["rep"] in base]
            paid = st.mean(r["paid"] for r in rs)
            out.append({
                "topology": topo, "eps": eps, "n_reps": len(rs),
                "paid_vs_postfork": round(paid / post, 4),
                "tolerant_share": round(st.mean(r["reused_eps"] for r in rs) / post, 4),
                "calls_per_rep": round(ctrl + paid, 1),
                "delta_mean": round(st.mean(d), 5),
                "delta_sd": round(st.stdev(d), 5) if len(d) > 1 else None,
                "bias_vs_eps0": round(st.mean(diff), 5) if diff else None,
                "bias_se": round(st.stdev(diff) / len(diff) ** 0.5, 5) if len(diff) > 1 else None,
            })
    return out


def _errors(audits: list[dict]) -> dict:
    errs = []
    for a in audits:
        x, y = parse_reply(a["reused"]), parse_reply(a["exact"])
        if x and y:
            errs.append(y["opinion"] - x["opinion"])
    if not errs:
        return {"n": 0}
    se = st.stdev(errs) / len(errs) ** 0.5 if len(errs) > 1 else None
    return {"n": len(errs), "mean_error": round(st.mean(errs), 5),
            "mean_abs_error": round(st.mean(abs(e) for e in errs), 5),
            "se": round(se, 5) if se is not None else None,
            "ci95": [round(st.mean(errs) - 1.96 * se, 5), round(st.mean(errs) + 1.96 * se, 5)]
            if se is not None else None}


def audit_summary(cache: PairedCache) -> dict:
    """Mean audited-minus-reused opinion over all audits, and per (topology, eps)."""
    out = _errors(cache.audits)
    groups: dict[str, list[dict]] = {}
    for a in cache.audits:
        topo, eps = a["ns"].split("/")[0], a["ns"].rsplit("_e", 1)[1]
        groups.setdefault(f"{topo}/eps{eps}", []).append(a)
    out["by_cell"] = {k: _errors(v) for k, v in sorted(groups.items())}
    return out


if __name__ == "__main__":
    main()
