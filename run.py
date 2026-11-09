"""Entry point.

  python run.py                     # free: every design on all three testbeds with the stand-in LLM
  python run.py --mode live --study od --model deepseek-flash --reps 5
                                    # real LLM (key in SocioVerse2/.env); reruns resume from the cache

Run ``bash scripts/setup.sh`` once first. Seeds are fixed in each study's bundle factory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

AD = Path(__file__).resolve().parent / "adacall"

STANDIN = [
    ("standin40", ["run_od.py", "--reps", "40"]),
    ("standin_germany", ["run_paired.py", "--study", "germany", "--reps", "20"]),
    ("standin_campus", ["run_paired.py", "--study", "campus", "--reps", "20"]),
    ("eps_standin", ["run_eps.py", "--source", "runs/standin40", "--reps", "20"]),
    ("audit_standin", ["run_audit.py", "--source", "runs/standin40", "--checks", "placebo", "temp0",
                       "paraphrase", "--reps", "20", "10", "10"]),
]


def py(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=AD, check=True)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--mode", choices=["standin", "live"], default="standin")
    ap.add_argument("--study", choices=["od", "germany", "campus"], default="od")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=None, help="live mode: output directory under adacall/")
    args, extra = ap.parse_known_args()
    if args.mode == "standin":
        for name, a in STANDIN:
            py([*a, "--llm-kind", "standin", "--out", f"runs/{name}"])
    else:
        out = args.out or f"runs/llm_{args.study}_{args.model.replace(':', '_')}"
        runner = ["run_od.py"] if args.study == "od" else ["run_paired.py", "--study", args.study]
        py([*runner, "--llm-kind", "openai", "--model", args.model, "--reps", str(args.reps),
            "--out", out, *extra])


if __name__ == "__main__":
    main()
