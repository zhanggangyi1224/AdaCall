"""Paired response cache: the C1 mechanism of AdaCall.

Every LLM reply is stored under a content key (namespace, agent, step, prompt, model settings).
Each arm of each replication writes to its own namespace. A paired treatment arm may also read
its control arm's namespace: when a treatment agent's prompt is byte-identical to the control
prompt for the same agent and step, the control reply is reused instead of a new call. Under
temperature > 0 that reply is an exact draw from the same distribution, so reuse is unbiased and
couples the two arms (common random numbers). An independent arm reads only its own namespace.

Entries are keyed per namespace, so every arm's paid calls are counted the same way whether the
run is fresh or replayed from a full cache (replay makes no API calls). Failed calls are cached
as empty replies too, so a replay reproduces the rule fallbacks exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Callable


class PairedCache:
    """Thread-safe, append-only JSONL cache of LLM replies shared across arms."""

    def __init__(self, path: str | Path, *, model: str, temperature: float, max_tokens: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.settings = [model, temperature, max_tokens]
        self.entries: dict[str, str] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.entries[r["key"]] = r["reply"]
        self._lock = threading.Lock()
        self.paid: Counter[str] = Counter()     # namespace -> decisions that needed its own call
        self.reused: Counter[str] = Counter()   # namespace -> decisions served by the control arm
        self.reused_eps: Counter[str] = Counter()  # of those, served only under tolerance eps > 0
        self.api_calls = 0                      # real API calls made in this process
        self.prompts: dict[tuple[str, str, int], str] = {}  # (ns, agent, step) -> prompt seen
        self.audits: list[dict] = []            # C2 audits of tolerant reuse (see ArmClient)
        self.usage_log = os.environ.get("ADACALL_USAGE_LOG")  # usage.py: sizes of paid decisions

    def key(self, ns: str, agent_id: str, step: int, prompt: str) -> str:
        raw = json.dumps([*self.settings, ns, agent_id, step, prompt], ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def client(self, backend: Callable[[str], str], *, ns: str, reuse_ns: str | None = None,
               eps: float = 0.0, audit: float = 0.0, audit_seed: int = 0) -> "ArmClient":
        """The callable a decision model uses for one arm. ``reuse_ns`` is the paired control.
        ``eps > 0`` (C2) also reuses the control reply when the two prompts differ only in numbers
        that are each within ``eps``; a seeded fraction ``audit`` of those decisions is also paid
        under the exact prompt, in namespace ``<ns>/audit``, to estimate the bias reuse adds."""
        return ArmClient(self, backend, ns=ns, reuse_ns=reuse_ns, eps=eps, audit=audit,
                         audit_seed=audit_seed)

    def _lookup(self, k: str) -> str | None:
        with self._lock:
            return self.entries.get(k)

    def _get_or_call(self, backend, ns, reuse_ns, agent_id, step, prompt, ctx,
                     eps: float = 0.0, audit=None) -> str:
        with self._lock:
            self.prompts[(ns, agent_id, step)] = prompt
        own = self.key(ns, agent_id, step, prompt)
        hit = self._lookup(own)
        if hit is not None:                      # this arm's own earlier call (replay)
            with self._lock:
                self.paid[ns] += 1
                self._log_usage(own, ns, prompt, hit)
            return hit
        if reuse_ns is not None:
            hit = self._lookup(self.key(reuse_ns, agent_id, step, prompt))
            if hit is not None:                  # identical prompt in the control arm
                with self._lock:
                    self.reused[ns] += 1
                return hit
            ctrl = self.prompts.get((reuse_ns, agent_id, step))
            if eps > 0 and ctrl is not None and within(prompt, ctrl, eps):
                hit = self._lookup(self.key(reuse_ns, agent_id, step, ctrl))
                if hit is not None:              # C2: control prompt within tolerance eps
                    with self._lock:
                        self.reused[ns] += 1
                        self.reused_eps[ns] += 1
                    if audit is not None and audit(agent_id, step):
                        exact = self._get_or_call(backend, f"{ns}/audit", None, agent_id, step,
                                                  prompt, ctx)
                        with self._lock:
                            self.audits.append({"ns": ns, "agent_id": agent_id, "step": step,
                                                "reused": hit, "exact": exact})
                    return hit
        reply = backend(prompt, ns=ns, agent_id=agent_id, step=step, **ctx) or ""
        with self._lock:
            if own in self.entries:              # another thread filled it meanwhile
                return self.entries[own]
            self.api_calls += 1
            self.paid[ns] += 1
            self._log_usage(own, ns, prompt, reply)
            self.entries[own] = reply
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": own, "ns": ns, "agent_id": agent_id, "step": step,
                                    "reply": reply}, ensure_ascii=False) + "\n")
        return reply

    def _log_usage(self, key: str, ns: str, prompt: str, reply: str) -> None:
        if self.usage_log:
            with open(self.usage_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "ns": ns, "prompt_chars": len(prompt),
                                    "reply_chars": len(reply)}) + "\n")


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def within(a: str, b: str, eps: float) -> bool:
    """True when two prompts have the same text around their numbers and every pair of numbers
    differs by at most ``eps``."""
    if _NUMBER.sub("#", a) != _NUMBER.sub("#", b):
        return False
    return all(abs(float(x) - float(y)) <= eps + 1e-12
               for x, y in zip(_NUMBER.findall(a), _NUMBER.findall(b)))


class ArmClient:
    """Bound view of the cache for one arm; called as ``client(prompt, agent_id=, step=, **ctx)``.
    ``ctx`` is forwarded to the backend only (it never enters the cache key)."""

    def __init__(self, cache: PairedCache, backend, *, ns: str, reuse_ns: str | None,
                 eps: float = 0.0, audit: float = 0.0, audit_seed: int = 0):
        self.cache, self.backend, self.ns, self.reuse_ns = cache, backend, ns, reuse_ns
        self.eps, self.audit, self.audit_seed = eps, audit, audit_seed

    def _draw(self, agent_id: str, step: int) -> bool:
        """Seeded per decision, so the audited set does not depend on thread order."""
        return random.Random(f"{self.audit_seed}/{self.ns}/{agent_id}/{step}").random() < self.audit

    def __call__(self, prompt: str, *, agent_id: str, step: int, **ctx) -> str:
        return self.cache._get_or_call(self.backend, self.ns, self.reuse_ns, agent_id, step,
                                       prompt, ctx, eps=self.eps,
                                       audit=self._draw if self.audit > 0 else None)
