"""C2 tolerant reuse in the paired cache (free: a counting fake backend)."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from cache import PairedCache, within  # noqa: E402


def test_within():
    assert within("x 0.50 n [0.1, 0.2]", "x 0.52 n [0.1, 0.19]", 0.03)
    assert not within("x 0.50 n [0.1, 0.2]", "x 0.55 n [0.1, 0.2]", 0.03)
    assert not within("x 0.50 campaign", "x 0.50", 0.1)        # text must match exactly


def _setup(tmp_path, eps, audit=0.0):
    calls = []

    def backend(prompt, **ctx):
        calls.append(prompt)
        return f"reply to {prompt}"

    cache = PairedCache(tmp_path / "c.jsonl", model="fake", temperature=0.7, max_tokens=80)
    cache.client(backend, ns="ctrl")("x 0.50", agent_id="a", step=3)
    tr = cache.client(backend, ns="tr", reuse_ns="ctrl", eps=eps, audit=audit)
    return cache, tr, calls


def test_eps_zero_pays_for_any_difference(tmp_path):
    cache, tr, calls = _setup(tmp_path, eps=0.0)
    assert tr("x 0.51", agent_id="a", step=3) == "reply to x 0.51"
    assert cache.paid["tr"] == 1 and cache.reused_eps["tr"] == 0


def test_eps_reuses_the_control_reply_within_tolerance(tmp_path):
    cache, tr, calls = _setup(tmp_path, eps=0.02)
    assert tr("x 0.51", agent_id="a", step=3) == "reply to x 0.50"
    assert tr("x 0.60", agent_id="a", step=3) == "reply to x 0.60"
    assert cache.reused_eps["tr"] == 1 and cache.paid["tr"] == 1


def test_audit_pays_the_exact_prompt_but_returns_the_reused_reply(tmp_path):
    cache, tr, calls = _setup(tmp_path, eps=0.02, audit=1.0)
    assert tr("x 0.51", agent_id="a", step=3) == "reply to x 0.50"
    assert cache.paid["tr/audit"] == 1
    assert cache.audits == [{"ns": "tr", "agent_id": "a", "step": 3,
                             "reused": "reply to x 0.50", "exact": "reply to x 0.51"}]


def test_served_from_exact_lists_only_paid_replies_the_exact_arm_had(tmp_path):
    from run_eps import served_from_exact
    cache, tr, calls = _setup(tmp_path, eps=0.02)
    exact = cache.client(lambda prompt, **ctx: f"reply to {prompt}", ns="exact", reuse_ns="ctrl")
    exact("x 0.60", agent_id="a", step=3)                     # the exact arm paid for this prompt
    tr("x 0.51", agent_id="a", step=3)                        # tolerant reuse: not paid, not served
    tr("x 0.60", agent_id="b", step=3)                        # paid; the exact arm never saw agent b
    tr("x 0.60", agent_id="a", step=4)                        # paid; exact arm had no step 4
    cache.client(lambda prompt, **ctx: "r", ns="tr", reuse_ns="ctrl")("x 0.60", agent_id="a", step=3)
    served = served_from_exact(cache, "tr", "exact")
    assert served == [cache.key("tr", "a", 3, "x 0.60")]
