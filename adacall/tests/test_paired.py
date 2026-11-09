"""Correctness tests for paired, divergence-triggered calls (free: stand-in backend)."""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "SocioVerse2"))
sys.path.insert(0, str(HERE))

import studies.adacall_od as od  # noqa: E402
from cache import PairedCache  # noqa: E402
from socioverse.engine import build_simulator  # noqa: E402
from socioverse.schemas.simulation import WarmStartSpec  # noqa: E402

N, T, TSTAR = 40, 6, 2
ARGS = {"llm_kind": "standin", "max_workers": 8}


def _run(tmp: Path, cache: PairedCache, name: str, *, reach, ns, reuse_ns=None, warm=None,
         topology="ring"):
    _, env, pop, sim = od.make_bundles(n_agents=N, topology=topology, n_steps=T, t_star=TSTAR,
                                       reach=reach, decision_args=dict(ARGS))
    if warm is not None:
        sim.warm_start = WarmStartSpec(source_version="ctrl", source_trajectory=str(warm),
                                       resume_from=TSTAR - 1)
    db = tmp / name / "study.duckdb"
    db.parent.mkdir(parents=True)
    s = build_simulator(env_bundle=env, pop_bundle=pop, sim_config=sim, store_path=db)
    s.decision.client = cache.client(s.decision.backend, ns=ns, reuse_ns=reuse_ns)
    s.run()
    return db


def _opinions(db: Path, step: int) -> dict[str, float]:
    con = duckdb.connect(str(db), read_only=True)
    rows = con.execute("SELECT agent_id, CAST(state->>'opinion' AS DOUBLE) FROM panel "
                       "WHERE step = ?", [step]).fetchall()
    con.close()
    return dict(rows)


@pytest.fixture
def cache(tmp_path):
    return PairedCache(tmp_path / "cache.jsonl", model="standin", temperature=0.7, max_tokens=80)


def test_identical_branch_pays_nothing_and_matches_control(tmp_path, cache):
    ctrl = _run(tmp_path, cache, "ctrl", reach=None, ns="c")
    br = _run(tmp_path, cache, "br", reach=None, ns="b", reuse_ns="c", warm=ctrl)
    assert cache.paid["b"] == 0
    assert cache.reused["b"] == N * (T - TSTAR + 1)
    for t in range(T + 1):
        assert _opinions(br, t) == _opinions(ctrl, t)


def test_only_targeted_agents_pay_at_the_fork_step(tmp_path, cache):
    ctrl = _run(tmp_path, cache, "ctrl", reach=None, ns="c")
    calls_before = cache.api_calls
    tr = _run(tmp_path, cache, "tr", reach=0.1, ns="t", reuse_ns="c", warm=ctrl)
    targets = set(od.target_ids(N, 0.1))
    a, b = _opinions(ctrl, TSTAR), _opinions(tr, TSTAR)
    changed = {k for k in a if a[k] != b[k]}
    assert changed <= targets                         # untargeted agents reused control draws
    assert all(_opinions(tr, t) == _opinions(ctrl, t) for t in range(TSTAR))  # replayed prefix
    assert 0 < cache.paid["t"] < N * (T - TSTAR + 1)  # paid only for diverged agents
    assert cache.api_calls - calls_before == cache.paid["t"]


def test_independent_arm_never_reuses(tmp_path, cache):
    _run(tmp_path, cache, "ctrl", reach=None, ns="c")
    _run(tmp_path, cache, "ind", reach=0.1, ns="i")
    assert cache.reused["i"] == 0 and cache.paid["i"] == N * T


def test_replay_from_cache_makes_no_calls_and_reproduces(tmp_path):
    path = tmp_path / "cache.jsonl"
    c1 = PairedCache(path, model="standin", temperature=0.7, max_tokens=80)
    ctrl1 = _run(tmp_path / "a", c1, "ctrl", reach=None, ns="c")
    tr1 = _run(tmp_path / "a", c1, "tr", reach=0.1, ns="t", reuse_ns="c", warm=ctrl1)
    c2 = PairedCache(path, model="standin", temperature=0.7, max_tokens=80)
    ctrl2 = _run(tmp_path / "b", c2, "ctrl", reach=None, ns="c")
    tr2 = _run(tmp_path / "b", c2, "tr", reach=0.1, ns="t", reuse_ns="c", warm=ctrl2)
    assert c2.api_calls == 0
    assert c2.paid["t"] == c1.paid["t"] and c2.reused["t"] == c1.reused["t"]
    assert _opinions(tr2, T) == _opinions(tr1, T)
