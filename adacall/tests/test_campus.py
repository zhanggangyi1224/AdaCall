"""Paired-call tests on the group-spillover testbed (free: stand-in backend)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "SocioVerse2"))
sys.path.insert(0, str(HERE))

import studies.adacall_campus as cd  # noqa: E402
from cache import PairedCache  # noqa: E402
from socioverse.engine import build_simulator  # noqa: E402
from socioverse.schemas.simulation import WarmStartSpec  # noqa: E402
from studies.adacall_campus.model import build_prompt, parse_reply  # noqa: E402
from studies.adacall_germany.model import is_cjk  # noqa: E402

N, TSTAR = 100, 2
ARGS = {"llm_kind": "standin", "max_workers": 8}


def _run(tmp: Path, cache: PairedCache, name: str, *, target, ns, steps, reuse_ns=None, warm=None):
    _, env, pop, sim = cd.make_bundles(n_agents=N, n_steps=steps, t_star=TSTAR, target=target,
                                       decision_args=dict(ARGS))
    if warm is not None:
        sim.warm_start = WarmStartSpec(source_version="ctrl", source_trajectory=str(warm),
                                       resume_from=TSTAR - 1)
    db = tmp / name / "study.duckdb"
    db.parent.mkdir(parents=True)
    s = build_simulator(env_bundle=env, pop_bundle=pop, sim_config=sim, store_path=db)
    s.decision.client = cache.client(s.decision.backend, ns=ns, reuse_ns=reuse_ns)
    s.run()
    return db, s


@pytest.fixture
def cache(tmp_path):
    return PairedCache(tmp_path / "cache.jsonl", model="standin", temperature=0.7, max_tokens=80)


def test_identical_branch_pays_nothing(tmp_path, cache):
    ctrl, _ = _run(tmp_path, cache, "ctrl", target=None, ns="c", steps=5)
    _run(tmp_path, cache, "br", target=None, ns="b", steps=5, reuse_ns="c", warm=ctrl)
    assert cache.paid["b"] == 0 and cache.reused["b"] == N * (5 - TSTAR + 1)


def test_only_subsidised_students_pay_at_the_fork_step(tmp_path, cache):
    ctrl, _ = _run(tmp_path, cache, "ctrl", target=None, ns="c", steps=TSTAR)
    _, s = _run(tmp_path, cache, "tr", target=5, ns="t", steps=TSTAR, reuse_ns="c", warm=ctrl)
    assert 0 < len(s.env.subsidy) < N
    assert cache.paid["t"] == len(s.env.subsidy)      # spillover can only start a month later


def test_spillover_after_the_fork_costs_more_than_the_targets(tmp_path, cache):
    ctrl, _ = _run(tmp_path, cache, "ctrl", target=None, ns="c", steps=6)
    _, s = _run(tmp_path, cache, "tr", target=5, ns="t", steps=6, reuse_ns="c", warm=ctrl)
    assert cache.paid["t"] > len(s.env.subsidy) * (6 - TSTAR + 1)


def test_prompt_is_english_and_shows_the_subsidy_only_to_targets(tmp_path):
    _, env, pop, sim = cd.make_bundles(n_agents=N, n_steps=4, t_star=TSTAR, target=5)
    s = build_simulator(env_bundle=env, pop_bundle=pop, sim_config=sim,
                        store_path=tmp_path / "s.duckdb")
    s.env.reset(sim.seed)
    s.env.advance_to(TSTAR)
    for ob in s.env.observe_batch(list(s.env.students), TSTAR):
        p = build_prompt(ob)
        assert not is_cjk(p)
        assert ("canteen subsidy" in p) == (ob.agent_id in s.env.subsidy)


def test_parse_reply():
    r = parse_reply('{"choice": "delivery", "satisfaction": 0.42, "reason": "lazy"}')
    assert r == {"choice": "delivery", "satisfaction": 0.42, "reason": "lazy"}
    assert parse_reply('{"choice": "Canteen", "satisfaction": "0.7"}')["satisfaction"] == 0.7
    assert parse_reply("canteen") is None
