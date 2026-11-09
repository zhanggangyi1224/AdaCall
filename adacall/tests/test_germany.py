"""Paired-call tests on the no-spillover testbed (free: stand-in backend)."""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "SocioVerse2"))
sys.path.insert(0, str(HERE))

import studies.adacall_germany as de  # noqa: E402
from cache import PairedCache  # noqa: E402
from socioverse.engine import build_simulator  # noqa: E402
from socioverse.schemas.simulation import WarmStartSpec  # noqa: E402
from studies.adacall_germany.model import build_prompt, is_cjk, parse_reply  # noqa: E402

N, T, TSTAR = 100, 5, 2
ARGS = {"llm_kind": "standin", "max_workers": 8}


def _run(tmp: Path, cache: PairedCache, name: str, *, target, ns, reuse_ns=None, warm=None):
    _, env, pop, sim = de.make_bundles(n_agents=N, n_steps=T, t_star=TSTAR, target=target,
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


def _panel(db: Path, step: int) -> dict[str, tuple]:
    con = duckdb.connect(str(db), read_only=True)
    rows = con.execute("SELECT agent_id, state->>'bundesland', state->>'choice' FROM panel "
                       "WHERE step = ?", [step]).fetchall()
    con.close()
    return {a: (b, c) for a, b, c in rows}


@pytest.fixture
def cache(tmp_path):
    return PairedCache(tmp_path / "cache.jsonl", model="standin", temperature=0.7, max_tokens=80)


def test_regional_branch_pays_only_for_targeted_buyers(tmp_path, cache):
    ctrl, _ = _run(tmp_path, cache, "ctrl", target=None, ns="c")
    tr, s = _run(tmp_path, cache, "tr", target="Bayern", ns="t", reuse_ns="c", warm=ctrl)
    bavarians = {a for a, (state, _) in _panel(ctrl, 0).items() if state == "Bayern"}
    assert 0 < len(bavarians) < N
    assert cache.paid["t"] == len(bavarians) * (T - TSTAR + 1)   # no spillover: exact count
    for t in range(T + 1):
        a, b = _panel(ctrl, t), _panel(tr, t)
        assert {k for k in a if a[k] != b[k]} <= bavarians


def test_nationwide_branch_pays_every_post_fork_decision(tmp_path, cache):
    ctrl, _ = _run(tmp_path, cache, "ctrl", target=None, ns="c")
    _run(tmp_path, cache, "tr", target="all", ns="t", reuse_ns="c", warm=ctrl)
    assert cache.paid["t"] == N * (T - TSTAR + 1) and cache.reused["t"] == 0


def test_prompt_is_english_and_names_the_incentive_only_when_exposed(tmp_path):
    _, env, pop, sim = de.make_bundles(n_agents=N, n_steps=T, t_star=TSTAR, target="Bayern")
    s = build_simulator(env_bundle=env, pop_bundle=pop, sim_config=sim,
                        store_path=tmp_path / "s.duckdb")
    s.env.reset(sim.seed)
    s.env.advance_to(TSTAR)
    for ob in s.env.observe_batch(list(s.env.buyers), TSTAR):
        p = build_prompt(ob)
        assert not is_cjk(p)
        assert ("purchase bonus" in p) == (ob.local_physical["profile"]["demographics"]
                                           ["bundesland_name"] == "Bayern")


def test_parse_reply():
    assert parse_reply('{"choice": "seat_cupra", "reason": "x"}')["choice"] == "seat_cupra"
    assert parse_reply('{"choice": "Volkswagen", "reason": "x"}')["choice"] == "vw"
    assert parse_reply('{"choice": "BYD"}')["choice"] == "byd"
    assert parse_reply("I would buy a Tesla") is None
