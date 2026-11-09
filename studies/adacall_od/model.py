"""adacall_od — opinion diffusion with a targeted campaign, the primary AdaCall testbed.

Derived from the opinion_diffusion reference study, with three changes:
  - the interaction network is configurable (ring, small-world or dense random graph);
  - the campaign is a Broadcast whose audience may be a list of agent ids, so an intervention
    can reach k% of agents instead of everyone (the reference provider drops non-"all" audiences);
  - the decision model calls its LLM through ``self.client(prompt, agent_id=, step=, **ctx)``,
    which the AdaCall runner replaces with a paired-cache client (adacall/cache.py).

Decision modes (``llm_kind``):
  - ``scripted``: the deterministic bounded-confidence rule, no client involved;
  - ``standin``: the rule plus Gaussian noise drawn per (namespace, agent, step, prompt), returned
    as a JSON reply through the same client path as a real LLM (free pipeline tests);
  - ``openai``: an OpenAI-compatible endpoint (SV_LLM_API_KEY / SV_LLM_BASE_URL / SV_LLM_MODEL, or
    INTERN_API_KEY / INTERN_BASE_URL for ``intern-*``, ZHIPU_API_KEY / ZHIPU_BASE_URL for ``glm-*``),
    optionally capped at ``rpm`` calls/min.

Prompts depend only on the agent's own opinion, its neighbours' opinions and its own exposure,
so a treatment agent's prompt equals its control prompt exactly until the campaign or a changed
neighbour reaches it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, pstdev
from typing import Any, Callable

import networkx as nx
import numpy as np

from socioverse.abc.decision import DecisionModel
from socioverse.abc.environment import EnvironmentProvider
from socioverse.abc.population import PopulationProvider
from socioverse.abc.trajectory import MetricCollector
from socioverse.engine.registry import register
from socioverse.schemas.environment import EnvironmentBundle, ScheduledEvent
from socioverse.schemas.population import InteractionStructure, Persona, PopulationBundle
from socioverse.schemas.runtime import Action, Observation

_log = logging.getLogger(__name__)


# --- shared pure helpers (both providers rebuild the same graph from the same args) -----------

def agent_id(i: int) -> str:
    return f"ac-{i:03d}"


def build_graph(args: dict[str, Any]) -> nx.Graph:
    n = int(args.get("n_agents", 100))
    topo = args.get("topology", "ring")
    seed = int(args.get("graph_seed", 7))
    if topo == "ring":
        return nx.cycle_graph(n)
    if topo == "small_world":
        return nx.connected_watts_strogatz_graph(n, k=int(args.get("k", 4)),
                                                 p=float(args.get("p", 0.1)), seed=seed)
    if topo == "dense":
        return nx.erdos_renyi_graph(n, p=float(args.get("mean_degree", 10)) / n, seed=seed)
    raise ValueError(f"unknown topology: {topo!r}")


def initial_opinions(args: dict[str, Any]) -> list[float]:
    n = int(args.get("n_agents", 100))
    rng = np.random.default_rng(int(args.get("init_seed", 12345)))
    return [round(float(x), 4) for x in rng.uniform(0.0, 1.0, n)]


# --- P ----------------------------------------------------------------------------------------

@register("population", "adacall.pop")
class AdaCallPopulationProvider(PopulationProvider):
    def __init__(self, bundle: PopulationBundle):
        self.bundle = bundle
        self.graph = build_graph(bundle.provider_args)

    def build(self, seed: int) -> list[Persona]:
        x0 = initial_opinions(self.bundle.provider_args)
        return [Persona(agent_id=agent_id(i), group_key="high" if x >= 0.5 else "low",
                        attributes={"opinion0": x, "degree": self.graph.degree(i)},
                        init_state={"opinion": x})
                for i, x in enumerate(x0)]

    def neighbors(self, aid: str) -> list[str]:
        return [agent_id(j) for j in sorted(self.graph.neighbors(int(aid.split("-")[1])))]

    def interaction_structure(self) -> InteractionStructure:
        return self.bundle.interaction


# --- E ----------------------------------------------------------------------------------------

@register("environment", "adacall.env")
class AdaCallEnvironmentProvider(EnvironmentProvider):
    """Owns opinions and campaign exposure. A Broadcast with audience "all" reaches everyone;
    audience {"agent_ids": [...]} reaches only those agents, for ttl steps from at_step."""

    def __init__(self, bundle: EnvironmentBundle):
        self.bundle = bundle
        args = bundle.provider_args
        self.n = int(args.get("n_agents", 100))
        self.graph = build_graph(args)
        self.nbrs = {agent_id(i): [agent_id(j) for j in sorted(self.graph.neighbors(i))]
                     for i in range(self.n)}
        self.opinions: dict[str, float] = {}
        self.exposure: dict[str, str] = {}      # agent -> campaign text active this step

    def reset(self, seed: int) -> None:
        self.opinions = {agent_id(i): x for i, x in enumerate(initial_opinions(self.bundle.provider_args))}
        self.exposure = {}

    def advance_to(self, t: int) -> list[ScheduledEvent]:
        self.exposure = {}
        fired: list[ScheduledEvent] = []
        for b in self.bundle.information_program.broadcasts:
            if not (b.at_step <= t < b.at_step + b.ttl):
                continue
            ids = list(self.opinions) if b.is_macro else list(b.audience.get("agent_ids", []))
            for aid in ids:
                self.exposure[aid] = b.content
            if b.at_step == t:
                fired.append(ScheduledEvent(at_step=t, target_layer="campaign", op="set",
                                            value=len(ids),
                                            note=f"campaign {b.message_id} reaches {len(ids)} agents"))
        return fired

    def apply(self, actions: list[Action]) -> None:
        for a in actions:
            if a.kind == "update_opinion":
                self.opinions[a.agent_id] = float(a.payload["opinion"])

    def observe_batch(self, agent_ids: list[str], t: int, round_idx: int = 0) -> list[Observation]:
        out = []
        for aid in agent_ids:
            campaign = self.exposure.get(aid)
            out.append(Observation(
                agent_id=aid, step=t,
                local_physical={"own_opinion": self.opinions[aid],
                                "neighbor_opinions": [self.opinions[j] for j in self.nbrs[aid]],
                                "media_pressure": 1.0 if campaign else 0.0},
                local_information={"campaign": campaign} if campaign else {},
            ))
        return out

    def agent_state(self, aid: str) -> dict:
        return {"opinion": self.opinions[aid], "exposed": aid in self.exposure}

    def snapshot(self) -> dict:
        return {"n_exposed": len(self.exposure)}


# --- B ----------------------------------------------------------------------------------------

PROMPT = """You are one person in a public-opinion survey. On a continuous scale from 0 to 1, rate how much you support adopting a new policy right now
(0 = fully against, 0.5 = neutral, 1 = fully for). Weigh the information below and give your updated stance:
- Your current stance: {x}
- The current stances of your neighbours: {n}
- Current media pressure (0 = none, 1 = strongest): {m}{c}
How you behave: you move toward neighbours whose stance is close to yours (you ignore opinions that are too far from yours);
media pressure and campaign messages gently push you toward support (closer to 1), but you never change all at once.
Output one line of JSON only: {{"opinion": <a number from 0 to 1>, "reason": "<a reason of at most 10 words>"}}"""

# C3 paraphrase: same information and output format, different wording and order
PARAPHRASE = """Imagine you are answering a poll about a proposed new policy. Give your current level of support on a scale from 0 (completely opposed) through 0.5 (undecided) to 1 (completely in favour).
What you know right now:
- Media pressure in favour of the policy, from 0 (none) to 1 (very strong): {m}
- What the people you talk to think: {n}
- What you thought until now: {x}{c}
Your habits: people whose views are near yours can pull you toward them, but views far from yours do not move you;
media and campaign messages nudge you a little toward supporting the policy, and you never change your mind abruptly.
Output one line of JSON only: {{"opinion": <a number from 0 to 1>, "reason": "<a reason of at most 10 words>"}}"""
TEMPLATES = {"base": PROMPT, "paraphrase": PARAPHRASE}

# tolerant to misspelled keys ("opinance", "opinience") that DeepSeek emits ~1% of the time
_NUM = re.compile(r'"opin\w*"\s*:\s*"?([0-9]*\.?[0-9]+)')
_REASON = re.compile(r'"reason"\s*:\s*"([^"]*)"')


def build_prompt(ob: Observation, variant: str = "base", exposure: str = "both") -> str:
    """``exposure`` splits a campaign into its two prompt channels (C3 decomposition): "both"
    shows the message and media pressure 1, "message" only the message (pressure 0), "pressure"
    only the media pressure (no message line)."""
    lp = ob.local_physical
    campaign = ob.local_information.get("campaign")
    m = float(lp["media_pressure"])
    if campaign and exposure == "message":
        m = 0.0
    if exposure == "pressure":
        campaign = None
    c = f'\n- You just saw a campaign message: "{campaign}"' if campaign else ""
    return TEMPLATES[variant].format(x=round(float(lp["own_opinion"]), 3),
                         n=[round(float(o), 3) for o in lp["neighbor_opinions"]],
                         m=round(m, 3), c=c)


def parse_reply(resp: str) -> dict | None:
    m = _NUM.search(resp or "")
    if m is None:
        return None
    r = _REASON.search(resp)
    return {"opinion": min(1.0, max(0.0, float(m.group(1)))), "reason": r.group(1) if r else ""}


# Billing failures abort the run instead of caching empty replies; rerun after a top-up resumes.
_FATAL = ("Insufficient Balance", "insufficient balance", "Arrearage", "balance is insufficient",
          "Error code: 402", "'code': '1113'", "余额不足")  # 1113: Zhipu out of balance


class OutOfCredit(RuntimeError):
    """The provider refused the call for billing reasons; nothing was cached."""


class RateLimiter:
    """Thread-safe sliding-window limiter: at most ``rpm`` calls in any 60-second window."""

    def __init__(self, rpm: int | None):
        self.rpm = rpm
        self._times: list[float] = []
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.rpm:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self._times = [t for t in self._times if now - t < 60.0]
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return
                delay = 60.0 - (now - self._times[0]) + 0.05
            time.sleep(delay)


class OpenAIBackend:
    """OpenAI-compatible chat call with rate limiting and retries. The provider comes from the
    model string (see ``route``); each provider's key and base URL are read from the environment
    and reasoning output is disabled so the reply is not empty under a small max_tokens."""

    # provider -> (key env, base-url env, default base URL, extra_body that disables reasoning)
    PROVIDERS = {
        "deepseek": ("SV_LLM_API_KEY", "SV_LLM_BASE_URL", "https://api.deepseek.com",
                     {"thinking": {"type": "disabled"}}),
        "intern": ("INTERN_API_KEY", "INTERN_BASE_URL", "https://chat.intern-ai.org.cn/api/v1/",
                   {"thinking_mode": False}),
        "zhipu": ("ZHIPU_API_KEY", "ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/",
                  {"thinking": {"type": "disabled"}}),
        "qwen": ("QWEN_API_KEY", "QWEN_BASE_URL",
                 "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", {"enable_thinking": False}),
        "siliconflow": ("SILICONFLOW_API_KEY", "SILICONFLOW_BASE_URL",
                        "https://api.siliconflow.com/v1", {"enable_thinking": False}),
    }

    def __init__(self, model: str | None, temperature: float, max_tokens: int,
                 rpm: int | None = None):
        self.model = model or os.environ.get("SV_LLM_MODEL", "deepseek-flash")
        self.provider, self.api_model = self.route(self.model)
        self.temperature, self.max_tokens = temperature, max_tokens
        self.limiter = RateLimiter(rpm)
        self._client = None

    @classmethod
    def route(cls, model: str) -> tuple[str, str]:
        """``provider:model`` picks the provider explicitly (e.g. ``qwen:deepseek-v4-flash`` is
        DeepSeek V4 Flash served by Alibaba Bailian); bare names keep the old routing."""
        if ":" in model:
            provider, name = model.split(":", 1)
            if provider not in cls.PROVIDERS:
                raise ValueError(f"unknown provider {provider!r} in model {model!r}")
            return provider, name
        if model.startswith("intern"):
            return "intern", model
        if model.startswith("glm"):
            return "zhipu", model
        return "deepseek", model

    def _connect(self):
        from openai import OpenAI
        key_env, base_env, base_default, _ = self.PROVIDERS[self.provider]
        return OpenAI(api_key=os.environ.get(key_env),
                      base_url=os.environ.get(base_env, base_default))

    def _extra(self) -> dict | None:
        return self.PROVIDERS[self.provider][3]

    def __call__(self, prompt: str, **_ctx) -> str:
        if self._client is None:
            self._client = self._connect()
        for attempt in range(6):
            self.limiter.wait()
            try:
                r = self._client.chat.completions.create(
                    model=self.api_model, temperature=self.temperature, max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}], extra_body=self._extra())
                reply = r.choices[0].message.content or ""
                if reply:
                    return reply
            except Exception as exc:  # transient API errors: back off and retry
                if any(m in str(exc) for m in _FATAL):
                    raise OutOfCredit(f"{self.provider}: {exc}") from exc
                _log.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(min(30, 2 ** attempt))
        return ""


class StandInBackend:
    """Free LLM stand-in: the bounded-confidence rule plus N(0, sigma) noise, as a JSON reply.
    The draw is keyed on (namespace, agent, step, prompt), so two arms that share a namespace and
    a prompt agree, and independent arms do not."""

    def __init__(self, sigma: float = 0.05):
        self.sigma = sigma

    def __call__(self, prompt: str, *, ns: str = "", agent_id: str = "", step: int = 0,
                 rule_value: float = 0.5, **_ctx) -> str:
        h = hashlib.blake2b(repr((ns, agent_id, step, prompt)).encode(), digest_size=8).digest()
        noise = float(np.random.default_rng(int.from_bytes(h, "little")).normal(0.0, self.sigma))
        return f'{{"opinion": {min(1.0, max(0.0, rule_value + noise)):.4f}, "reason": "stand-in"}}'


class _DirectClient:
    """Uncached client used when no runner injects one (e.g. a plain sv-run)."""

    def __init__(self, backend: Callable[..., str], ns: str = "direct"):
        self.backend, self.ns = backend, ns

    def __call__(self, prompt: str, *, agent_id: str, step: int, **ctx) -> str:
        return self.backend(prompt, ns=self.ns, agent_id=agent_id, step=step, **ctx)


@register("decision", "adacall.decision")
class AdaCallDecision(DecisionModel):
    def __init__(self, confidence: float = 0.3, peer_rate: float = 0.5, campaign_gain: float = 0.15,
                 llm_kind: str = "scripted", model: str | None = None, temperature: float = 0.7,
                 max_tokens: int = 80, max_workers: int = 32, sigma: float = 0.05,
                 rpm: int | None = None, prompt_variant: str = "base", exposure: str = "both"):
        self.confidence, self.peer_rate, self.campaign_gain = confidence, peer_rate, campaign_gain
        self.prompt_variant, self.exposure = prompt_variant, exposure
        self.llm_kind = llm_kind
        self.max_workers = max(1, int(max_workers))
        if llm_kind == "openai":
            self.backend: Callable[..., str] | None = OpenAIBackend(model, temperature, max_tokens,
                                                                    rpm=rpm)
        elif llm_kind == "standin":
            self.backend = StandInBackend(sigma)
        elif llm_kind == "scripted":
            self.backend = None
        else:
            raise ValueError(f"unknown llm_kind: {llm_kind!r}")
        self.client = _DirectClient(self.backend) if self.backend else None

    def rule(self, ob: Observation) -> float:
        lp = ob.local_physical
        x = lp["own_opinion"]
        within = [o for o in lp["neighbor_opinions"] if abs(o - x) <= self.confidence]
        new = x + self.peer_rate * (mean([x] + within) - x)
        if ob.local_information.get("campaign"):
            new += self.campaign_gain * (1.0 - new)
        return round(min(1.0, max(0.0, new)), 4)

    def _ask(self, ob: Observation) -> str:
        return self.client(build_prompt(ob, self.prompt_variant, self.exposure), agent_id=ob.agent_id,
                           step=ob.step, rule_value=self.rule(ob))

    def decide_batch(self, obs: list[Observation], memories: dict[str, Any]) -> list[Action]:
        if self.client is None:
            replies = [""] * len(obs)
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(obs))) as ex:
                replies = list(ex.map(self._ask, obs))
        actions = []
        for ob, resp in zip(obs, replies):
            dec = parse_reply(resp) if self.client is not None else None
            if dec is not None:
                new, source, reason = round(dec["opinion"], 4), "llm", dec["reason"]
            else:
                new, reason = self.rule(ob), ""
                source = "fallback" if self.client is not None else "rule"
            actions.append(Action(agent_id=ob.agent_id, step=ob.step, kind="update_opinion",
                                  payload={"opinion": new, "reason": reason}, source=source))
        return actions


# --- metrics ----------------------------------------------------------------------------------

@register("collector", "adacall.collector")
class AdaCallCollector(MetricCollector):
    def collect(self, env: Any, actions: list[Action], t: int) -> dict[str, Any]:
        vals = list(env.opinions.values())
        return {"mean_opinion": round(mean(vals), 5),
                "opinion_std": round(pstdev(vals), 5),
                "frac_above_0_5": round(sum(v >= 0.5 for v in vals) / len(vals), 4),
                "n_exposed": len(env.exposure)}

    def columns(self) -> list[str]:
        return ["mean_opinion", "opinion_std", "frac_above_0_5", "n_exposed"]
