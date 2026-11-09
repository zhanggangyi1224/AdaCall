"""adacall_germany — car-brand choice under a regional EV purchase incentive, AdaCall's
no-spillover testbed.

Derived from the germany_auto_market reference study (same real-population sampler, brand set and
data-driven EV / BYD / Tesla signals), with these changes:
  - an English buyer prompt; the reference prompt is Chinese-only;
  - an EV purchase incentive delivered as a Broadcast whose audience may be one or more federal
    states ({"bundesland": [...]}) or a list of agent ids, rendered in the prompt of exposed
    agents only (the reference provider ignores audiences);
  - no truth anchor and no cohort trend: brand appeal follows the real KBA share of the month,
    exogenously, so one agent's choice never enters another agent's prompt (no spillover);
  - the decision model calls its LLM through ``self.client(prompt, agent_id=, step=, **ctx)``,
    which the AdaCall runner replaces with a paired-cache client (adacall/cache.py).

A prompt depends only on the agent's own profile, its current brand, the month's market signals and
the news it sees. An untargeted agent's prompt therefore equals its control prompt at every step, and
a paired treatment branch pays only for the targeted agents.

Decision modes (``llm_kind``): ``scripted`` (random-utility rule with persistent taste noise),
``standin`` (the rule plus a per-(namespace, agent, step, prompt) Gumbel draw, returned as a JSON
reply through the client path; free pipeline tests) and ``openai`` (see adacall_od.model.OpenAIBackend).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from socioverse.abc import DecisionModel, EnvironmentProvider, MetricCollector, PopulationProvider
from socioverse.engine import register
from socioverse.schemas import (Action, EnvironmentBundle, InteractionStructure, Observation,
                                Persona, PopulationBundle, ScheduledEvent)
from studies.adacall_od.model import OpenAIBackend, _DirectClient
from studies.germany_auto_market.model import (BRAND_ATTR, BRANDS, BYD0, EV0, TSHOCK0, W_AVAIL,
                                               W_EV, W_OPEN, W_TESLA, _gumbel, byd_ramp,
                                               ev_climate, real_share, sample_buyer, tesla_shock)

BRAND_NAME = {
    "vw": "Volkswagen", "mercedes": "Mercedes-Benz", "bmw": "BMW", "audi": "Audi",
    "skoda": "Skoda", "seat_cupra": "SEAT/Cupra", "opel": "Opel", "toyota": "Toyota",
    "hyundai": "Hyundai", "renault": "Renault", "tesla": "Tesla", "byd": "BYD",
    "other": "another brand",
}
PURE_EV = ("tesla", "byd")
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
W_INC = 2.0     # utility bonus of the incentive for a fully EV-affine buyer and an all-electric brand


STATE_NAME = {
    "Baden-Wuerttemberg": "Baden-Wurttemberg", "Bayern": "Bavaria", "Berlin": "Berlin",
    "Brandenburg": "Brandenburg", "Bremen": "Bremen", "Hamburg": "Hamburg", "Hessen": "Hesse",
    "Mecklenburg-Vorpommern": "Mecklenburg-Western Pomerania", "Niedersachsen": "Lower Saxony",
    "Nordrhein-Westfalen": "North Rhine-Westphalia", "Rheinland-Pfalz": "Rhineland-Palatinate",
    "Saarland": "Saarland", "Sachsen": "Saxony", "Sachsen-Anhalt": "Saxony-Anhalt",
    "Schleswig-Holstein": "Schleswig-Holstein", "Thueringen": "Thuringia",
}


def agent_id(i: int) -> str:
    return f"de-{i:03d}"


def month_label(m: int) -> str:
    """Month index 0 = 2024-01."""
    return f"{MONTHS[m % 12]} {2024 + m // 12}"


def is_cjk(text: str) -> bool:
    return re.search(r"[぀-ヿ㐀-鿿가-힯]", text or "") is not None


# --- P ----------------------------------------------------------------------------------------

@register("population", "adacall_de.pop")
class BuyerPopulationProvider(PopulationProvider):
    def __init__(self, bundle: PopulationBundle):
        self.bundle = bundle
        self.n = int(bundle.provider_args.get("n_agents", 100))

    def build(self, seed: int) -> list[Persona]:
        out = []
        for i in range(self.n):
            b = sample_buyer(i, seed, self.n)
            out.append(Persona(agent_id=agent_id(i), group_key=b["demographics"]["bundesland_name"],
                               attributes=b, init_state={"choice": b["prior_brand"]}))
        return out

    def interaction_structure(self) -> InteractionStructure:
        return self.bundle.interaction


# --- E ----------------------------------------------------------------------------------------

@register("environment", "adacall_de.env")
class BuyerEnvironmentProvider(EnvironmentProvider):
    """Owns every buyer's current brand. Step t is calendar month ``month_offset + t``.
    Market news (audience "all") reaches everyone; the "incentive" channel reaches the audience it
    names: "all", {"bundesland": [...]} or {"agent_ids": [...]}."""

    def __init__(self, bundle: EnvironmentBundle):
        self.bundle = bundle
        self.n = int(bundle.provider_args.get("n_agents", 100))
        self.month_offset = int(bundle.provider_args.get("month_offset", 12))
        self.buyers: dict[str, dict[str, Any]] = {}
        self.choice: dict[str, str] = {}
        self.news: list[str] = []
        self.incentive: dict[str, str] = {}      # agent -> incentive text active this step
        self.month = self.month_offset

    def reset(self, seed: int) -> None:
        self.buyers = {agent_id(i): sample_buyer(i, seed, self.n) for i in range(self.n)}
        self.choice = {aid: b["prior_brand"] for aid, b in self.buyers.items()}
        self.news, self.incentive, self.month = [], {}, self.month_offset

    def _audience(self, audience: Any) -> list[str]:
        if audience == "all":
            return list(self.buyers)
        if isinstance(audience, dict) and "bundesland" in audience:
            states = set(audience["bundesland"])
            return [a for a, b in self.buyers.items()
                    if b["demographics"]["bundesland_name"] in states]
        if isinstance(audience, dict):
            return [a for a in audience.get("agent_ids", []) if a in self.buyers]
        return []

    def advance_to(self, t: int) -> list[ScheduledEvent]:
        self.month = self.month_offset + t
        self.news, self.incentive = [], {}
        fired: list[ScheduledEvent] = []
        for b in self.bundle.information_program.broadcasts:
            if not (b.at_step <= t < b.at_step + b.ttl):
                continue
            if b.channel != "incentive":
                self.news.append(b.content)
                continue
            ids = self._audience(b.audience)
            for aid in ids:
                self.incentive[aid] = b.content
            if b.at_step == t:
                fired.append(ScheduledEvent(at_step=t, target_layer="incentive", op="set",
                                            value=len(ids),
                                            note=f"{b.message_id} reaches {len(ids)} buyers"))
        return fired

    def observe_batch(self, agent_ids: list[str], t: int, round_idx: int = 0) -> list[Observation]:
        m = self.month
        macro = {"month": m, "ev_climate": ev_climate(m), "byd_ramp": byd_ramp(m),
                 "tesla_shock": tesla_shock(m)}
        out = []
        for aid in agent_ids:
            inc = self.incentive.get(aid)
            out.append(Observation(
                agent_id=aid, step=t,
                local_physical={"profile": self.buyers[aid], "choice": self.choice[aid]},
                macro_physical=macro, macro_information={"news": list(self.news)},
                local_information={"incentive": inc} if inc else {}))
        return out

    def apply(self, actions: list[Action]) -> None:
        for a in actions:
            b = (a.payload or {}).get("choice")
            if a.agent_id in self.buyers and b in BRAND_ATTR:
                self.choice[a.agent_id] = b

    def agent_state(self, aid: str) -> dict:
        d = self.buyers[aid]["demographics"]
        return {"bundesland": d["bundesland_name"], "choice": self.choice[aid],
                "exposed": aid in self.incentive}

    def snapshot(self) -> dict:
        return {"month": self.month, "n_exposed": len(self.incentive)}


# --- B ----------------------------------------------------------------------------------------

PROMPT = """You are an adult in Germany who is thinking about buying a new car. Stay in this role and use only the information below to pick the brand you would most likely buy this month.

About you:
- You live in {state} ({urbanicity}). Age group {age}, {gender}, {education} education, {employment}.
- Household: {household}; public-transport access: {pt}.
- Price sensitivity {ps:.2f}, preference for premium brands {prem:.2f}, openness to electric cars {ev:.2f}, preference for German brands {home:.2f}, openness to new brands {open:.2f} (all from 0 = low to 1 = high).
- The brand you drive now or last leaned towards: {current}. Germans replace cars rarely and most stay with their brand unless they have a clear reason to switch.

The market in {month}:
- {climate}
{news}
Choose one brand key: {menu}.
Output one line of JSON only: {{"choice": "<brand key>", "reason": "<one casual first-person sentence>"}}"""

_WORDS = {"large_city": "a large city", "medium_city": "a medium-sized city",
          "small_town": "a small town", "rural": "a rural area", "low": "low", "medium": "medium",
          "high": "high", "employed": "employed", "unemployed": "unemployed",
          "inactive": "not in the labour force", "single": "living alone",
          "couple_no_child": "a couple without children", "couple_with_child": "a family with children",
          "single_parent": "a single parent", "multi_adult": "several adults"}


def build_prompt(ob: Observation) -> str:
    p = ob.local_physical["profile"]
    d = p["demographics"]
    w = lambda k: _WORDS.get(d[k], str(d[k]).replace("_", " "))  # noqa: E731
    lines = [f"- {n}" for n in ob.macro_information.get("news", [])]
    inc = ob.local_information.get("incentive")
    if inc:
        lines.append(f"- {inc}")
    ev = ob.macro_physical["ev_climate"]
    climate = "Overall demand for battery-electric cars in Germany is " + (
        "weak." if ev < 0.4 else "moderate." if ev < 0.55 else "strong.")
    return PROMPT.format(
        state=STATE_NAME.get(d["bundesland_name"], d["bundesland_name"]),
        urbanicity=w("urbanicity"), age=d["age_band"], gender=d["gender"],
        education=w("education"), employment=w("employment_status"),
        household=w("household_type"), pt=w("pt_access"), ps=p["price_sensitivity"],
        prem=p["premium_affinity"], ev=p["ev_affinity"], home=p["home_bias"],
        open=p["openness"], current=BRAND_NAME[ob.local_physical["choice"]],
        month=month_label(ob.macro_physical["month"]), climate=climate,
        news="\n".join(lines) if lines else "- No special news this month.",
        menu=", ".join(f"{b} ({BRAND_NAME[b]})" for b in BRANDS))


_SYN = {b: [b, BRAND_NAME[b].lower()] for b in BRANDS}
_SYN["vw"] += ["volkswagen", "vw"]
_SYN["mercedes"] += ["mercedes", "benz"]
_SYN["seat_cupra"] += ["seat", "cupra"]
_SYN["other"] = ["other"]
_ORDER = sorted(BRANDS, key=lambda b: b != "seat_cupra")   # "seat_cupra" before shorter keys


def parse_reply(resp: str) -> dict | None:
    m = re.search(r'"choice"\s*:\s*"([^"]*)"', resp or "")
    if m is None:
        return None
    raw = m.group(1).strip().lower()
    choice = next((b for b in _ORDER if raw == b), None) or next(
        (b for b in _ORDER if any(s in raw for s in _SYN[b])), None)
    if choice is None:
        return None
    r = re.search(r'"reason"\s*:\s*"([^"]*)"', resp)
    return {"choice": choice, "reason": r.group(1)[:120] if r else ""}


def utilities(ob: Observation, w_inc: float = W_INC, stay: float = 1.0) -> dict[str, float]:
    """Random-utility rule: the month's real KBA share as baseline appeal, the reference study's
    EV / BYD / Tesla deviation terms, brand inertia and the incentive bonus for exposed buyers."""
    p, env = ob.local_physical["profile"], ob.macro_physical
    m, cur = env["month"], ob.local_physical["choice"]
    inc = bool(ob.local_information.get("incentive"))
    u = {}
    for b in BRANDS:
        a = BRAND_ATTR[b]
        x = math.log(max(real_share(m, b), 1e-4))
        x += W_EV * p["ev_affinity"] * a["ev"] * (env["ev_climate"] - EV0)
        if b == "byd":
            ramp = env["byd_ramp"] - BYD0
            x += W_AVAIL * ramp + W_OPEN * ramp * p["openness"]
        if b == "tesla":
            x += W_TESLA * (env["tesla_shock"] - TSHOCK0) * (0.6 + 0.4 * p["openness"])
        if inc:
            x += w_inc * p["ev_affinity"] * (a["ev"] if a["ev"] >= 1.0 else 0.0)
        if b == cur:
            x += stay
        u[b] = x
    return u


class StandInBackend:
    """Free LLM stand-in: argmax of the rule utilities plus the agent's persistent taste noise and a
    per-(namespace, agent, step, prompt) Gumbel draw of scale ``sigma``, as a JSON reply."""

    def __init__(self, sigma: float = 0.5):
        self.sigma = sigma

    def __call__(self, prompt: str, *, ns: str = "", agent_id: str = "", step: int = 0,
                 utilities: dict[str, float] | None = None, **_ctx) -> str:
        h = hashlib.blake2b(repr((ns, agent_id, step, prompt)).encode(), digest_size=8).hexdigest()
        u = utilities or {b: 0.0 for b in BRANDS}
        pick = max(BRANDS, key=lambda b: u[b] + _gumbel(f"{agent_id}|{b}")
                   + self.sigma * _gumbel(f"{h}|{b}"))
        return json.dumps({"choice": pick, "reason": "stand-in"})


@register("decision", "adacall_de.decision")
class BuyerDecision(DecisionModel):
    def __init__(self, llm_kind: str = "scripted", model: str | None = None,
                 temperature: float = 0.7, max_tokens: int = 80, max_workers: int = 32,
                 sigma: float = 0.5, rpm: int | None = None, w_inc: float = W_INC):
        self.llm_kind, self.w_inc = llm_kind, w_inc
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

    def rule(self, ob: Observation) -> str:
        u = utilities(ob, self.w_inc)
        return max(BRANDS, key=lambda b: u[b] + _gumbel(f"{ob.agent_id}|{b}"))

    def _ask(self, ob: Observation) -> str:
        return self.client(build_prompt(ob), agent_id=ob.agent_id, step=ob.step,
                           utilities=utilities(ob, self.w_inc))

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
                choice, reason, source = dec["choice"], dec["reason"], "llm"
            else:   # unparsable reply: keep the current brand (the reference study's fallback)
                choice = ob.local_physical["choice"] if self.client is not None else self.rule(ob)
                reason, source = "", "fallback" if self.client is not None else "rule"
            actions.append(Action(agent_id=ob.agent_id, step=ob.step, kind="choose_brand",
                                  payload={"choice": choice, "reason": reason}, source=source))
        return actions


# --- metrics ----------------------------------------------------------------------------------

@register("collector", "adacall_de.collector")
class BuyerCollector(MetricCollector):
    def collect(self, env: Any, actions: list[Action], t: int) -> dict[str, Any]:
        ch = list(env.choice.values())
        n = len(ch) or 1
        row = {f"share_{b}": round(sum(c == b for c in ch) / n, 4) for b in BRANDS}
        row["share_pure_ev"] = round(sum(c in PURE_EV for c in ch) / n, 4)
        row["ev_index"] = round(sum(BRAND_ATTR[c]["ev"] for c in ch) / n, 4)
        row["n_exposed"] = len(env.incentive)
        return row

    def columns(self) -> list[str]:
        return [f"share_{b}" for b in BRANDS] + ["share_pure_ev", "ev_index", "n_exposed"]
