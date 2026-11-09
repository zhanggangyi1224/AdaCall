"""adacall_campus — students' dining choice under a dorm-targeted canteen subsidy, AdaCall's
group-spillover testbed.

Derived from the campus_dining_choice reference study (same student archetypes, price schedule and
random-utility rule), with these changes:
  - an English student prompt; the reference prompt is Chinese-only;
  - one decision per month (interaction_rounds = 1), so a treatment can branch from the control run;
  - a canteen subsidy delivered as a Broadcast whose audience is a set of dorms ({"dorms": [...]}),
    "all" or a list of agent ids; exposed students see a lower canteen price in their prompt;
  - spillover runs through the group channel: each prompt shows how students of the same archetype
    chose last month, rounded to ``cohort_round`` (default 0.1). A subsidised student who switches
    changes that line for every student of the archetype only when a rounded share moves, so the
    rounding step sets how far the treatment spreads into other students' prompts;
  - the decision model calls its LLM through ``self.client(prompt, agent_id=, step=, **ctx)``,
    which the AdaCall runner replaces with a paired-cache client (adacall/cache.py).

Decision modes (``llm_kind``): ``scripted`` (the reference rule), ``standin`` (the rule with its
taste draw keyed on (namespace, agent, step, prompt), returned as a JSON reply through the client
path; free pipeline tests) and ``openai`` (see adacall_od.model.OpenAIBackend).
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from statistics import mean
from typing import Any, Callable

from socioverse.abc import DecisionModel, EnvironmentProvider, MetricCollector, PopulationProvider
from socioverse.engine import register
from socioverse.schemas import (Action, EnvironmentBundle, InteractionStructure, Observation,
                                Persona, PopulationBundle, ScheduledEvent)
from studies.adacall_od.model import OpenAIBackend, _DirectClient
from studies.campus_dining_choice.model import (MODES, TAU, _gumbel, _satisfaction, _utilities,
                                                agent_id, dorm_assignment, sample_student)

ARCHETYPE_LABEL = {
    "canteen_regular": "a canteen regular who usually eats on campus",
    "delivery_lover": "a heavy food-delivery user who values convenience",
    "budget_saver": "a budget saver on a tight budget who watches prices closely",
    "off_campus_cook": "an off-campus student with a kitchen who can cook",
    "foodie": "a foodie who wants variety and strong flavours",
}
ARCHETYPE_GROUP = {
    "canteen_regular": "canteen regulars", "delivery_lover": "food-delivery regulars",
    "budget_saver": "budget savers", "off_campus_cook": "off-campus cooks", "foodie": "foodies",
}
MODE_WORD = {"canteen": "the canteen", "delivery": "food delivery", "cook": "cooking for yourself"}


# --- P ----------------------------------------------------------------------------------------

@register("population", "adacall_cd.pop")
class StudentPopulationProvider(PopulationProvider):
    def __init__(self, bundle: PopulationBundle):
        self.bundle = bundle
        self.n = int(bundle.provider_args.get("n_agents", 100))

    def build(self, seed: int) -> list[Persona]:
        out = []
        for i in range(self.n):
            a = sample_student(i, seed, self.n)
            out.append(Persona(agent_id=agent_id(i), group_key=a["archetype"], attributes=a,
                               init_state={"choice": a["base_choice"]}))
        return out

    def interaction_structure(self) -> InteractionStructure:
        return self.bundle.interaction


# --- E ----------------------------------------------------------------------------------------

@register("environment", "adacall_cd.env")
class StudentEnvironmentProvider(EnvironmentProvider):
    """Owns prices and every student's choice and satisfaction. Prices move by ScheduledEvents as
    in the reference study; the "subsidy" channel lowers the canteen price for its audience."""

    def __init__(self, bundle: EnvironmentBundle):
        self.bundle = bundle
        pa = bundle.provider_args
        self.n = int(pa.get("n_agents", 100))
        self.cohort_round = float(pa.get("cohort_round", 0.1))
        self.subsidy_yuan = float(pa.get("subsidy_yuan", 4.0))
        self.base_prices = {"canteen": float(pa.get("base_canteen", 12.0)),
                            "delivery": float(pa.get("base_delivery", 25.0)),
                            "cook": float(pa.get("base_cook", 7.0))}
        self.prices: dict[str, float] = {}
        self.students: dict[str, dict[str, Any]] = {}
        self.dorm: dict[str, int] = {}
        self.choice: dict[str, str] = {}
        self.satis: dict[str, float] = {}
        self.cohort: dict[str, dict[str, float]] = {}
        self.news: list[str] = []
        self.subsidy: dict[str, str] = {}       # agent -> subsidy text active this step

    def reset(self, seed: int) -> None:
        self.prices = dict(self.base_prices)
        self.students = {agent_id(i): sample_student(i, seed, self.n) for i in range(self.n)}
        self.dorm = {agent_id(i): d for i, d in dorm_assignment(self.n, seed).items()}
        self.choice = {a: s["base_choice"] for a, s in self.students.items()}
        self.satis = {a: _satisfaction(s, s["base_choice"], self.prices)
                      for a, s in self.students.items()}
        self.cohort, self.news, self.subsidy = {}, [], {}

    def _audience(self, audience: Any) -> list[str]:
        if audience == "all":
            return list(self.students)
        if isinstance(audience, dict) and "dorms" in audience:
            dorms = set(audience["dorms"])
            return [a for a in self.students if self.dorm[a] in dorms]
        if isinstance(audience, dict):
            return [a for a in audience.get("agent_ids", []) if a in self.students]
        return []

    def _rounded_cohort(self) -> dict[str, dict[str, float]]:
        groups: dict[str, list[str]] = {}
        for a, s in self.students.items():
            groups.setdefault(s["archetype"], []).append(self.choice[a])
        q = self.cohort_round
        return {k: {m: round(round(sum(c == m for c in cs) / len(cs) / q) * q, 3) for m in MODES}
                for k, cs in groups.items()}

    def advance_to(self, t: int) -> list[ScheduledEvent]:
        self.cohort = self._rounded_cohort()          # last month's choices, before this month's
        fired: list[ScheduledEvent] = []
        for ev in self.bundle.scheduled_events:
            if ev.at_step == t:
                key = ev.target_layer.split("_")[0]
                if key in self.prices and ev.op == "multiply":
                    self.prices[key] = round(self.prices[key] * float(ev.value), 3)
                fired.append(ev)
        self.news, self.subsidy = [], {}
        for b in self.bundle.information_program.broadcasts:
            if not (b.at_step <= t < b.at_step + b.ttl):
                continue
            if b.channel != "subsidy":
                self.news.append(b.content)
                continue
            ids = self._audience(b.audience)
            for a in ids:
                self.subsidy[a] = b.content
            if b.at_step == t:
                fired.append(ScheduledEvent(at_step=t, target_layer="subsidy", op="set",
                                            value=len(ids),
                                            note=f"{b.message_id} reaches {len(ids)} students"))
        return fired

    def personal_prices(self, aid: str) -> dict[str, float]:
        p = dict(self.prices)
        if aid in self.subsidy:
            p["canteen"] = round(max(0.0, p["canteen"] - self.subsidy_yuan), 3)
        return p

    def observe_batch(self, agent_ids: list[str], t: int, round_idx: int = 0) -> list[Observation]:
        out = []
        for aid in agent_ids:
            s = self.students[aid]
            info = {"cohort_trend": self.cohort.get(s["archetype"])}
            if aid in self.subsidy:
                info["subsidy"] = self.subsidy[aid]
            out.append(Observation(
                agent_id=aid, step=t,
                local_physical={"profile": s, "prices": self.personal_prices(aid),
                                "choice": self.choice[aid], "satisfaction": self.satis[aid]},
                macro_information={"news": list(self.news)}, local_information=info))
        return out

    def apply(self, actions: list[Action]) -> None:
        for a in actions:
            p = a.payload or {}
            if a.agent_id in self.students and p.get("choice") in MODES:
                self.choice[a.agent_id] = p["choice"]
                if p.get("satisfaction") is not None:
                    self.satis[a.agent_id] = float(p["satisfaction"])

    def agent_state(self, aid: str) -> dict:
        return {"archetype": self.students[aid]["archetype"], "dorm": self.dorm[aid],
                "choice": self.choice[aid], "satisfaction": self.satis[aid],
                "subsidised": aid in self.subsidy}

    def snapshot(self) -> dict:
        return {"prices": dict(self.prices), "n_subsidised": len(self.subsidy)}


# --- B ----------------------------------------------------------------------------------------

PROMPT = """You are a university student. Stay in this role and use only the information below to decide your main way of eating this month.

About you:
- You are {label}.
- Monthly food budget: about {budget} yuan; you eat about {meals} main meals a month.
- Price sensitivity {ps:.2f}; how easy cooking is where you live {feas:.2f}; cooking skill {skill:.2f}; time pressure {tp:.2f} (all from 0 = low to 1 = high).
- Last month you mainly used {last} (satisfaction {sat:.2f}).

Cost per meal this month:
- Canteen: {canteen:.1f} yuan
- Food delivery: {delivery:.1f} yuan per order
- Cooking for yourself: {cook:.1f} yuan in ingredients, plus time and effort

News:
{news}
{cohort}
Choose one: canteen, delivery or cook, and rate your satisfaction with eating this month from 0 to 1.
Output one line of JSON only: {{"choice": "canteen" | "delivery" | "cook", "satisfaction": <0-1>, "reason": "<one sentence>"}}"""


def build_prompt(ob: Observation) -> str:
    s, lp = ob.local_physical["profile"], ob.local_physical
    p = lp["prices"]
    lines = [f"- {n}" for n in ob.macro_information.get("news", [])]
    if ob.local_information.get("subsidy"):
        lines.append(f"- {ob.local_information['subsidy']}")
    trend = ob.local_information.get("cohort_trend")
    cohort = ""
    if trend:
        shares = ", ".join(f"{MODE_WORD[m]} {trend[m]:.0%}" for m in MODES)
        cohort = (f"Last month, {ARCHETYPE_GROUP[s['archetype']]} like you mainly used: "
                  f"{shares} (rounded).\n")
    return PROMPT.format(
        label=ARCHETYPE_LABEL[s["archetype"]], budget=s["monthly_food_budget"],
        meals=s["meals_per_month"], ps=s["price_sensitivity"], feas=s["cook_feasibility"],
        skill=s["cook_skill"], tp=s["time_pressure"], last=MODE_WORD[lp["choice"]],
        sat=float(lp["satisfaction"]), canteen=p["canteen"], delivery=p["delivery"], cook=p["cook"],
        news="\n".join(lines) if lines else "- No special news this month.", cohort=cohort)


_SYN = {"canteen": ["canteen"], "delivery": ["delivery", "deliver", "takeout", "takeaway"],
        "cook": ["cook"]}


def parse_reply(resp: str) -> dict | None:
    m = re.search(r'"choice"\s*:\s*"([^"]*)"', resp or "")
    if m is None:
        return None
    raw = m.group(1).lower()
    choice = next((k for k, syns in _SYN.items() if any(x in raw for x in syns)), None)
    if choice is None:
        return None
    sm = re.search(r'"satisfaction"\s*:\s*"?([0-9]*\.?[0-9]+)', resp)
    r = re.search(r'"reason"\s*:\s*"([^"]*)"', resp)
    return {"choice": choice, "satisfaction": min(1.0, max(0.0, float(sm.group(1)))) if sm else None,
            "reason": r.group(1)[:120] if r else ""}


def utilities(ob: Observation) -> dict[str, float]:
    lp = ob.local_physical
    return _utilities(lp["profile"], lp["prices"], ob.local_information.get("cohort_trend"))


class StandInBackend:
    """Free LLM stand-in: the reference rule with its taste draw keyed on
    (namespace, agent, step, prompt), as a JSON reply with the rule's satisfaction."""

    def __init__(self, sigma: float = TAU):
        self.sigma = sigma

    def __call__(self, prompt: str, *, ns: str = "", agent_id: str = "", step: int = 0,
                 utilities: dict[str, float] | None = None, satisfaction: dict | None = None,
                 **_ctx) -> str:
        h = hashlib.blake2b(repr((ns, agent_id, step, prompt)).encode(), digest_size=8).hexdigest()
        u = utilities or {m: 0.0 for m in MODES}
        pick = max(MODES, key=lambda m: u[m] + self.sigma * _gumbel(f"{h}|{m}"))
        return json.dumps({"choice": pick, "satisfaction": (satisfaction or {}).get(pick, 0.5),
                           "reason": "stand-in"})


@register("decision", "adacall_cd.decision")
class StudentDecision(DecisionModel):
    def __init__(self, llm_kind: str = "scripted", model: str | None = None,
                 temperature: float = 0.7, max_tokens: int = 80, max_workers: int = 32,
                 sigma: float = TAU, rpm: int | None = None):
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

    @staticmethod
    def _sat(ob: Observation) -> dict[str, float]:
        lp = ob.local_physical
        return {m: _satisfaction(lp["profile"], m, lp["prices"]) for m in MODES}

    def rule(self, ob: Observation) -> str:
        u = utilities(ob)
        return max(MODES, key=lambda m: u[m] + TAU * _gumbel(f"{ob.agent_id}|{ob.step}|{m}"))

    def _ask(self, ob: Observation) -> str:
        return self.client(build_prompt(ob), agent_id=ob.agent_id, step=ob.step,
                           utilities=utilities(ob), satisfaction=self._sat(ob))

    def decide_batch(self, obs: list[Observation], memories: dict[str, Any]) -> list[Action]:
        if self.client is None:
            replies = [""] * len(obs)
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(obs))) as ex:
                replies = list(ex.map(self._ask, obs))
        actions = []
        for ob, resp in zip(obs, replies):
            lp = ob.local_physical
            dec = parse_reply(resp) if self.client is not None else None
            if dec is not None:
                choice, sat, source = dec["choice"], dec["satisfaction"], "llm"
                reason = dec["reason"]
            else:   # unparsable reply: keep last month's choice (the reference study's fallback)
                choice = lp["choice"] if self.client is not None else self.rule(ob)
                sat = lp["satisfaction"] if self.client is not None else self._sat(ob)[choice]
                reason, source = "", "fallback" if self.client is not None else "rule"
            if sat is None:
                sat = lp["satisfaction"]
            actions.append(Action(agent_id=ob.agent_id, step=ob.step, kind="choose_meal",
                                  payload={"choice": choice, "satisfaction": round(float(sat), 3),
                                           "reason": reason}, source=source))
        return actions


# --- metrics ----------------------------------------------------------------------------------

@register("collector", "adacall_cd.collector")
class StudentCollector(MetricCollector):
    def collect(self, env: Any, actions: list[Action], t: int) -> dict[str, Any]:
        ch = list(env.choice.values())
        n = len(ch) or 1
        row = {f"share_{m}": round(sum(c == m for c in ch) / n, 4) for m in MODES}
        row["mean_satisfaction"] = round(mean(env.satis.values()), 4)
        row["n_subsidised"] = len(env.subsidy)
        return row

    def columns(self) -> list[str]:
        return [f"share_{m}" for m in MODES] + ["mean_satisfaction", "n_subsidised"]
