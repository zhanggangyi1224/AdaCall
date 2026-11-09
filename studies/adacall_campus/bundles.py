"""Bundle factory for adacall_campus: the control study plus its subsidy variants.

Prices follow the reference schedule (canteen x1.08, delivery x1.06, cooking x1.02 per month). A
treatment arm adds one canteen subsidy Broadcast at ``t_star`` that lasts to the end of the run,
aimed at a set of dorms or at everyone; all entries before ``t_star`` are identical, which is the
SocioVerse2 branch invariant.
"""

from __future__ import annotations

import random

from socioverse.schemas import (EnvironmentBundle, PopulationBundle, ScheduledEvent,
                                SimulationConfig, StudySpec)
from socioverse.schemas.environment import Broadcast, EnvironmentLayer, InformationProgram
from socioverse.schemas.population import InteractionStructure, PropagationMode
from studies.campus_dining_choice.model import dorm_members

STUDY_ID = "adacall_campus"
METRICS = ["share_canteen", "share_delivery", "share_cook", "mean_satisfaction", "n_subsidised"]
NEWS = [
    ("CANTEEN_NOTICE", "Campus canteen notice: because of rising costs, canteen prices will go up "
                       "every month for the next few months (about 8% a month)."),
    ("DELIVERY_NEWS", "Local food delivery: platform delivery fees and order prices keep rising "
                      "(about 6% a month) and discounts are shrinking."),
]


def target_dorms(n_agents: int, k: int, seed: int = 42, target_seed: int = 11) -> list[int]:
    """The k dorms that receive the subsidy (seeded, reproducible, recorded in the bundle)."""
    dorms = sorted(dorm_members(n_agents, seed))
    return sorted(random.Random(target_seed).sample(dorms, k))


def make_bundles(*, study_id: str = STUDY_ID, n_agents: int = 100, n_steps: int = 6,
                 t_star: int = 2, target: str | int | None = None, subsidy_yuan: float = 4.0,
                 cohort_round: float = 0.1, decision_args: dict | None = None, seed: int = 42,
                 target_seed: int = 11
                 ) -> tuple[StudySpec, EnvironmentBundle, PopulationBundle, SimulationConfig]:
    """``target=None`` builds the control arm, ``"all"`` a campus-wide subsidy and an integer k a
    subsidy for k seeded dorms of six students."""
    study = StudySpec(
        study_id=study_id,
        title="AdaCall testbed: dining choice under a dorm-targeted canteen subsidy",
        research_question=("How much does a canteen subsidy for a few dorms raise the canteen share "
                           "while canteen and delivery prices rise every month?"),
        hypothesis=("Subsidised students switch to the canteen; others react only when the rounded "
                    "share of their archetype moves, so a paired branch pays mostly for targets."),
        n_steps=n_steps, seed=seed, time_unit="month", step_meaning="one month of dining choice",
        metrics=METRICS, display_metrics=METRICS[:4],
        metric_descriptions={
            "share_canteen": "Share of students eating mainly in the canteen",
            "share_delivery": "Share of students mainly ordering food delivery",
            "share_cook": "Share of students mainly cooking for themselves",
            "mean_satisfaction": "Mean self-rated satisfaction with eating this month (0-1)",
            "n_subsidised": "Students receiving the canteen subsidy this month",
        },
        domain="consumer-behavior",
        tags=["dining-choice", "price-response", "subsidy", "counterfactual", "adacall"],
        legacy_simulator="from_scratch",
        provider_refs=["adacall_cd.env", "adacall_cd.pop", "adacall_cd.decision",
                       "adacall_cd.collector"],
        adjustable_params=["n_agents", "target dorms", "subsidy_yuan", "cohort_round", "t_star",
                           "llm_kind"],
        status="draft", maintainer="Antarctic penguins",
        demonstrates=["info-broadcast", "branch-replay"],
        teaches="paired, divergence-triggered LLM calls when a treatment spills over within groups",
    )
    events = [ScheduledEvent(at_step=t, target_layer=f"{key}_price", op="multiply",
                             property_name="price", value=v, note=f"month {t}: {key} price x{v}")
              for t in range(1, n_steps + 1)
              for key, v in (("canteen", 1.08), ("delivery", 1.06), ("cook", 1.02))]
    broadcasts = [Broadcast(message_id=mid, channel="news", content=text, at_step=0,
                            ttl=n_steps + 1, audience="all") for mid, text in NEWS]
    if target is not None:
        if target == "all":
            audience, who = "all", "all students"
        else:
            audience = {"dorms": target_dorms(n_agents, int(target), seed, target_seed)}
            who = "students in your dorm"
        broadcasts.append(Broadcast(
            message_id="CANTEEN_SUBSIDY", channel="subsidy", at_step=t_star,
            ttl=n_steps - t_star + 1, audience=audience,
            content=(f"University pilot: from this month, {who} get a canteen subsidy of "
                     f"{subsidy_yuan:.0f} yuan per meal (already included in the canteen price above)."),
            note=f"canteen subsidy, target {target}"))
    env = EnvironmentBundle(
        study_id=study_id, provider_ref="adacall_cd.env",
        provider_args={"n_agents": n_agents, "subsidy_yuan": subsidy_yuan,
                       "cohort_round": cohort_round},
        layers=[
            *[EnvironmentLayer(name=f"{key}_price", modality="physical", scope="macro",
                               dynamics="scheduled", description=f"per-meal price: {key}")
              for key in ("canteen", "delivery", "cook")],
            EnvironmentLayer(name="news", modality="information", scope="macro",
                             dynamics="scheduled", description="canteen and delivery price news"),
            EnvironmentLayer(name="subsidy", modality="physical", scope="local",
                             dynamics="scheduled", description="canteen subsidy, by dorm"),
            EnvironmentLayer(name="cohort_trend", modality="information", scope="local",
                             dynamics="endogenous",
                             description="last month's rounded choice shares of the archetype"),
        ],
        scheduled_events=events, information_program=InformationProgram(broadcasts=broadcasts),
    )
    pop = PopulationBundle(study_id=study_id, provider_ref="adacall_cd.pop",
                           provider_args={"n_agents": n_agents},
                           interaction=InteractionStructure(kind="none"),
                           propagation=PropagationMode.INDEPENDENT)
    sim = SimulationConfig(study_id=study_id, n_steps=n_steps, seed=seed,
                           decision_ref="adacall_cd.decision",
                           decision_args=decision_args or {"llm_kind": "scripted"},
                           collector_ref="adacall_cd.collector", interaction_rounds=1)
    return study, env, pop, sim
