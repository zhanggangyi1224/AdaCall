"""Bundle factory for adacall_germany: the control study plus its incentive variants.

The control arm carries only the market news of the reference study (translated to English). A
treatment arm adds one EV purchase incentive Broadcast at ``t_star`` that lasts to the end of the
run, aimed at one or more federal states or at everyone; all environment entries before ``t_star``
are identical, which is the SocioVerse2 branch invariant.
"""

from __future__ import annotations

from socioverse.schemas import EnvironmentBundle, PopulationBundle, SimulationConfig, StudySpec
from socioverse.schemas.environment import Broadcast, EnvironmentLayer, InformationProgram
from socioverse.schemas.population import InteractionStructure, PropagationMode

from .model import BRANDS, STATE_NAME

STUDY_ID = "adacall_germany"
METRICS = [f"share_{b}" for b in BRANDS] + ["share_pure_ev", "ev_index", "n_exposed"]

# (message id, first month, last month exclusive, text); month 0 = 2024-01, as in the reference
NEWS = [
    ("SUBSIDY", 0, 6, "Policy aftermath: the federal EV subsidy (Umweltbonus) ended abruptly in "
                      "December 2023, so battery-electric cars cost more and demand is under pressure."),
    ("BYD_ENTRY", 6, 16, "Market news: BYD is speeding up its entry into Germany, expanding its "
                         "dealer network and launching several good-value models."),
    ("TESLA_PRESS", 10, 22, "Public mood: controversy around the Tesla brand is growing and model "
                            "updates are slowing; some buyers are moving to other electric brands."),
    ("EV_RECOVER", 14, 29, "Industry trend: carmakers are launching more affordable electric cars to "
                           "meet EU CO2 fleet targets, and demand for electric cars is recovering."),
    ("BYD_EXPAND", 16, 29, "Market news: BYD sales in Germany are climbing fast, with more showrooms, "
                           "more models and very competitive prices."),
    ("TESLA_REFRESH", 24, 29, "Product news: Tesla's refreshed models are on sale and are drawing "
                              "some electric-car buyers back."),
]

def incentive_text(states: list[str] | None, bonus_eur: int) -> str:
    if states is None:
        return (f"National news: the federal government has announced a purchase bonus of "
                f"EUR {bonus_eur:,} for new battery-electric cars, starting this month.")
    names = " and ".join(STATE_NAME.get(s, s) for s in states)
    return (f"Regional news: the state government of {names} has announced a purchase bonus of "
            f"EUR {bonus_eur:,} for new battery-electric cars bought by residents, starting this month.")


def make_bundles(*, study_id: str = STUDY_ID, n_agents: int = 100, n_steps: int = 10,
                 t_star: int = 3, month_offset: int = 12, target: str | list[str] | None = None,
                 bonus_eur: int = 4000, decision_args: dict | None = None, seed: int = 42
                 ) -> tuple[StudySpec, EnvironmentBundle, PopulationBundle, SimulationConfig]:
    """``target=None`` builds the control arm, ``"all"`` a nationwide incentive, and a state name
    or list of state names (as in the population data, e.g. "Bayern") a regional one."""
    study = StudySpec(
        study_id=study_id,
        title="AdaCall testbed: car-brand choice under a regional EV purchase incentive",
        research_question=("How much does an EV purchase bonus in one federal state raise the share of "
                           "buyers choosing an all-electric brand, compared with a nationwide bonus?"),
        hypothesis=("Without peer effects only the targeted buyers' prompts change, so a paired "
                    "treatment branch pays for the targeted share of agents only."),
        n_steps=n_steps, seed=seed, time_unit="month",
        step_meaning=f"one month of brand choice (step 0 = month {month_offset}, 0 = 2024-01)",
        metrics=METRICS, display_metrics=["share_pure_ev", "ev_index", "share_vw"],
        metric_descriptions={
            "share_pure_ev": "Share of buyers choosing an all-electric brand (Tesla or BYD)",
            "ev_index": "Mean EV orientation of the chosen brands (0.1 weak .. 1.0 all-electric)",
            "n_exposed": "Buyers who saw the incentive this month",
            **{f"share_{b}": f"Share of buyers choosing {b}" for b in BRANDS},
        },
        domain="consumer-behavior",
        tags=["auto-market", "brand-choice", "ev-incentive", "counterfactual", "adacall"],
        legacy_simulator="from_scratch",
        provider_refs=["adacall_de.env", "adacall_de.pop", "adacall_de.decision",
                       "adacall_de.collector"],
        adjustable_params=["n_agents", "target states", "bonus_eur", "t_star", "llm_kind"],
        status="draft", maintainer="Antarctic penguins",
        demonstrates=["real-population-sampling", "info-broadcast", "branch-replay"],
        teaches="paired, divergence-triggered LLM calls when a treatment has no spillover",
    )
    broadcasts = []
    for mid, start, stop, text in NEWS:
        a, b = max(start - month_offset, 0), min(stop - month_offset, n_steps + 1)
        if a < b:
            broadcasts.append(Broadcast(message_id=mid, channel="market_news", content=text,
                                        at_step=a, ttl=b - a, audience="all"))
    if target is not None:
        states = None if target == "all" else ([target] if isinstance(target, str) else list(target))
        broadcasts.append(Broadcast(
            message_id="EV_BONUS", channel="incentive", content=incentive_text(states, bonus_eur),
            at_step=t_star, ttl=n_steps - t_star + 1,
            audience="all" if states is None else {"bundesland": states},
            note=f"EV purchase bonus, target {target}"))
    env = EnvironmentBundle(
        study_id=study_id, provider_ref="adacall_de.env",
        provider_args={"n_agents": n_agents, "month_offset": month_offset},
        layers=[
            EnvironmentLayer(name="market_news", modality="information", scope="macro",
                             dynamics="scheduled", description="market and policy news"),
            EnvironmentLayer(name="incentive", modality="information", scope="local",
                             dynamics="scheduled", description="EV purchase bonus, by state"),
            EnvironmentLayer(name="ev_climate", modality="physical", scope="macro",
                             dynamics="scheduled", description="EV demand climate (real data)"),
        ],
        information_program=InformationProgram(broadcasts=broadcasts),
    )
    pop = PopulationBundle(study_id=study_id, provider_ref="adacall_de.pop",
                           provider_args={"n_agents": n_agents},
                           interaction=InteractionStructure(kind="none"),
                           propagation=PropagationMode.INDEPENDENT)
    sim = SimulationConfig(study_id=study_id, n_steps=n_steps, seed=seed,
                           decision_ref="adacall_de.decision",
                           decision_args=decision_args or {"llm_kind": "scripted"},
                           collector_ref="adacall_de.collector", interaction_rounds=1)
    return study, env, pop, sim
