"""Bundle factory for adacall_od: the control study plus its treatment variants.

A control arm has no campaign. A treatment arm adds one targeted Broadcast at ``t_star``; all
environment entries before ``t_star`` are identical, which is the SocioVerse2 branch invariant.
"""

from __future__ import annotations

import numpy as np

from socioverse.schemas import EnvironmentBundle, PopulationBundle, SimulationConfig, StudySpec
from socioverse.schemas.environment import Broadcast, EnvironmentLayer, InformationProgram
from socioverse.schemas.population import InteractionStructure, PropagationMode

from .model import agent_id, build_graph

STUDY_ID = "adacall_od"
METRICS = ["mean_opinion", "opinion_std", "frac_above_0_5", "n_exposed"]
CAMPAIGN = "Adopt the new policy - it benefits everyone."
PLACEBO = "The city library will extend its opening hours next month."   # C3 negative control


def target_ids(n_agents: int, reach: float, target_seed: int = 11) -> list[str]:
    """The k% of agents the campaign reaches (seeded, reproducible, recorded in the bundle)."""
    k = max(1, int(round(reach * n_agents)))
    idx = np.random.default_rng(target_seed).choice(n_agents, size=k, replace=False)
    return [agent_id(int(i)) for i in sorted(idx)]


def make_bundles(*, study_id: str = STUDY_ID, n_agents: int = 100, topology: str = "small_world",
                 n_steps: int = 8, t_star: int = 2, reach: float | None = None, ttl: int = 2,
                 campaign: str = CAMPAIGN,
                 decision_args: dict | None = None, seed: int = 42, graph_seed: int = 7,
                 init_seed: int = 12345, target_seed: int = 11
                 ) -> tuple[StudySpec, EnvironmentBundle, PopulationBundle, SimulationConfig]:
    """``reach=None`` builds the control arm; a float builds the treatment arm."""
    graph_args = {"n_agents": n_agents, "topology": topology, "k": 4, "p": 0.1,
                  "mean_degree": 10, "graph_seed": graph_seed, "init_seed": init_seed}
    study = StudySpec(
        study_id=study_id,
        title="AdaCall testbed: opinion diffusion under a targeted campaign",
        research_question=("How much does a campaign reaching k% of agents shift mean opinion, and "
                           "how many LLM calls does a paired treatment branch need to estimate it?"),
        hypothesis=("Local campaigns in sparse networks have small effects that single runs estimate "
                    "unstably; pairing the treatment with its control reuses most calls there."),
        n_steps=n_steps, seed=seed, time_unit="abstract",
        step_meaning="one round of opinion updates", metrics=METRICS, display_metrics=METRICS[:3],
        metric_descriptions={
            "mean_opinion": "Mean opinion (0-1, higher = more support for the policy)",
            "opinion_std": "Population standard deviation of opinions",
            "frac_above_0_5": "Share of agents with opinion >= 0.5",
            "n_exposed": "Agents who saw the campaign this step",
        },
        domain="opinion-dynamics",
        tags=["opinion-dynamics", "bounded-confidence", "network", "counterfactual", "adacall"],
        legacy_simulator="from_scratch",
        provider_refs=["adacall.env", "adacall.pop", "adacall.decision", "adacall.collector"],
        adjustable_params=["n_agents", "topology", "campaign reach", "t_star", "llm_kind"],
        status="draft", maintainer="Antarctic penguins",
        demonstrates=["from-scratch-core", "info-broadcast", "branch-replay"],
        teaches="paired, divergence-triggered LLM calls between a control run and its treatment branch",
    )
    broadcasts = []
    if reach is not None:
        broadcasts.append(Broadcast(message_id="CAMPAIGN", channel="campaign", content=campaign,
                                    at_step=t_star, ttl=ttl,
                                    audience={"agent_ids": target_ids(n_agents, reach, target_seed)},
                                    note=f"targeted campaign, reach {reach:.0%}"))
    env = EnvironmentBundle(
        study_id=study_id, provider_ref="adacall.env", provider_args=graph_args,
        layers=[
            EnvironmentLayer(name="peer_opinions", modality="physical", scope="local",
                             dynamics="endogenous", description="neighbours' current opinions"),
            EnvironmentLayer(name="campaign", modality="information", scope="local",
                             dynamics="scheduled", description="targeted campaign message"),
        ],
        information_program=InformationProgram(broadcasts=broadcasts),
    )
    g = build_graph(graph_args)
    pop = PopulationBundle(
        study_id=study_id, provider_ref="adacall.pop", provider_args=graph_args,
        interaction=InteractionStructure(kind="explicit_network",
                                         edges=[(agent_id(a), agent_id(b)) for a, b in g.edges()]),
        propagation=PropagationMode.CONTAGION,
    )
    sim = SimulationConfig(
        study_id=study_id, n_steps=n_steps, seed=seed,
        decision_ref="adacall.decision", decision_args=decision_args or {"llm_kind": "scripted"},
        collector_ref="adacall.collector", interaction_rounds=1,
    )
    return study, env, pop, sim
