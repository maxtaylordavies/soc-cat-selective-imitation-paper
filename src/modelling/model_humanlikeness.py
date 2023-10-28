from jax import random
import jax.numpy as jnp
import pandas as pd

from .weighting_functions.baselines import indiscriminate, ingroup_bias
from .weighting_functions.full_bayesian import full_bayesian
from ..utils import v_domain_2d


def _init_results_dict():
    return {
        "agent": [],
        "similarity": [],
        "agents_known": [],
        "own_group_visible": [],
        "groups_relevant": [],
    }


def simulate_imitation_choices(
    rng_key: random.KeyArray,
    phis: jnp.ndarray,
    weights: jnp.ndarray,
    beta: float,
    repeats: int,
    agents_known: bool,
    own_group_visible: bool,
    groups_relevant: bool,
) -> pd.DataFrame:
    results = _init_results_dict()

    p = jnp.exp(weights[phis] / beta)
    p /= jnp.sum(p)
    choices = random.choice(rng_key, phis, shape=(repeats,), p=p)

    for choice in choices:
        for phi in phis:
            results["agent"].append(int(phi))
            results["similarity"].append(int(choice == phi))
            results["agents_known"].append(agents_known)
            results["own_group_visible"].append(own_group_visible)
            results["groups_relevant"].append(groups_relevant)

    return pd.DataFrame(results)


def simulate_strategy(
    rng_key: random.KeyArray,
    strategy: str,
    obs_history: list,
    target_phis: jnp.ndarray,
    v_self: jnp.ndarray,
    phi_self: int,
    ingroup_strength: float,
    beta: float,
    beta_self: float,
    repeats: int,
    plot_dir: str = "results/tmp",
):
    df = pd.DataFrame(_init_results_dict())

    v_domain = v_domain_2d()
    all_phis = jnp.unique(jnp.array([a["phi"] for a in obs_history]))

    for ogv in [True, False]:  # own group visible
        for gr in [True, False]:
            if strategy == "full bayesian":
                weights = full_bayesian(
                    rng_key,
                    obs_history,
                    beta,
                    v_self,
                    v_domain,
                    plot_dir=plot_dir,
                )
            elif strategy == "ingroup bias" and ogv:
                weights = ingroup_bias(len(all_phis), phi_self, ingroup_strength)
            else:
                weights = indiscriminate(len(all_phis))

            tmp = simulate_imitation_choices(
                rng_key,
                target_phis,
                weights,
                beta_self,
                repeats,
                agents_known=False,
                own_group_visible=ogv,
                groups_relevant=gr,
            )
            df = pd.concat([df, tmp])

    return df
