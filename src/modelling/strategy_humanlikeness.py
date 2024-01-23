from jax import random
import jax.numpy as jnp
import pandas as pd

from .preliminary import simulate_choices
from ..utils import v_domain_2d


def simulate_imitation_choices(
    rng_key: random.KeyArray,
    weights: jnp.ndarray,
    beta: float,
    repeats: int,
) -> pd.DataFrame:
    results = {
        "group": [],
        "imitation": [],
    }

    p = jnp.exp(weights / beta)
    p /= jnp.sum(p)
    for choice in random.choice(rng_key, len(weights), shape=(repeats,), p=p):
        for m in range(len(weights)):
            results["group"].append(m)
            results["imitation"].append(int(choice == m))

    return pd.DataFrame(results)


def simulate_strategy(
    rng_key: random.KeyArray,
    strategy: callable,
    obs_history: list[dict],
    new_obs: list[dict],
    v_self: jnp.ndarray,
    phi_self: int,
    ingroup_strength: float,
    beta: float,
    beta_self: float,
    repeats: int,
    **kwargs,
):
    df = pd.DataFrame(
        {
            "group": [],
            "imitation": [],
            "own group label": [],
        }
    )

    for ogl in ["hidden", "arbitrary", "matched", "mismatched"]:
        _phi_self = None if ogl == "hidden" else phi_self
        if ogl == "mismatched":
            for i, a in enumerate(obs_history):
                obs_history[i]["phi"] = 1 - a["phi"]
            for i, a in enumerate(new_obs):
                new_obs[i]["phi"] = 1 - a["phi"]

        weights = strategy(
            rng_key,
            obs_history,
            new_obs,
            beta,
            v_self,
            v_domain_2d(),
            _phi_self,
            ingroup_strength,
            **kwargs,
        )
        results = simulate_imitation_choices(
            rng_key,
            weights,
            beta_self,
            repeats,
        )

        results["own group label"] = ogl
        df = pd.concat([df, results])
    return df


def analyse_strategy_humanlikeness(
    rng_key, obs_history, results, strat, strat_name, args, **kwargs
):
    # # agents known
    v_self = jnp.array([0.0, 0.5])
    # new_obs = []
    # for m in range(2):
    #     choices = simulate_choices(rng_key, agent_vs[m], beta=beta, N=N)
    #     new_obs.append({"phi": None, "choices": choices})

    # tmp = simulate_strategy(
    #     rng_key,
    #     strategy,
    #     obs_history,
    #     new_obs,
    #     v_self,
    #     0,
    #     0.75,
    #     beta,
    #     beta_self,
    #     repeats,
    #     **kwargs,
    # )
    # tmp["strategy"] = strat_name
    # tmp["agents known"] = True
    # tmp["groups relevant"] = True
    # results = pd.concat([results, tmp])

    # agents unknown, groups relevant
    new_obs = []
    for m in range(2):
        new_obs.append({"phi": m, "choices": None})
    tmp = simulate_strategy(
        rng_key,
        strat,
        obs_history,
        new_obs,
        v_self,
        0,
        0.75,
        args.beta,
        args.beta_self,
        args.N,
        **kwargs,
    )
    tmp["strategy"] = strat_name
    tmp["agents known"] = False
    tmp["groups relevant"] = True
    results = pd.concat([results, tmp])

    # agents unknown, groups irrelevant
    v_self = jnp.array([0.5, 0])
    tmp = simulate_strategy(
        rng_key,
        strat,
        obs_history,
        new_obs,
        v_self,
        0,
        0.75,
        args.beta,
        args.beta_self,
        args.N,
        **kwargs,
    )
    tmp["strategy"] = strat_name
    tmp["agents known"] = False
    tmp["groups relevant"] = False
    results = pd.concat([results, tmp])

    return results
