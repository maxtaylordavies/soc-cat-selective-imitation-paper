import argparse
from copy import deepcopy

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
    args: argparse.Namespace,
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
        obs_hist = deepcopy(obs_history)
        obs_new = deepcopy(new_obs)
        # _phi_self = None if ogl == "hidden" else phi_self
        # if ogl == "mismatched":
        #     for i, a in enumerate(obs_hist):
        #         if a["phi"] is not None:
        #             obs_hist[i]["phi"] = 1 - a["phi"]
        #     for i, a in enumerate(obs_new):
        #         if a["phi"] is not None:
        #             obs_new[i]["phi"] = 1 - a["phi"]

        _phi_self = phi_self
        if ogl == "hidden":
            _phi_self = None
        elif ogl == "mismatched":
            _phi_self = 1 - phi_self

        weights = strategy(
            rng_key,
            obs_hist,
            obs_new,
            args.beta,
            v_self,
            v_domain_2d(),
            _phi_self,
            args.ingroup_strength,
            **kwargs,
        )

        results = simulate_imitation_choices(
            rng_key,
            weights,
            args.beta_self,
            args.N,
        )
        results["own group label"] = ogl
        df = pd.concat([df, results])
    return df


def analyse_strategy_humanlikeness(
    rng_key, agent_vs, obs_history, results, strat, strat_name, args, **kwargs
):
    # agents known
    if not "group" in strat_name:
        new_obs = []
        for m in range(2):
            choices = simulate_choices(rng_key, agent_vs[m], beta=args.beta, N=args.N)
            new_obs.append({"phi": None, "choices": choices})

        tmp = simulate_strategy(
            rng_key,
            strat,
            obs_history,
            new_obs,
            agent_vs[0],
            0,
            args,
            **kwargs,
        )
        tmp["strategy"] = strat_name
        tmp["agents known"] = True
        tmp["groups relevant"] = True
        results = pd.concat([results, tmp])

    # agents unknown, groups relevant
    new_obs = []
    for m in range(2):
        new_obs.append({"phi": m, "choices": None})
    tmp = simulate_strategy(
        rng_key,
        strat,
        obs_history,
        new_obs,
        agent_vs[0],
        0,
        args,
        **kwargs,
    )
    tmp["strategy"] = strat_name
    tmp["agents known"] = False
    tmp["groups relevant"] = True
    results = pd.concat([results, tmp])

    # agents unknown, groups irrelevant
    tmp = simulate_strategy(
        rng_key,
        strat,
        obs_history,
        new_obs,
        jnp.array([0.5, 0]),
        0,
        args,
        **kwargs,
    )
    tmp["strategy"] = strat_name
    tmp["agents known"] = False
    tmp["groups relevant"] = False
    results = pd.concat([results, tmp])

    return results
