from jax import random
import jax.numpy as jnp
from tqdm import tqdm
import pandas as pd

from .preliminary import simulate_choices, simulate_trajectories, simulate_imitation
from .weighting_functions.explicit_value_functions import (
    value_funcs_known,
    value_funcs_inferred_individual,
    value_funcs_inferred_group,
)


MODELS = {
    "value functions known": value_funcs_known,
    "value functions inferred (individual)": value_funcs_inferred_individual,
    "value functions inferred (group)": value_funcs_inferred_group,
}


def analyse_model_effectiveness(rng_key, model_names, use_2d=False, **kwargs):
    for model_name in model_names:
        if model_name not in MODELS:
            raise ValueError(f"Unknown model name: {model_name}")

    if use_2d:
        return analyse_model_effectiveness_2d(rng_key, model_names, **kwargs)
    return analyse_model_effectiveness_1d(rng_key, model_names, **kwargs)


def analyse_model_effectiveness_2d(
    rng_key, model_names, agents_per_phi=100, sigma=0.01, beta=0.05, c=0.1, num_traj=100
):
    phis = list(range(5))
    mus = [
        jnp.array([0.1, 0.1]),
        jnp.array([0.1, 0.9]),
        jnp.array([0.5, 0.5]),
        jnp.array([0.9, 0.1]),
        jnp.array([0.9, 0.9]),
    ]
    cov = jnp.array([[sigma, 0.0], [0.0, sigma]])
    vselfs = [
        jnp.array([0.5, 0.5]),
        jnp.array([1.0, 0.5]),
        jnp.array([0.5, 1.0]),
        jnp.array([1.0, 1.0]),
    ]

    agents = generate_behaviour(
        rng_key, phis, mus, cov, agents_per_phi, num_traj, beta=beta, c=c
    )
    imitation_results = {"phi": [], "vx": [], "vy": [], "reward": [], "vself": []}
    weights = {model_name: [] for model_name in model_names}

    for i, vself in enumerate(vselfs):
        # simulate imitating each agent
        for a in tqdm(agents):
            reward, _ = simulate_imitation(rng_key, vself, a["v"], beta=beta, trials=1000)
            imitation_results["phi"].append(a["phi"])
            imitation_results["vx"].append(float(a["v"][0]))
            imitation_results["vy"].append(float(a["v"][1]))
            imitation_results["reward"].append(reward)
            imitation_results["vself"].append(i)

        # compute weights
        for model_name in model_names:
            weights[model_name].append(MODELS[model_name](rng_key, agents, phis, vself))

    return pd.DataFrame(imitation_results), weights, phis, vselfs


def analyse_model_effectiveness_1d(
    rng_key,
    model_names,
    num_phis=5,
    agents_per_phi=100,
    sigma=0.01,
    beta=0.05,
    c=0.1,
    num_traj=100,
):
    phis = list(range(num_phis))
    mus = [jnp.array([mu]) for mu in jnp.linspace(0.1, 0.9, num_phis)]
    sigmas = [jnp.array([sigma, sigma]) for _ in range(num_phis)]
    vselfs = [jnp.array([0.0]), jnp.array([0.5]), jnp.array([1.0])]

    agents = generate_behaviour(
        rng_key,
        phis,
        mus,
        sigmas,
        agents_per_phi,
        num_traj,
        beta=beta,
        c=c,
    )
    imitation_results = {"phi": [], "v": [], "reward": [], "vself": []}
    weights = {model_name: [] for model_name in model_names}

    for i, vself in enumerate(vselfs):
        # simulate imitating each agent
        for a in tqdm(agents, desc=f"simulating imitation for vself={vself}"):
            reward, _ = simulate_imitation(rng_key, vself, a["v"], beta=beta, trials=100)
            imitation_results["phi"].append(a["phi"])
            imitation_results["v"].append(float(a["v"]))
            imitation_results["reward"].append(reward)
            imitation_results["vself"].append(i)

        # compute weights
        for model_name in model_names:
            weights[model_name].append(MODELS[model_name](rng_key, agents, phis, vself))

    return pd.DataFrame(imitation_results), weights, phis, vselfs


def generate_behaviour(
    rng_key, weights, mus, sigmas, num_agents, num_trials, beta=0.01, c=0.1
):
    agents = []
    z = random.categorical(rng_key, weights, shape=(num_agents,))
    for m in tqdm(range(num_agents), desc="sampling behaviour"):
        v = random.multivariate_normal(rng_key, mus[z[m]], jnp.diag(sigmas[z[m]]))
        choices, trajs = simulate_trajectories(rng_key, v, c=c, beta=beta, N=num_trials)
        agents.append({"phi": z[m], "v": v, "choices": choices, "trajs": trajs})

    return agents


def generate_behaviour_simple(
    rng_key, weights, mus, sigmas, num_agents, num_trials, beta=0.01
):
    agents = []
    z = random.categorical(rng_key, weights, shape=(num_agents,))
    for m in tqdm(range(num_agents), desc="sampling behaviour"):
        v = random.multivariate_normal(rng_key, mus[z[m]], jnp.diag(sigmas[z[m]]))
        choices = simulate_choices(rng_key, v, beta=beta, N=num_trials)
        agents.append({"phi": z[m], "v": v, "choices": choices})

    return agents
