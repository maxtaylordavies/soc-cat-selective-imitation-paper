import jax.numpy as jnp
import numpyro.distributions as dist
from tqdm import tqdm
import pandas as pd

from .preliminary import simulate_imitation
from .weighting_functions.explicit_value_functions import explicit_value_funcs


def analyse_model_effectiveness(rng_key, model_name, use_2d=False, **kwargs):
    if model_name == "value_functions_known":
        compute_weights = explicit_value_funcs
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if use_2d:
        return analyse_model_effectiveness_2d(rng_key, compute_weights, **kwargs)
    return analyse_model_effectiveness_1d(rng_key, compute_weights, **kwargs)


def analyse_model_effectiveness_2d(
    rng_key, compute_weights, agents_per_phi=100, sigma=0.01, beta=0.05
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

    agents = _sample_agents(rng_key, phis, mus, cov, agents_per_phi)
    imitation_results = {"phi": [], "vx": [], "vy": [], "reward": [], "vself": []}
    weights = []

    for i, vself in enumerate(vselfs):
        # simulate imitating each agent
        for a in tqdm(agents):
            reward, _ = simulate_imitation(vself, a["v"], beta=beta, trials=1000)
            imitation_results["phi"].append(a["phi"])
            imitation_results["vx"].append(float(a["v"][0]))
            imitation_results["vy"].append(float(a["v"][1]))
            imitation_results["reward"].append(reward)
            imitation_results["vself"].append(i)

        # compute weights
        weights.append(
            compute_weights(agents, phis, vself, type="fit_distribution", rng_key=rng_key)
        )

    return pd.DataFrame(imitation_results), weights, phis, vselfs


def analyse_model_effectiveness_1d(
    rng_key, compute_weights, num_phis=5, agents_per_phi=100, sigma=0.01, beta=0.05
):
    phis = list(range(num_phis))
    mus = [jnp.array([mu]) for mu in jnp.linspace(0.1, 0.9, num_phis)]
    cov = jnp.array([[sigma]])
    vselfs = [jnp.array([0.0]), jnp.array([0.5]), jnp.array([1.0])]

    agents = _sample_agents(rng_key, phis, mus, cov, agents_per_phi)
    imitation_results = {"phi": [], "v": [], "reward": [], "vself": []}
    weights = []

    for i, vself in enumerate(vselfs):
        # simulate imitating each agent
        for a in tqdm(agents):
            reward, _ = simulate_imitation(vself, a["v"], beta=beta, trials=1000)
            imitation_results["phi"].append(a["phi"])
            imitation_results["v"].append(float(a["v"]))
            imitation_results["reward"].append(reward)
            imitation_results["vself"].append(i)

        # compute weights
        weights.append(
            compute_weights(agents, phis, vself, type="fit_distribution", rng_key=rng_key)
        )

    return pd.DataFrame(imitation_results), weights, phis, vselfs


def _sample_agents(rng_key, phis, mus, cov, agents_per_phi):
    agents = []
    for phi in phis:
        value_funcs = dist.MultivariateNormal(mus[phi], cov).sample(
            rng_key, (agents_per_phi,)
        )
        agents.extend([{"phi": phi, "v": v} for v in value_funcs])
    return agents
