from jax import random
import jax.numpy as jnp
from tqdm import tqdm
import pandas as pd

from .preliminary import simulate_choices, simulate_trajectories, simulate_imitation
from .weighting_functions.explicit_value_functions import value_funcs_known
from .weighting_functions.full_bayesian import full_bayesian
from ..utils import v_domain_2d


MODELS = {
    "value functions known": value_funcs_known,
    "full bayesian": full_bayesian,
}


def analyse_model_effectiveness(
    rng_key,
    model_names,
    obs_history,
    K=5,
    beta=0.1,
    plot_dir="results/tmp",
):
    phis = jnp.arange(K)
    v_selfs = jnp.array(
        [
            [0.5, 0.5],
            [1.0, 0.5],
            [0.5, 1.0],
            [1.0, 1.0],
        ]
    )
    v_domain = v_domain_2d()

    imitation_results = {"phi": [], "vx": [], "vy": [], "reward": [], "vself": []}
    weights = {model_name: [] for model_name in model_names}

    for i, v_self in enumerate(v_selfs):
        # simulate imitating each agent
        for agent in tqdm(obs_history, desc="simulating imitation"):
            reward = simulate_imitation(v_self, agent["choices"])
            imitation_results["phi"].append(int(agent["phi"]))
            imitation_results["vx"].append(float(agent["v"][0]))
            imitation_results["vy"].append(float(agent["v"][1]))
            imitation_results["reward"].append(float(reward))
            imitation_results["vself"].append(i)

        # compute weights
        for model_name in model_names:
            weights[model_name].append(
                MODELS[model_name](
                    rng_key, obs_history, beta, v_self, v_domain, plot_dir=plot_dir
                )
            )

    return pd.DataFrame(imitation_results), weights, phis, v_selfs


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
    keys = random.split(rng_key, num_agents)
    for m in tqdm(range(num_agents), desc="sampling behaviour"):
        v = random.multivariate_normal(keys[m], mus[z[m]], jnp.diag(sigmas[z[m]]))
        choices = simulate_choices(keys[m], v, beta=beta, N=num_trials)
        agents.append({"phi": z[m], "v": v, "choices": choices})

    return agents
