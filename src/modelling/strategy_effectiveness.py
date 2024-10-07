from jax import random
import jax.numpy as jnp
from tqdm import tqdm
import pandas as pd

from .preliminary import simulate_choices, simulate_trajectories, simulate_imitation
from ..utils import v_domain_2d


def analyse_strategy_performance(
    rng_key,
    strategy_dict,
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
    weights = {name: [] for name in strategy_dict.keys()}

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
        for name, strategy in strategy_dict.items():
            weights[name].append(
                strategy(
                    rng_key, obs_history, beta, v_self, v_domain, plot_dir=plot_dir
                )
            )

    return pd.DataFrame(imitation_results), weights, phis, v_selfs


def generate_behaviour(
    rng_key,
    weights,
    mus,
    sigmas,
    num_agents,
    num_trials,
    simulation_func=None,
    beta=0.01,
    c=0.1,
    shortcut=False,
):
    agents = []
    z = random.categorical(rng_key, weights, shape=(num_agents,))
    for m in tqdm(range(num_agents), desc="sampling behaviour"):
        v = random.multivariate_normal(rng_key, mus[z[m]], jnp.diag(sigmas[z[m]]))
        if simulation_func:
            choices, trajs = simulation_func(rng_key, v, beta=beta, N=num_trials)
        elif shortcut:
            choices, trajs = simulate_choices(rng_key, v, beta=beta, N=num_trials), []
        else:
            choices, trajs = simulate_trajectories(
                rng_key, v, c=c, beta=beta, N=num_trials
            )
        agents.append({"phi": z[m], "v": v, "choices": choices, "trajs": trajs})

    return agents
