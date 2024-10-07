from datetime import datetime
import os
import time
from typing import Callable, List

import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import load_json, norm_unit_sum, make_barplots, v_domain_2d
from src.human import similarity_binary, similarity_continuous
from src.modelling import generate_behaviour
from src.modelling.strategies import individual_inference

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
AGENT_TRAJECTORIES = [
    {
        0: "1,1,1,1,1,1,1,1,1,1,1,1",
        1: "3,3,3,3,3,3,3,3,3,3,3,3",
        2: "2,2,2,2,2,2,2,2,2,2,2,2",
        3: "4,4,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2",
        4: "4,4,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2",
        5: "1,1,1,1,1,1,1,1,1,1,1,1",
        6: "3,3,3,3,3,3,3,3,3,3,3,3",
        7: "4,4,4,4,4,4,4,4,4,4,4",
        8: "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3",
        9: "",
        10: "",
    },
    {
        0: "2,2,2,2,2,2,2,2,2,2,2,2",
        1: "1,1,1,1,1,1,1,1,1,1,1,1",
        2: "3,3,3,3,3,3,3,3,3,3,3,3",
        3: "4,4,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2",
        4: "4,4,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2",
        5: "3,3,3,3,3,3,3,3,3,3,3,3",
        6: "1,1,1,1,1,1,1,1,1,1,1,1",
        7: "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3",
        8: "4,4,4,4,4,4,4,4,4,4,4",
        9: "",
        10: "",
    },
    {
        0: "",
        1: "",
        2: "",
        3: "",
        4: "",
        5: "",
        6: "",
        7: "4,4,4,4,4,4,4,4,4,4,4",
        8: "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3",
        9: "1,1,1,1,1,1,1,1,1,1,1,1",
        10: "3,3,3,3,3,3,3,3,3,3,3,3",
    },
    {
        0: "",
        1: "",
        2: "",
        3: "",
        4: "",
        5: "",
        6: "",
        7: "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3",
        8: "4,4,4,4,4,4,4,4,4,4,4",
        9: "3,3,3,3,3,3,3,3,3,3,3,3",
        10: "1,1,1,1,1,1,1,1,1,1,1,1",
    },
]
LEVEL_AGENT_MAP = {
    l: [a for a in range(len(AGENT_TRAJECTORIES)) if AGENT_TRAJECTORIES[a][l] != ""]
    for l in AGENT_TRAJECTORIES[0].keys()
}
START_POSITIONS = [
    (12, 12),
    (12, 12),
    (12, 12),
    (5, 9),
    (5, 9),
    (6, 12),
    (6, 12),
    (2, 2),
    (2, 2),
    (6, 12),
    (6, 12),
]
# ------------------------------------------------------------


def load_sessions(sessions_path):
    sessions, counter = [], {}
    for f in os.listdir(sessions_path):
        if not f.endswith(".json"):
            continue
        s = load_json(os.path.join(sessions_path, f))
        sessions.append(s)

    print(f"Loaded {len(sessions)} sessions")
    return sessions


def analyse_sessions(sessions, sim_func):
    data = {
        "level": [],
        "group": [],
        "imitation": [],
    }

    # populate data dict
    for s in tqdm(sessions):
        for l in (3, 4):
            # for each level, for each agent, record the similarity between their trajectory and
            # the participant's trajectory. if participant did not play this level, the similarity
            # function will return NaN.
            #
            # to make the data easier to work with, we also record the similarity between the
            # participant's trajectory and the aligned agent's trajectory, and the similarity
            # between the participant's trajectory and the misaligned agent's trajectory
            start = ",".join([str(x) for x in START_POSITIONS[l]])
            agents = LEVEL_AGENT_MAP[l]

            if str(l) in s["trajectories"]:
                sims = [
                    sim_func(AGENT_TRAJECTORIES[a][l], s["trajectories"][str(l)], start)
                    for a in agents
                ]
                sims = norm_unit_sum(jnp.array(sims))
            else:
                sims = [jnp.nan for _ in agents]

            # collapse counterbalancing
            if s["utility"]["goals"][1] == max(s["utility"]["goals"]):
                sims = sims[::-1]

            for i in range(len(sims)):
                data["level"].append(l)
                data["group"].append(i)
                data["imitation"].append(float(sims[i]))

    # convert to dataframe, remove nans and return
    df = pd.DataFrame(data)
    df["strategy"] = "human"
    df["agents known"] = True
    return df[df["imitation"].isna() == False]


def simulate_choices(rng_key: jax.Array, u, beta, N):
    traj_options = [AGENT_TRAJECTORIES[i][3] for i in (0, 1)]
    p = jnp.exp(u / beta)
    p /= jnp.sum(u)
    choices = jax.random.choice(rng_key, len(u), shape=(N,), p=p)
    trajs = [traj_options[c] for c in choices]
    return choices, trajs


def generate_obs_history(
    rng_key: jax.Array, mus: jnp.array, sigma=0.1, M=100, N=1000, beta=0.1, c=0.1
):
    K = mus.shape[0]
    weights = jnp.ones(K) / K
    sigmas = sigma * jnp.ones((K, 2))
    obs_history = generate_behaviour(
        rng_key,
        weights,
        mus,
        sigmas,
        M,
        N,
        beta=beta,
        c=c,
        simulation_func=simulate_choices,
    )

    new_obs = []
    for m in range(2):
        choices, _ = simulate_choices(rng_key, mus[m], beta=beta, N=N)
        new_obs.append({"choices": choices})

    return obs_history, new_obs


def model(data):
    """
    Dirichlet process mixture model via the stick-breaking construction
    """
    choices, noise = data
    v = numpyro.sample(
        "v", dist.MultivariateNormal(jnp.zeros(2) + 0.5, 1.0 * jnp.eye(2))
    )
    p = jnp.exp(v / noise)
    p /= jnp.sum(p, axis=-1, keepdims=True)

    # choices contains the empirical choice proportions for each agent
    # we can compute their log likelihood under a multinomial distribution
    choices_log_prob = dist.Multinomial(probs=p, total_count=jnp.sum(choices)).log_prob(
        choices
    )
    numpyro.factor(
        "obs",
        choices_log_prob,
    )


def run_model(
    rng_key: jax.random.KeyArray,
    obs_history: List[dict],
    new_obs: List[dict],
    v_self: jnp.ndarray,
    beta=0.1,
    beta_self=0.1,
    N=1000,
    **kwargs,
):
    weights = individual_inference(
        rng_key,
        obs_history,
        new_obs,
        beta,
        v_self,
        v_domain_2d(),
        0,
        0,
        model=model,
        num_options=2,
    )

    results = {
        "group": [],
        "imitation": [],
    }

    p = jnp.exp(weights / beta_self)
    p /= jnp.sum(p)
    for choice in jax.random.choice(rng_key, len(weights), shape=(N,), p=p):
        for m in range(len(weights)):
            results["group"].append(m)
            results["imitation"].append(int(choice == m))

    results["agents known"] = [True] * len(results["group"])
    results["strategy"] = ["model"] * len(results["group"])
    return pd.DataFrame(results)


# M, N, beta, beta_self = 100, 1000, 0.1, 0.2

# # set random key for reproducibility
# seed = int(time.time())
# rng_key = jax.random.PRNGKey(seed)
# print(f"using seed {seed}")

# mus = jnp.array([[0, 0.5], [1, 0.5]])
# obs_history, new_obs = generate_obs_history(
#     rng_key, mus, sigma=0.1, M=M, N=N, beta=beta, c=0.1
# )

# model_results = run_model(
#     rng_key,
#     obs_history,
#     new_obs,
#     v_self=mus[0],
#     beta=beta,
#     beta_self=beta_self,
#     N=N,
# )

# load sessions
sessions = load_sessions(
    sessions_path="/Users/max/Code/experiment-analyses/conference/data/sessions"
)
filter = lambda s: (not s["isTest"]) and s["trajectories"] and s["humanId"] != "h-max"
sessions = [s for s in sessions if filter(s)]
print(f"Num sessions after filtering: {len(sessions)}")

print(sessions[0])

# human_results = analyse_sessions(sessions, sim_func=similarity_continuous)

# # combine results
# results = pd.concat([human_results, model_results])
# make_barplots(results)
