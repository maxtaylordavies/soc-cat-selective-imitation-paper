import numpy as np
import pandas as pd
from tqdm import tqdm

from .baselines import indiscriminate, ingroup_bias


def _init_results_dict():
    return {
        "agent": [],
        "similarity": [],
        "own_group_visible": [],
        "agents_known": [],
        "groups_relevant": [],
    }


def choose_agent(
    agent_categories: np.ndarray, category_weights: np.ndarray, beta: float
) -> int:
    # compute boltzmann probabilities
    w = category_weights[agent_categories]
    p = np.exp(w / beta)
    p /= np.sum(p)

    # Choose agent
    return np.random.choice(len(w), p=p)


def simulate_agent_choice(
    agent_categories: np.ndarray,
    category_weights: np.ndarray,
    beta: float,
    repeats: int,
    own_group_visible: bool,
    agents_known: bool,
) -> pd.DataFrame:
    results = _init_results_dict()

    for _ in tqdm(range(repeats)):
        choice = choose_agent(agent_categories, category_weights, beta)
        for j in range(len(agent_categories)):
            results["agent"].append(j)
            results["similarity"].append(int(choice == j))
            results["agents_known"].append(False)
            results["own_group_visible"].append(own_group_visible)
            results["groups_relevant"].append(agents_known)

    return pd.DataFrame(results)


def simulate_strategy(
    agent_categories: np.ndarray,
    strategy: str,
    beta: float,
    repeats: int,
    own_cat=0,
    ingroup_strength=1.0,
) -> pd.DataFrame:
    df = pd.DataFrame(_init_results_dict())
    for ogv in [True, False]:
        for gr in [True, False]:
            if strategy == "value func inference":
                raise NotImplementedError
            elif strategy == "latent group inference":
                raise NotImplementedError
            elif strategy == "ingroup bias" and ogv:
                weights = ingroup_bias(len(agent_categories), own_cat, ingroup_strength)
            else:
                weights = indiscriminate(len(agent_categories))

            tmp = simulate_agent_choice(
                agent_categories,
                weights,
                beta,
                repeats,
                own_group_visible=ogv,
                agents_known=gr,
            )
            df = pd.concat([df, tmp])

    return df
