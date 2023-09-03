import numpy as np

from src.utils import value_similarity


def empirical_mean(agents):
    tmp = {}
    for a in agents:
        if a["phi"] not in tmp:
            tmp[a["phi"]] = []
        tmp[a["phi"]].append(a["v"])

    return {k: np.mean(v) for k, v in tmp.items()}


def explicit_value_funcs(known_agents, cat_phis, vself, type="empirical_mean"):
    if type == "empirical_mean":
        mapping = empirical_mean(known_agents)
    else:
        raise ValueError(f"Unknown type: {type}")

    return np.array([value_similarity(vself, mapping[phi]) for phi in cat_phis])
