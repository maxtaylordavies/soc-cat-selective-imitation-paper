import numpy as np


def indiscriminate(num_cats: int) -> np.ndarray:
    return np.ones(num_cats) / num_cats


def ingroup_bias(num_cats: int, own_cat: int, strength: float) -> np.ndarray:
    weights = -np.ones(num_cats)
    weights[own_cat] = 1
    return weights * strength
