from jax import random
import jax.numpy as jnp

from src.modelling.probabilistic import boltzmann1d


LENGTH = 11
ITEMS = ("A", "B", "C", "D")
ITEM_TO_LOC = {
    "A": (0, 0),
    "B": (0, LENGTH - 1),
    "C": (LENGTH - 1, 0),
    "D": (LENGTH - 1, LENGTH - 1),
}
LOC_TO_ITEM = {v: k for k, v in ITEM_TO_LOC.items()}


def item_values(vx, vy, as_dict=True):
    va = vy * (1 - vx)  # top left
    vb = vy * vx  # top right
    vc = (1 - vy) * (1 - vx)  # bottom left
    vd = (1 - vy) * vx  # bottom right

    if as_dict:
        return {
            "A": va,
            "B": vb,
            "C": vc,
            "D": vd,
        }

    return va, vb, vc, vd


def random_locs(rng_key, n):
    all_locs = [
        (r, c) for r in range(LENGTH) for c in range(LENGTH) if (r, c) not in LOC_TO_ITEM
    ]
    indices = random.choice(rng_key, len(all_locs), shape=(n,), replace=True)
    return [all_locs[i] for i in indices]


def choose(rng_key, values, beta):
    probs = boltzmann1d(values, beta)
    return random.choice(rng_key, len(values), p=probs)


def path_to_item(start, item):
    def l_path(start, dest):
        sr, sc = start
        dr, dc = dest

        if dr < sr:
            return [start] + l_path((sr - 1, sc), dest)
        elif dr > sr:
            return [start] + l_path((sr + 1, sc), dest)
        elif dc < sc:
            return [start] + l_path((sr, sc - 1), dest)
        elif dc > sc:
            return [start] + l_path((sr, sc + 1), dest)
        else:
            return [start]

    dest = ITEM_TO_LOC[item]
    return l_path(start, dest)


def traj_reward(traj, v, c):
    r = 0
    for s in traj:
        r += v[LOC_TO_ITEM[s]] if s in LOC_TO_ITEM else -c
    return r


def square_value(v, square, c):
    def item_val(item):
        dest = ITEM_TO_LOC[item]
        distance = jnp.abs(square[0] - dest[0]) + jnp.abs(square[1] - dest[1])
        return v[item] - c * distance

    vals = jnp.array([item_val(item) for item in v.keys()])
    return jnp.max(jnp.array(vals))


def compute_state_values(v, c):
    return jnp.array(
        [[square_value(v, (i, j), c) for j in range(LENGTH)] for i in range(LENGTH)]
    )
