from jax import random
import jax.numpy as jnp

from src.modelling.probabilistic import boltzmann1d


LENGTH = 11
ITEMS_1D = ("A", "B")
ITEM_TO_LOC_1D = {"A": 0, "B": LENGTH - 1}
LOC_TO_ITEM_1D = {v: k for k, v in ITEM_TO_LOC_1D.items()}
ITEMS_2D = ("A", "B", "C", "D")
ITEM_TO_LOC_2D = {
    "A": (0, 0),
    "B": (0, LENGTH - 1),
    "C": (LENGTH - 1, 0),
    "D": (LENGTH - 1, LENGTH - 1),
}
LOC_TO_ITEM_2D = {v: k for k, v in ITEM_TO_LOC_2D.items()}


def item_values_1d(v):
    return {
        "A": v,
        "B": 1 - v,
    }


def item_values_2d(vx, vy):
    return {
        "A": vy * (1 - vx),  # top left
        "B": vy * vx,  # top right
        "C": (1 - vy) * (1 - vx),  # bottom left
        "D": (1 - vy) * vx,  # bottom right
    }


def random_locs_1d(rng_key, n):
    return random.randint(rng_key, (n,), 1, LENGTH - 1)


def random_locs_2d(rng_key, n):
    all_locs = [
        (r, c) for r in range(LENGTH) for c in range(LENGTH) if (r, c) not in LOC_TO_ITEM_1D
    ]
    indices = random.choice(rng_key, len(all_locs), shape=(n,), replace=True)
    return [all_locs[i] for i in indices]


def choose(rng_key, values, beta):
    probs = boltzmann1d(values, beta)
    return random.choice(rng_key, len(values), p=probs)


def path_to_item_1d(start, item):
    dest = ITEM_TO_LOC_1D[item]
    if start == dest:
        return [start]
    elif start < dest:
        return list(range(start, dest + 1))
    else:
        return list(range(start, dest - 1, -1))


def path_to_item_2d(start, item):
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

    dest = ITEM_TO_LOC_2D[item]
    return l_path(start, dest)


def traj_reward_1d(traj, v, c):
    return traj_reward(traj, v, c, LOC_TO_ITEM_1D)


def traj_reward_2d(traj, v, c):
    return traj_reward(traj, v, c, LOC_TO_ITEM_2D)


def traj_reward(traj, v, c, map):
    r = 0
    for s in traj:
        r += v[map[s]] if s in map else -c
    return r


def square_value_1d(v, x, c):
    def item_val(item):
        dest = ITEM_TO_LOC_1D[item]
        return v[item] - c * jnp.abs(x - dest)

    vals = jnp.array([item_val(item) for item in v.keys()])
    return jnp.max(vals)


def square_value_2d(v, square, c):
    def item_val(item):
        dest = ITEM_TO_LOC_2D[item]
        distance = jnp.abs(square[0] - dest[0]) + jnp.abs(square[1] - dest[1])
        return v[item] - c * distance

    vals = jnp.array([item_val(item) for item in v.keys()])
    return jnp.max(jnp.array(vals))


def compute_state_values_1d(v, c):
    return jnp.array([square_value_1d(v, x, c) for x in range(LENGTH)])


def compute_state_values_2d(v, c):
    return jnp.array(
        [[square_value_2d(v, (i, j), c) for j in range(LENGTH)] for i in range(LENGTH)]
    )
