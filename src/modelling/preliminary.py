import numpy as np
import jax.numpy as jnp
from jax import random

from .item_gridworld import (
    LENGTH,
    LOC_TO_ITEM,
    item_values,
    random_locs,
    compute_state_values,
    path_to_item,
    traj_reward,
    choose,
)

MAX_STEPS = 50


def simulate_choices(rng_key, v, beta, N):
    v_ = jnp.array(item_values(v[0], v[1], as_dict=False))
    p = jnp.exp(v_ / beta)
    p /= jnp.sum(p)
    return random.choice(rng_key, len(v_), shape=(N,), p=p)


def simulate_trajectories(rng_key, v, c, beta, N):
    choices, trajs = [], []
    v_ = item_values(v[0], v[1])
    for start in random_locs(rng_key, N):
        choice, traj = simulate_trajectory(rng_key, v_, start, c, beta)
        choices.append(choice)
        trajs.append(traj)
    return choices, trajs


def simulate_trajectory(rng_key, v, start, c, beta, level="traj", V=None):
    # make choice at trajectory level
    if level == "traj":
        trajs = [path_to_item(start, item) for item in v.keys()]
        vals = [traj_reward(traj, v, c) for traj in trajs]
        choice_idx = int(choose(rng_key, np.array(vals), beta))
        return choice_idx, trajs[choice_idx]

    # make choices at step level
    if V is None:
        V = compute_state_values(v, c)

    states, steps = [start], 0
    while steps < MAX_STEPS:
        pos = states[-1]
        if pos in LOC_TO_ITEM:
            break

        # determine legal moves
        next_states = []
        r, c = pos
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if 0 <= r + dr < LENGTH and 0 <= c + dc < LENGTH:
                next_states.append((r + dr, c + dc))

        # make choice
        vals = [V[s[0], s[1]] for s in next_states]
        choice_idx = choose(rng_key, np.array(vals), beta)
        states.append(next_states[choice_idx])
        steps += 1

    return states


def visualise_trajectory(traj):
    grid = np.zeros((LENGTH, LENGTH))
    for i in range(LENGTH):
        for j in range(LENGTH):
            if (i, j) in LOC_TO_ITEM:
                grid[i, j] = 0.5
            elif (i, j) in traj:
                grid[i, j] = 1
    return grid


def simulate_imitation(vself, choices):
    vself_ = item_values(vx=float(vself[0]), vy=float(vself[1]), as_dict=False)
    vself_ = jnp.array(vself_)
    return jnp.mean(vself_[choices])


def simulate_imitation_old(rng_key, vself, vm, start_locs, c=0.1, beta=0.01):
    vself_ = item_values(vx=float(vself[0]), vy=float(vself[1]))
    vm_ = item_values(vx=float(vm[0]), vy=float(vm[1]))
    # V = compute_state_values(vm_, c)

    rewards = np.zeros(len(start_locs))
    keys = random.split(rng_key, len(start_locs))

    # generate trajectory choices from random starting locations
    for i, start in enumerate(start_locs):
        _, traj = simulate_trajectory(keys[0], vm_, start, c=c, beta=beta)
        rewards[i] = traj_reward(traj, vself_, c)

    return np.mean(rewards)
