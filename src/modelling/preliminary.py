import numpy as np

from .worlds.item_gridworld import (
    LENGTH,
    LOC_TO_ITEM_1D,
    LOC_TO_ITEM_2D,
    item_values_1d,
    item_values_2d,
    random_locs_1d,
    random_locs_2d,
    compute_state_values_1d,
    compute_state_values_2d,
    path_to_item_1d,
    path_to_item_2d,
    traj_reward_1d,
    traj_reward_2d,
    choose,
)

MAX_STEPS = 50


def simulate_trajectories(rng_key, v, c, beta, N):
    if len(v) == 1:
        return simulate_trajectories_1d(rng_key, v, c, beta, N)
    elif len(v) == 2:
        return simulate_trajectories_2d(rng_key, v, c, beta, N)
    raise Exception(f"got unsupported number of dimensions {len(v)}")


def simulate_trajectories_1d(rng_key, v, c, beta, N):
    trajs = []
    v_ = item_values_1d(v[0])
    for start in random_locs_1d(rng_key, N):
        trajs.append(simulate_trajectory_1d(rng_key, v_, start, c, beta))
    return trajs


def simulate_trajectories_2d(rng_key, v, c, beta, N):
    trajs = []
    v_ = item_values_2d(v[0], v[1])
    for start in random_locs_2d(rng_key, N):
        trajs.append(simulate_trajectory_2d(rng_key, v_, start, c, beta))
    return trajs


def simulate_trajectory_1d(rng_key, v, start, c, beta, level="traj", V=None):
    # make choice at trajectory level
    if level == "traj":
        trajs = [path_to_item_1d(start, item) for item in v.keys()]
        vals = [traj_reward_1d(traj, v, c) for traj in trajs]
        choice_idx = choose(rng_key, np.array(vals), beta)
        return trajs[choice_idx]

    # make choices at step level
    if V is None:
        V = compute_state_values_1d(v, c)

    states, steps = [start], 0
    while steps < MAX_STEPS:
        pos = states[-1]
        if pos in LOC_TO_ITEM_1D:
            break

        next_states = [pos - 1, pos + 1]
        vals = [V[s] for s in next_states]
        choice_idx = choose(rng_key, np.array(vals), beta)
        states.append(next_states[choice_idx])
        steps += 1

    return states


def simulate_trajectory_2d(rng_key, v, start, c, beta, level="traj", V=None):
    # make choice at trajectory level
    if level == "traj":
        trajs = [path_to_item_2d(start, item) for item in v.keys()]
        vals = [traj_reward_2d(traj, v, c) for traj in trajs]
        choice_idx = choose(rng_key, np.array(vals), beta)
        return trajs[choice_idx]

    # make choices at step level
    if V is None:
        V = compute_state_values_2d(v, c)

    states, steps = [start], 0
    while steps < MAX_STEPS:
        pos = states[-1]
        if pos in LOC_TO_ITEM_2D:
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


def visualise_trajectory_2d(traj):
    grid = np.zeros((LENGTH, LENGTH))
    for i in range(LENGTH):
        for j in range(LENGTH):
            if (i, j) in LOC_TO_ITEM_2D:
                grid[i, j] = 0.5
            elif (i, j) in traj:
                grid[i, j] = 1
    return grid


def simulate_imitation_1d(rng_key, vself, vm, c, beta, trials, level):
    vself_ = item_values_1d(v=vself[0])
    vm_ = item_values_1d(v=vm[0])
    V = compute_state_values_1d(vm_, c)

    rewards, proportions = np.zeros(trials), np.zeros(trials)

    # generate trajectory choices from random starting locations
    for i, start in enumerate(random_locs_1d(rng_key, trials)):
        traj = simulate_trajectory_1d(rng_key, vm_, start, c=c, beta=beta, level=level, V=V)
        rewards[i] = traj_reward_1d(traj, vself_, c)
        proportions[i] = traj_reward_1d(traj, vm_, c) / V[start]

    return np.mean(rewards), np.mean(proportions)


def simulate_imitation_2d(rng_key, vself, vm, c, beta, trials, level):
    vself_ = item_values_2d(vx=vself[0], vy=vself[1])
    vm_ = item_values_2d(vx=vm[0], vy=vm[1])
    V = compute_state_values_2d(vm_, c)

    rewards, proportions = np.zeros(trials), np.zeros(trials)

    # generate trajectory choices from random starting locations
    for i, start in enumerate(random_locs_2d(rng_key, trials)):
        traj = simulate_trajectory_2d(rng_key, vm_, start, c=c, beta=beta, level=level, V=V)
        rewards[i] = traj_reward_2d(traj, vself_, c)
        proportions[i] = traj_reward_2d(traj, vm_, c) / V[start[0], start[1]]

    return np.mean(rewards), np.mean(proportions)


def simulate_imitation(rng_key, vself, vm, c=0.1, beta=0.01, trials=10000, level="traj"):
    if len(vm) == 1:
        return simulate_imitation_1d(rng_key, vm, vself, c, beta, trials, level)
    elif len(vm) == 2:
        return simulate_imitation_2d(rng_key, vm, vself, c, beta, trials, level)
    raise Exception(f"got unsupported number of dimensions {len(vm)}")
