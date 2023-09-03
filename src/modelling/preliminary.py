import numpy as np

from src.modelling.distributions import boltzmann1d

MAX_STEPS = 50
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


def random_locs_1d(n):
    return np.random.randint(1, LENGTH - 1, size=n)


def random_locs_2d(n):
    all_locs = [
        (r, c) for r in range(LENGTH) for c in range(LENGTH) if (r, c) not in LOC_TO_ITEM_1D
    ]
    indices = np.random.choice(len(all_locs), size=n, replace=True)
    return [all_locs[i] for i in indices]


def choose(values, beta):
    probs = boltzmann1d(values, beta)
    return np.random.choice(len(values), p=probs)


def value_similarity(vself, v):
    sims = 1 - np.abs(vself - v)
    weights = np.abs(vself - 0.5)
    weights /= np.sum(weights)
    return sims, np.sum(sims * weights)


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
    vals = []
    for dest in v.keys():
        vals.append(v[dest] - c * np.abs(x - ITEM_TO_LOC_1D[dest]))
    return np.max(vals)


def square_value_2d(v, square, c):
    vals = []
    for item in v.keys():
        dest = ITEM_TO_LOC_2D[item]
        distance = np.abs(square[0] - dest[0]) + np.abs(square[1] - dest[1])
        vals.append(v[item] - c * distance)
    return np.max(vals)


def compute_state_values_1d(v, c):
    V = np.zeros(LENGTH)
    for i in range(LENGTH):
        V[i] = square_value_1d(v, i, c)
    return V


def compute_state_values_2d(v, c):
    V = np.zeros((LENGTH, LENGTH))
    for i in range(LENGTH):
        for j in range(LENGTH):
            V[i, j] = square_value_2d(v, (i, j), c)
    return V


def simulate_trajectory_1d(v, start, c, beta, level="traj", V=None):
    # make choice at trajectory level
    if level == "traj":
        trajs = [path_to_item_1d(start, item) for item in v.keys()]
        vals = [traj_reward_1d(traj, v, c) for traj in trajs]
        choice_idx = choose(np.array(vals), beta)
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
        choice_idx = choose(np.array(vals), beta)
        states.append(next_states[choice_idx])
        steps += 1

    return states


def simulate_trajectory_2d(v, start, c, beta, level="traj", V=None):
    # make choice at trajectory level
    if level == "traj":
        trajs = [path_to_item_2d(start, item) for item in v.keys()]
        vals = [traj_reward_2d(traj, v, c) for traj in trajs]
        choice_idx = choose(np.array(vals), beta)
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
        choice_idx = choose(np.array(vals), beta)
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


def simulate_imitation_1d(vm, vself, c, beta, trials, level):
    vself_ = item_values_1d(v=vself[0])
    vm_ = item_values_1d(v=vm[0])
    V = compute_state_values_1d(vm_, c)

    rewards, proportions = np.zeros(trials), np.zeros(trials)

    # generate trajectory choices from random starting locations
    for i, start in enumerate(random_locs_1d(trials)):
        traj = simulate_trajectory_1d(vm_, start, c=c, beta=beta, level=level, V=V)
        rewards[i] = traj_reward_1d(traj, vself_, c)
        proportions[i] = traj_reward_1d(traj, vm_, c) / V[start]

    return np.mean(rewards), np.mean(proportions)


def simulate_imitation_2d(vm, vself, c, beta, trials, level):
    vself_ = item_values_2d(vx=vself[0], vy=vself[1])
    vm_ = item_values_2d(vx=vm[0], vy=vm[1])
    V = compute_state_values_2d(vm_, c)

    rewards, proportions = np.zeros(trials), np.zeros(trials)

    # generate trajectory choices from random starting locations
    for i, start in enumerate(random_locs_2d(trials)):
        traj = simulate_trajectory_2d(vm_, start, c=c, beta=beta, level=level, V=V)
        rewards[i] = traj_reward_2d(traj, vself_, c)
        proportions[i] = traj_reward_2d(traj, vm_, c) / V[start[0], start[1]]

    return np.mean(rewards), np.mean(proportions)


def simulate_imitation(vm, vself, c=0.1, beta=0.01, trials=10000, level="traj"):
    if len(vm) == 1:
        return simulate_imitation_1d(vm, vself, c, beta, trials, level)
    elif len(vm) == 2:
        return simulate_imitation_2d(vm, vself, c, beta, trials, level)
    raise Exception(f"got unsupported number of dimensions {len(vm)}")
