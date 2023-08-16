import numpy as np
import pandas as pd

# 1 = left, 2 = up, 3 = right, 4 = down
ACTION_MAP = {
    "1": np.array([-1, 0]),
    "2": np.array([0, 1]),
    "3": np.array([1, 0]),
    "4": np.array([0, -1]),
}

MIN_X, MAX_X = 0, 10
MIN_Y, MAX_Y = 0, 8


# helper function to convert action string into sequence of coordinates
def to_coords(start, trajectory):
    actions = trajectory.split(",")
    coords = np.zeros((len(actions) + 1, 2))

    x, y = start.split(",")
    coords[0] = np.array([int(x), int(y)])

    for i, a in enumerate(actions):
        c = coords[i] + ACTION_MAP[a]
        if c[0] < MIN_X or c[0] > MAX_X or c[1] < MIN_Y or c[1] > MAX_Y:
            break
        coords[i + 1] = c

    return coords


# helper function to compute the frechet distance between two trajectories
# naive recursive implementation, not super performant but that's ok because
# we're working with small trajectories
def frechet_dist(traj1: np.ndarray, traj2: np.ndarray) -> float:
    memo = np.zeros((traj1.shape[0], traj2.shape[0])) - 1

    def recurse(i, j):
        if memo[i, j] > -1:
            return memo[i, j]

        # euclidean distance between the two points
        d = np.linalg.norm(traj1[i] - traj2[j])

        if i > 0 and j > 0:
            tmp = min(recurse(i - 1, j), recurse(i, j - 1), recurse(i - 1, j - 1))
            memo[i, j] = max(d, tmp)
        elif i > 0 and j == 0:
            memo[i, j] = max(d, recurse(i - 1, j))
        elif i == 0 and j > 0:
            memo[i, j] = max(d, recurse(i, j - 1))
        else:
            memo[i, j] = d

        return memo[i, j]

    return recurse(traj1.shape[0] - 1, traj2.shape[0] - 1)


def similarity_binary(traj1: str, traj2: str, start: str) -> float:
    t1 = traj1.split(",")
    t2 = traj2.split(",")
    return max(set(t1), key=t1.count) == max(set(t2), key=t2.count)


# compute similarity between two trajectories as the exponential of the negative frechet distance
def similarity_continuous(traj1: str, traj2: str, start: str) -> float:
    traj1 = to_coords(start, traj1)
    traj2 = to_coords(start, traj2)
    d = frechet_dist(traj1, traj2)
    return np.exp(-d)


def get_trajectory_dict(session, phase_idx):
    d = {
        "agent 1": {},
        "agent 2": {},
        "participant": {},
    }

    def store_traj(agent, start, traj):
        if start not in d[agent]:
            d[agent][start] = []
        d[agent][start].append(traj)

    try:
        # thetas = np.array(session["conditions"]["thetas"])
        phase = session["phases"][phase_idx]
        start = lambda sp: f"{sp['x']},{sp['y']}"

        replays = sorted(phase["agentReplays"], key=lambda a: a["agentName"])
        for i, agent in enumerate(replays):
            for r in agent["replays"]:
                store_traj(f"agent {i + 1}", start(r["startPos"]), traj=r["trajectory"])

        for i, l in enumerate(phase["levels"]):
            s = start(l["startPos"])
            traj = session["trajectories"][str(phase_idx)][str(i)]
            store_traj("participant", s, traj)

        return d
    except:
        return None


def analyse_trajectory_dict(td, flavour="binary"):
    data = {
        "sim_1": [],
        "sim_2": [],
    }

    sim_func = similarity_binary if flavour == "binary" else similarity_continuous

    starts = td["participant"].keys()
    for start in starts:
        for i in range(len(td["participant"][start])):
            traj_p = td["participant"][start][i]
            traj_1 = td["agent 1"][start][i]
            traj_2 = td["agent 2"][start][i]

            sim_1 = sim_func(traj_p, traj_1, start)
            sim_2 = sim_func(traj_p, traj_2, start)

            tot = sim_1 + sim_2
            if tot > 0:
                data["sim_1"].append(sim_1 / tot)
                data["sim_2"].append(sim_2 / tot)
            else:
                print(f"{traj_p} | {traj_1} | {traj_2}")

    return data


def analyse_sessions(sessions, flavour="binary"):
    results = {
        "agent": [],
        "similarity": [],
        "agents_known": [],
        "own_group_visible": [],
        "groups_relevant": [],
    }

    for s in sessions:
        for phase_idx in [2, 4]:
            d = get_trajectory_dict(s, phase_idx)
            if d is None:
                continue

            agents_known = phase_idx == 2
            own_group_visible = s["conditions"]["phi"] > -1
            groups_relevant = s["conditions"]["correlation"] == 0

            tmp = analyse_trajectory_dict(d, flavour=flavour)

            for a in [1, 2]:
                vals = tmp[f"sim_{a}"]
                results["agent"].extend([f"agent {a}"] * len(vals))
                results["similarity"].extend(vals)
                results["agents_known"].extend([agents_known] * len(vals))
                results["own_group_visible"].extend([own_group_visible] * len(vals))
                results["groups_relevant"].extend([groups_relevant] * len(vals))

    return pd.DataFrame(results)
