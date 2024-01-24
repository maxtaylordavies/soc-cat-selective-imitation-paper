import jax.numpy as jnp
import pandas as pd

# 1 = left, 2 = up, 3 = right, 4 = down
ACTION_MAP = {
    "1": jnp.array([-1, 0]),
    "2": jnp.array([0, 1]),
    "3": jnp.array([1, 0]),
    "4": jnp.array([0, -1]),
}

MIN_X, MAX_X = 0, 10
MIN_Y, MAX_Y = 0, 8


# helper function to convert action string into sequence of coordinates
def to_coords(start, trajectory):
    actions = trajectory.split(",")
    coords = jnp.zeros((len(actions) + 1, 2))

    x, y = start.split(",")
    coords[0] = jnp.array([int(x), int(y)])

    for i, a in enumerate(actions):
        c = coords[i] + ACTION_MAP[a]
        if c[0] < MIN_X or c[0] > MAX_X or c[1] < MIN_Y or c[1] > MAX_Y:
            break
        coords[i + 1] = c

    return coords


# helper function to compute the frechet distance between two trajectories
# naive recursive implementation, not super performant but that's ok because
# we're working with small trajectories
def frechet_dist(traj1: jnp.ndarray, traj2: jnp.ndarray) -> float:
    memo = jnp.zeros((traj1.shape[0], traj2.shape[0])) - 1

    def recurse(i, j):
        if memo[i, j] > -1:
            return memo[i, j]

        # euclidean distance between the two points
        d = jnp.linalg.norm(traj1[i] - traj2[j])

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
    return jnp.exp(-d)


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
        # thetas = jnp.array(session["conditions"]["thetas"])
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


def analyse_trajectory_dict(td, sim_func):
    data = {
        "sim_1": [],
        "sim_2": [],
    }

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
        "group": [],
        "imitation": [],
        "agents known": [],
        "own group label": [],
        "groups relevant": [],
    }

    for s in sessions:
        groups_relevant = s["conditions"]["correlation"] == 0
        own_group_label = "hidden"
        if s["conditions"]["phi"] > -1:
            own_group_label = "matched" if groups_relevant else "arbitrary"

        # collapse counterbalancing:
        # if the participant has an explicit group label, then that should
        # correspond to 'group 1'. otherwise, if the groups are relevant,
        # then the participant's utility function should put them in 'group 1'
        reverse = False
        if s["conditions"]["phi"] == 1:
            reverse = True
        elif s["conditions"]["phi"] == -1:
            thetas = jnp.array(s["conditions"]["thetas"])
            reverse = bool(thetas[0][1] > thetas[0][0] or thetas[1][1] > thetas[1][0])

        sim_func = similarity_binary if flavour == "binary" else similarity_continuous

        for phase_idx in [2, 4]:
            d = get_trajectory_dict(s, phase_idx)
            if d is None:
                continue
            tmp = analyse_trajectory_dict(d, sim_func)
            for i, group in enumerate([1, 0] if reverse else [0, 1]):
                vals = tmp[f"sim_{i + 1}"]
                results["group"].extend([group] * len(vals))
                results["imitation"].extend(vals)
                results["agents known"].extend([phase_idx == 2] * len(vals))
                results["own group label"].extend([own_group_label] * len(vals))
                results["groups relevant"].extend([groups_relevant] * len(vals))

    df = pd.DataFrame(results)
    df["strategy"] = "human"
    return df
