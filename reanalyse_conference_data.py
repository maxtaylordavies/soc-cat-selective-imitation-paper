from datetime import datetime
from os import path

import pandas as pd
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import load_sessions, norm_unit_sum, make_barplots
from src.human import similarity_binary, similarity_continuous

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
AGENT_TRAJECTORIES = [
    {
        0: "1,1,1,1,1,1,1,1,1,1,1,1",
        1: "3,3,3,3,3,3,3,3,3,3,3,3",
        2: "2,2,2,2,2,2,2,2,2,2,2,2",
        3: "4,4,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2",
        4: "4,4,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2",
        5: "1,1,1,1,1,1,1,1,1,1,1,1",
        6: "3,3,3,3,3,3,3,3,3,3,3,3",
        7: "4,4,4,4,4,4,4,4,4,4,4",
        8: "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3",
        9: "",
        10: "",
    },
    {
        0: "2,2,2,2,2,2,2,2,2,2,2,2",
        1: "1,1,1,1,1,1,1,1,1,1,1,1",
        2: "3,3,3,3,3,3,3,3,3,3,3,3",
        3: "4,4,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2",
        4: "4,4,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2",
        5: "3,3,3,3,3,3,3,3,3,3,3,3",
        6: "1,1,1,1,1,1,1,1,1,1,1,1",
        7: "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3",
        8: "4,4,4,4,4,4,4,4,4,4,4",
        9: "",
        10: "",
    },
    {
        0: "",
        1: "",
        2: "",
        3: "",
        4: "",
        5: "",
        6: "",
        7: "4,4,4,4,4,4,4,4,4,4,4",
        8: "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3",
        9: "1,1,1,1,1,1,1,1,1,1,1,1",
        10: "3,3,3,3,3,3,3,3,3,3,3,3",
    },
    {
        0: "",
        1: "",
        2: "",
        3: "",
        4: "",
        5: "",
        6: "",
        7: "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3",
        8: "4,4,4,4,4,4,4,4,4,4,4",
        9: "3,3,3,3,3,3,3,3,3,3,3,3",
        10: "1,1,1,1,1,1,1,1,1,1,1,1",
    },
]
LEVEL_AGENT_MAP = {
    l: [a for a in range(len(AGENT_TRAJECTORIES)) if AGENT_TRAJECTORIES[a][l] != ""]
    for l in AGENT_TRAJECTORIES[0].keys()
}
START_POSITIONS = [
    (12, 12),
    (12, 12),
    (12, 12),
    (5, 9),
    (5, 9),
    (6, 12),
    (6, 12),
    (2, 2),
    (2, 2),
    (6, 12),
    (6, 12),
]
# ------------------------------------------------------------


def analyse_sessions(sessions, sim_func):
    # initialise data dict
    # cols = []
    # for l in (3, 4):
    #     cols += [f"level_{l}_agent_{a + 1}" for a in LEVEL_AGENT_MAP[l]] + [
    #         f"level_{l}_aligned",
    #         f"level_{l}_misaligned",
    #     ]
    # data_dict = {col: [] for col in cols}
    data = {
        "level": [],
        "group": [],
        "imitation": [],
    }

    # populate data dict
    for s in tqdm(sessions):
        for l in (3, 4):
            # for each level, for each agent, record the similarity between their trajectory and
            # the participant's trajectory. if participant did not play this level, the similarity
            # function will return NaN.
            #
            # to make the data easier to work with, we also record the similarity between the
            # participant's trajectory and the aligned agent's trajectory, and the similarity
            # between the participant's trajectory and the misaligned agent's trajectory
            start = ",".join([str(x) for x in START_POSITIONS[l]])
            agents = LEVEL_AGENT_MAP[l]

            if str(l) in s["trajectories"]:
                sims = [
                    sim_func(AGENT_TRAJECTORIES[a][l], s["trajectories"][str(l)], start)
                    for a in agents
                ]
                sims = norm_unit_sum(jnp.array(sims))
            else:
                sims = [jnp.nan for _ in agents]

            # collapse counterbalancing
            if s["utility"]["goals"][1] == max(s["utility"]["goals"]):
                sims = sims[::-1]

            for i in range(len(sims)):
                data["level"].append(l)
                data["group"].append(i)
                data["imitation"].append(float(sims[i]))

    # convert to dataframe, remove nans and return
    df = pd.DataFrame(data)
    df["strategy"] = "human"
    df["agents known"] = True
    return df[df["imitation"].isna() == False]


sessions = load_sessions(
    sessions_path="/Users/max/Code/experiment-analyses/conference/data/sessions"
)

filter = lambda s: (not s["isTest"]) and s["trajectories"] and s["humanId"] != "h-max"
sessions = [s for s in sessions if filter(s)]
print(f"Num sessions after filtering: {len(sessions)}")

results = analyse_sessions(sessions, sim_func=similarity_continuous)
make_barplots(results, plot_dir="results/conference")
