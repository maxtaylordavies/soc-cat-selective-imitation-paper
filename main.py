import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import (
    load_sessions,
    save_responses,
    generate_bonus_file,
    make_condition_barplots,
    barplot,
    surfaceplot,
)
from src.human import analyse_sessions
from src.modelling import simulate_strategy, simulate_imitation, value_similarity


def run_preliminary_simulations():
    def make_surface_plots(data, betas, folder):
        if not os.path.exists(f"results/{folder}"):
            os.makedirs(f"results/{folder}")

        for fmt in ["svg", "pdf"]:
            for level in data["level"].unique():
                d = data[data["level"] == level]

                surfaceplot(
                    d[d["beta"] == np.log10(betas[0])],
                    ["sim_x", "sim_y", "reward"],
                    ["Value similarity (x)", "Value similarity (y)", "Average reward"],
                    filename=f"results/{folder}/surface_{level}_sim",
                    format=fmt,
                )

                surfaceplot(
                    d,
                    ["sim", "beta", "reward"],
                    ["Value similarity (combined)", "Decision noise", "Average reward"],
                    ticks={"y": [np.log10(b) for b in betas]},
                    ticklabels={"y": [str(b) for b in betas]},
                    filename=f"results/{folder}/surface_{level}_beta",
                    format=fmt,
                )

                # surfaceplot(
                #     data[data["level"] == level],
                #     ["similarity", "competence", "reward"],
                #     ["Value similarity", "Competence", "Average reward"],
                #     filename=f"{folder}/surface_{level}_competence",
                #     format=fmt,
                # )

    c = 0.05
    betas = [0.01, 0.1, 1, 10, 100]
    trials = 1000

    # 1-dimensional case
    vself = np.array([1])

    data = {"sim": [], "beta": [], "competence": [], "reward": [], "level": []}
    for vm in tqdm([np.array([v]) for v in np.linspace(0, 1, 11)]):
        for beta in betas:
            for level in ["traj", "step"]:
                reward, proportion = simulate_imitation(
                    vm, vself, c, beta, level=level, trials=trials
                )
                data["reward"].append(reward)
                data["competence"].append(proportion)
                data["sim"].append(value_similarity(vm, vself))
                data["beta"].append(np.log10(beta))
                data["level"].append(level)

    data = pd.DataFrame(data)
    make_surface_plots(data, betas, "1d")

    # 2-dimensional case

    data = {
        "sim_x": [],
        "sim_y": [],
        "sim": [],
        "beta": [],
        "competence": [],
        "reward": [],
        "level": [],
        "vself_idx": [],
    }

    vselfs = [
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([1, 0.5]),
        np.array([0.5, 1]),
    ]

    for i, vself in enumerate(vselfs):
        for vx in tqdm(np.linspace(0, 1, 11)):
            for vy in np.linspace(0, 1, 11):
                vm = np.array([vx, vy])
                for beta in betas:
                    for level in ["traj", "step"]:
                        reward, proportion = simulate_imitation(
                            vm, vself, c, beta, level=level, trials=trials
                        )
                        sims, combined = value_similarity(vself, vm)
                        data["sim_x"].append(sims[0])
                        data["sim_y"].append(sims[1])
                        data["sim"].append(combined)
                        data["reward"].append(reward)
                        data["beta"].append(np.log10(beta))
                        data["competence"].append(proportion)
                        data["level"].append(level)
                        data["vself_idx"].append(i)

    data = pd.DataFrame(data)

    for i, vself in enumerate(vselfs):
        d = data[data["vself_idx"] == i]
        make_surface_plots(d, betas, f"2d/step/{i}")

    for fmt in ["svg", "pdf"]:
        surfaceplot(
            d,
            ["sim", "beta", "reward"],
            ["Value similarity (combined)", "Decision noise", "Average reward"],
            ticks={"y": [np.log10(b) for b in betas]},
            ticklabels={"y": [str(b) for b in betas]},
            filename=f"results/2d/surface_{level}_beta",
            format=fmt,
        )


def run_human_data_analysis():
    filters = {
        "experimentId": [
            "prolific-test-2",
            "prolific-test-3",
            "prolific-test-4",
            "prolific-test-5",
            "prolific-test-6",
            "prolific-test-7",
            "prolific-test-8",
        ],
    }
    sessions = load_sessions(filters)

    generate_bonus_file(sessions, "bonuses.txt")
    save_responses(sessions, f"../results/responses.txt")

    flavour = "binary"
    data = analyse_sessions(sessions, flavour=flavour)
    make_condition_barplots(data, flavour=flavour)


def run_model_analysis():
    for strategy in ["indiscriminate", "ingroup bias"]:
        results = simulate_strategy(
            agent_categories=np.array([0, 1]),
            strategy=strategy,
            beta=1,
            repeats=1000,
            own_cat=0,
            ingroup_strength=0.75,
        )
        make_condition_barplots(results, flavour="binary", filename=strategy)


def main():
    # run_preliminary_simulations()
    # run_human_data_analysis()
    run_model_analysis()


if __name__ == "__main__":
    main()
