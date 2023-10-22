from datetime import datetime
import os

from jax import random
import jax.numpy as jnp
import pandas as pd
from tqdm import tqdm

from src.utils import (
    load_sessions,
    save_responses,
    generate_bonus_file,
    make_condition_barplots,
    plot_model_effectiveness,
    surfaceplot,
    value_similarity,
)
from src.human import analyse_sessions
from src.modelling import simulate_strategy, simulate_imitation, analyse_model_effectiveness


rng_key = random.PRNGKey(0)


def run_preliminary_simulations():
    def make_surface_plots(data, betas, folder):
        if not os.path.exists(f"results/{folder}"):
            os.makedirs(f"results/{folder}")

        for fmt in ["svg", "pdf"]:
            for level in data["level"].unique():
                d = data[data["level"] == level]

                # surfaceplot(
                #     d[d["beta"] == jnp.log10(betas[0])],
                #     ["sim_x", "sim_y", "reward"],
                #     ["Value similarity (x)", "Value similarity (y)", "Average reward"],
                #     filename=f"results/{folder}/surface_{level}_sim",
                #     format=fmt,
                # )

                surfaceplot(
                    d,
                    ["sim", "beta", "reward"],
                    ["Value similarity (combined)", "Decision noise", "Average reward"],
                    ticks={"y": [jnp.log10(b) for b in betas]},
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
    vself = jnp.array([1])

    data = {"sim": [], "beta": [], "competence": [], "reward": [], "level": []}
    for vm in tqdm([jnp.array([v]) for v in jnp.linspace(0, 1, 11)]):
        for beta in betas:
            for level in ["traj"]:
                reward, proportion = simulate_imitation(
                    vself, vm, c, beta, level=level, trials=trials
                )
                data["reward"].append(reward)
                data["competence"].append(proportion)
                data["sim"].append(value_similarity(vself, vm))
                data["beta"].append(jnp.log10(beta))
                data["level"].append(level)

    data = pd.DataFrame(data)
    make_surface_plots(data, betas, "1d")

    # # 2-dimensional case

    # data = {
    #     "sim_x": [],
    #     "sim_y": [],
    #     "sim": [],
    #     "beta": [],
    #     "competence": [],
    #     "reward": [],
    #     "level": [],
    #     "vself_idx": [],
    # }

    # vselfs = [
    #     jnp.array([0, 1]),
    #     jnp.array([1, 0]),
    #     jnp.array([1, 0.5]),
    #     jnp.array([0.5, 1]),
    # ]

    # for i, vself in enumerate(vselfs):
    #     for vx in tqdm(jnp.linspace(0, 1, 11)):
    #         for vy in jnp.linspace(0, 1, 11):
    #             vm = jnp.array([vx, vy])
    #             for beta in betas:
    #                 for level in ["traj", "step"]:
    #                     reward, proportion = simulate_imitation(
    #                         vself, vm, c, beta, level=level, trials=trials
    #                     )
    #                     sims, combined = value_similarity(vself, vm)
    #                     data["sim_x"].append(sims[0])
    #                     data["sim_y"].append(sims[1])
    #                     data["sim"].append(combined)
    #                     data["reward"].append(reward)
    #                     data["beta"].append(jnp.log10(beta))
    #                     data["competence"].append(proportion)
    #                     data["level"].append(level)
    #                     data["vself_idx"].append(i)

    # data = pd.DataFrame(data)

    # for i, vself in enumerate(vselfs):
    #     d = data[data["vself_idx"] == i]
    #     make_surface_plots(d, betas, f"2d/step/{i}")

    # for fmt in ["svg", "pdf"]:
    #     surfaceplot(
    #         d,
    #         ["sim", "beta", "reward"],
    #         ["Value similarity (combined)", "Decision noise", "Average reward"],
    #         ticks={"y": [jnp.log10(b) for b in betas]},
    #         ticklabels={"y": [str(b) for b in betas]},
    #         filename=f"results/2d/surface_{level}_beta",
    #         format=fmt,
    #     )


def run_model_analysis(results_dir: str):
    K, M, N = 5, 1000, 10000
    sigma = 0.01
    beta = 1.0
    c = 0.1

    model_names = [
        # "value functions known",
        # "value functions inferred (individual)",
        # "value functions inferred (group)",
        "full bayesian",
    ]

    results, weights, phis, vselfs = analyse_model_effectiveness(
        rng_key, model_names, M=M, N=N, K=K, sigma=sigma, beta=beta, c=c, plot_dir=results_dir
    )

    for mn in model_names:
        key = mn.replace(" ", "_")
        plot_model_effectiveness(
            results,
            weights,
            phis,
            vselfs,
            mn,
            filename=f"{results_dir}/model_effectiveness_{key}.png",
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


def run_human_model_comparison():
    for strategy in ["indiscriminate", "ingroup bias"]:
        results = simulate_strategy(
            agent_categories=jnp.array([0, 1]),
            strategy=strategy,
            beta=1,
            repeats=1000,
            own_cat=0,
            ingroup_strength=0.75,
        )
        make_condition_barplots(results, flavour="binary", filename=strategy)


def main():
    # define run name based on current date and time
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"results/{run_name}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    run_model_analysis(results_dir)
    # run_human_data_analysis(results_dir)
    # run_human_model_comparison(results_dir)


if __name__ == "__main__":
    main()
