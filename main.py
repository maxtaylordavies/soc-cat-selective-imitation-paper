import argparse
import time
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
from src.modelling import (
    simulate_strategy,
    simulate_imitation,
    analyse_model_effectiveness,
    generate_behaviour_simple,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--M",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--N",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--K",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--beta-self",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--c",
        type=float,
        default=0.1,
    )
    return parser.parse_args()


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


def generate_obs_history(rng_key: random.KeyArray, K, M, N, sigma, beta):
    weights = jnp.ones(K) / K
    sigmas = sigma * jnp.ones((K, 2))
    mus = jnp.array([[0.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [1.0, 1.0]])

    return (
        mus,
        sigmas,
        generate_behaviour_simple(rng_key, weights, mus, sigmas, M, N, beta=beta),
    )


def run_model_analysis(rng_key: random.KeyArray, args):
    model_names = [
        # "value functions known",
        # "value functions inferred (individual)",
        # "value functions inferred (group)",
        "full bayesian",
    ]

    _, _, obs_history = generate_obs_history(
        rng_key, args.K, args.M, args.N, args.sigma, args.beta
    )
    results, weights, phis, vselfs = analyse_model_effectiveness(
        rng_key, model_names, obs_history, K=args.K, beta=args.beta, plot_dir=args.results_dir
    )

    for mn in model_names:
        key = mn.replace(" ", "_")
        plot_model_effectiveness(
            results,
            weights,
            phis,
            vselfs,
            mn,
            filename=f"{args.results_dir}/model_effectiveness_{key}.png",
        )


def run_human_model_comparison(rng_key: random.KeyArray, args):
    mus, _, obs_history = generate_obs_history(
        rng_key, args.K, args.M, args.N, args.sigma, args.beta
    )

    phi_self = 0
    target_phis = jnp.array([phi_self, args.K - 1])
    v_self = mus[phi_self]

    for strategy in ["indiscriminate", "ingroup bias", "full bayesian"]:
        results = simulate_strategy(
            rng_key=rng_key,
            strategy=strategy,
            obs_history=obs_history,
            target_phis=target_phis,
            v_self=v_self,
            phi_self=phi_self,
            beta=args.beta,
            beta_self=args.beta_self,
            repeats=1000,
            ingroup_strength=0.75,
            plot_dir=args.results_dir,
        )

        make_condition_barplots(
            results,
            flavour="binary",
            plot_dir=args.results_dir,
            filename=strategy,
            format="png",
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


def main():
    args = parse_args()
    print("Running with arguments:")
    print(args)

    # set random key for reproducibility
    seed = int(time.time())
    rng_key = random.PRNGKey(seed)
    print(f"using seed {seed}")

    # define run name based on current date and time
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.results_dir = f"results/{run_name}"
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    run_human_model_comparison(rng_key, args)


if __name__ == "__main__":
    main()
