import argparse
import time
from datetime import datetime
import os

import jax
from jax import random
import jax.numpy as jnp
import pandas as pd
from tqdm import tqdm

from src.utils import (
    load_sessions,
    save_responses,
    generate_bonus_file,
    surfaceplot,
    value_similarity,
    make_barplots,
)
from src.human import analyse_sessions, similarity_binary, similarity_continuous
from src.modelling import (
    simulate_imitation_old,
    generate_behaviour,
    analyse_strategy_humanlikeness,
    random_locs,
)
from src.modelling.strategies import (
    ingroup_bias,
    individual_inference,
    groups_inference,
)

STRATEGY_DICT = {
    "individual inference": individual_inference,
    "ingroup bias": ingroup_bias,
    "groups inference": groups_inference,
}


def run_preliminary_simulations(rng_key):
    def make_surface_plots(data, betas, filename):
        if not os.path.exists(f"results/surfaces"):
            os.makedirs(f"results/surfaces")

        for fmt in ["svg", "pdf"]:
            # surfaceplot(
            #     d[d["beta"] == jnp.log10(betas[0])],
            #     ["sim_x", "sim_y", "reward"],
            #     ["Value similarity (x)", "Value similarity (y)", "Average reward"],
            #     filename=f"results/{folder}/surface_{level}_sim",
            #     format=fmt,
            # )
            surfaceplot(
                data,
                ["sim", "beta", "reward"],
                ["Value similarity (combined)", "Decision noise", "Average reward"],
                ticks={"y": [jnp.log10(b) for b in betas]},
                ticklabels={"y": [str(b) for b in betas]},
                filename=f"results/surfaces/{filename}",
                format=fmt,
            )

    c = 0.1
    betas = [0.01, 0.1, 1, 10, 100]
    trials = 1000

    data = {
        "sim_x": [],
        "sim_y": [],
        "sim": [],
        "beta": [],
        "reward": [],
        "vself_idx": [],
    }

    vselfs = jnp.array(
        [
            # [0, 0],
            [0, 1],
            # [1, 0],
            # [1, 1],
            # [1, 0.5],
            # [0.5, 1],
        ]
    )

    n = 5
    vms = jnp.array(
        [[vx, vy] for vx in jnp.linspace(0, 1, n) for vy in jnp.linspace(0, 1, n)]
    )
    start_locs = random_locs(rng_key, trials)

    for i, vself in enumerate(vselfs):
        for vm in tqdm(vms, desc=f"vself {i + 1}/{len(vselfs)}"):
            for beta in betas:
                reward = simulate_imitation_old(
                    rng_key, vself, vm, start_locs, c=c, beta=beta
                )
                sims, combined = value_similarity(vself, vm)
                data["sim_x"].append(sims[0])
                data["sim_y"].append(sims[1])
                data["sim"].append(combined)
                data["reward"].append(reward)
                data["beta"].append(jnp.log10(beta))
                data["vself_idx"].append(i)

    data = pd.DataFrame(data)

    for i, vself in enumerate(vselfs):
        d = data[data["vself_idx"] == i]
        make_surface_plots(d, betas, i)
    make_surface_plots(data, betas, "combined")

    # for fmt in ["svg", "pdf"]:
    #     surfaceplot(
    #         d,
    #         ["sim", "beta", "reward"],
    #         ["Value similarity (combined)", "Decision noise", "Average reward"],
    #         ticks={"y": [jnp.log10(b) for b in betas]},
    #         ticklabels={"y": [str(b) for b in betas]},
    #         filename=f"results/surfaces",
    #         format=fmt,
    #     )


def generate_obs_history(rng_key: jax.Array, mus: jnp.array, args):
    K = mus.shape[0]
    weights = jnp.ones(K) / K
    sigmas = args.sigma * jnp.ones((K, 2))
    behaviour = generate_behaviour(
        rng_key,
        weights,
        mus,
        sigmas,
        args.M,
        args.N,
        beta=args.beta,
        c=args.c,
        shortcut=args.shortcut,
    )
    return mus, sigmas, behaviour


def setup():
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--run-name", type=str, default="")
        parser.add_argument(
            "--M",
            type=int,
            default=50,
        )
        parser.add_argument(
            "--N",
            type=int,
            default=500,
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
            default=0.5,
        )
        parser.add_argument(
            "--c",
            type=float,
            default=0.1,
        )
        parser.add_argument(
            "--ingroup-strength",
            type=float,
            default=0.75,
        )
        parser.add_argument(
            "--shortcut",
            default=False,
            action="store_true",
        )
        parser.add_argument(
            "--sim-type",
            type=str,
            default="continuous",
        )
        return parser.parse_args()

    # set random key for reproducibility
    seed = int(time.time())
    rng_key = random.PRNGKey(seed)
    print(f"using seed {seed}")

    # parse any command line arguments
    args = parse_args()

    # if not provided, set run name based on current date and time
    if args.run_name == "":
        args.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.results_dir = f"results/{args.run_name}"
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    print("Running with arguments:")
    print(args)

    return rng_key, args


rng_key, args = setup()

# load human experiment data
sessions = load_sessions(print_breakdown=True)

# generate_bonus_file(sessions, "bonuses.txt")
# save_responses(sessions, f"../results/responses.txt")

# analyse human experiment data
sim_func = similarity_binary if args.sim_type == "binary" else similarity_continuous
human_data = analyse_sessions(sessions, sim_func)
make_barplots(human_data, plot_dir=args.results_dir, filename="human")


# compare strategies to human data
mus = jnp.array([[0, 0.5], [1, 0.5]])
_, _, obs_history = generate_obs_history(rng_key, mus, args)
results = pd.DataFrame(
    {
        "group": [],
        "imitation": [],
        "agents known": [],
        "groups relevant": [],
        "own group label": [],
        "strategy": [],
    }
)
for name, strat in STRATEGY_DICT.items():
    results = analyse_strategy_humanlikeness(
        rng_key, mus, obs_history, results, strat, name, args
    )
results = pd.concat([results, human_data])
results["group"] = results["group"].astype(int)
make_barplots(results, plot_dir=args.results_dir, filename="models")
