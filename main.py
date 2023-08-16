import numpy as np
import matplotlib.pyplot as plt

from src.utils import (
    load_sessions,
    save_responses,
    generate_bonus_file,
    make_condition_barplots,
    barplot,
)
from src.human import analyse_sessions
from src.modelling import simulate_strategy


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
    # run_human_data_analysis()
    run_model_analysis()


if __name__ == "__main__":
    main()
