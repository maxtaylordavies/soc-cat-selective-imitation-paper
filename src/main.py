import matplotlib.pyplot as plt

from src.utils import (
    load_sessions,
    save_responses,
    generate_bonus_file,
    make_condition_barplots,
)
from src.human.analysis import analyse_sessions


def main():
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

    # generate_bonus_file(sessions, "bonuses.txt")
    # save_responses(sessions, f"../results/responses.txt")

    flavour = "binary"
    data = analyse_sessions(sessions, flavour=flavour)
    make_condition_barplots(data, flavour=flavour)


if __name__ == "__main__":
    main()
