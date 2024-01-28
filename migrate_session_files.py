import os
import json

import jax.numpy as jnp
from tqdm import tqdm

from src.utils import load_sessions, SESSIONS_PATH

EXPERIMENT_IDS = [
    "prolific-test-2",
    "prolific-test-3",
    "prolific-test-4",
    "prolific-test-5",
    "prolific-test-6",
    "prolific-test-7",
    "prolific-test-8",
]


def convert_session(session):
    cond = session["conditions"]
    del session["conditions"]

    session["thetas"] = cond["thetas"]
    session["phi"] = cond["phi"]

    thetas = jnp.array(cond["thetas"])
    dim = int(thetas[0][0] - thetas[0][1] == 0)
    relevant, phi_type = bool(cond["correlation"] == dim), ""

    if cond["phi"] == -1:
        phi_type = "hidden"
    elif not relevant:
        phi_type = "arbitrary"
    else:
        group = 1 - int(jnp.argmin(thetas[dim]))
        phi_type = "matched" if group == cond["phi"] else "mismatched"

    session["condition"] = {"phisRelevant": relevant, "participantPhiType": phi_type}
    return session


sessions = load_sessions(filters={"experimentId": EXPERIMENT_IDS}, print_stats=False)
for s in tqdm(sessions):
    updated = convert_session(s)
    with open(os.path.join(SESSIONS_PATH, f"{s['id']}.json"), "w") as f:
        json.dump(updated, f, indent=2)
