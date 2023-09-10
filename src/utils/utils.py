import datetime
import os
import json

import jax.numpy as jnp

SESSIONS_PATH = "/Users/max/Code/human-gridworlds/data/sessions"
PENCE_PER_POINT = 1


def log(msg):
    print(f"[{datetime.datetime.now()}] {msg}")


def load_json(fp):
    with open(fp) as f:
        data = json.load(f)
    return data


def load_sessions(filters):
    sessions = []
    for f in os.listdir(SESSIONS_PATH):
        if f.endswith(".json"):
            try:
                s = load_json(os.path.join(SESSIONS_PATH, f))
                if all([s[k] in v for k, v in filters.items()]):
                    sessions.append(s)
            except:
                continue

    print(f"Loaded {len(sessions)} sessions")
    return sessions


def save_responses(sessions, fname):
    responses = []
    for s in sessions:
        tr = s["textResponses"]
        if not tr:
            continue
        for i, r in enumerate(tr):
            if i >= len(responses):
                responses.append([])
            responses[i].append(r)

    with open(fname, "w") as f:
        for i, resps in enumerate(responses):
            f.write(f"Question {i+1}\n")
            for r in resps:
                f.write(f"{r}\n")
            f.write("\n")


def generate_bonus_file(sessions, fname):
    with open(fname, "w") as f:
        for s in sessions:
            print(s["finalScore"])
            amount = ((s["finalScore"] - 150) * PENCE_PER_POINT) / 100
            if amount > 0:
                f.write(f"{s['context']['PRLFC_ID']},{amount:.2f}\n")


# normalise array have sum 1
def norm_unit_sum(x: jnp.ndarray):
    if x.sum() == 0:
        return jnp.ones_like(x) / len(x)
    return x / x.sum()


def value_similarity(vself, v):
    sims = 1 - jnp.abs(vself - v)
    weights = jnp.abs(vself - 0.5)
    if jnp.sum(weights) == 0:
        return 0
    return jnp.sum(sims * norm_unit_sum(weights))
