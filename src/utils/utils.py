import os
import json

import numpy as np

SESSIONS_PATH = "/Users/max/Code/human-gridworlds/data/sessions"
PENCE_PER_POINT = 1


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


def value_similarity(vself, v):
    sims = 1 - np.abs(vself - v)
    weights = np.abs(vself - 0.5)
    weights /= np.sum(weights)
    return sims, np.sum(sims * weights)
