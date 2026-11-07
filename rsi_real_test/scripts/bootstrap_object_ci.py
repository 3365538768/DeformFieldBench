"""Paired object bootstrap, never treating adjacent frames as independent."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(a, b, draws=2000, seed=17):
    def load(path):
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        d = {r["sample_id"]: r for r in rows}
        if len(d) != len(rows):
            raise ValueError("Duplicate sample records")
        return d
    first, second = load(a), load(b)
    if set(first) != set(second):
        raise ValueError("Paired runs must evaluate exactly the same samples")
    groups = {}
    for sid, x in first.items():
        y = second[sid]
        if x["object_id"] != y["object_id"] or x["target_mask"] != y["target_mask"] or x["target_z"] != y["target_z"]:
            raise ValueError("Ground truth / statistics mismatch")
        m = np.array(x["target_mask"], bool)
        if not m.any():continue
        target = np.array(x["target_z"])
        ea = np.abs(np.array(x["prediction_z"])-target)[m].mean()
        eb = np.abs(np.array(y["prediction_z"])-target)[m].mean()
        groups.setdefault((x["source"], x["object_id"]), []).append(eb-ea)
    values = np.array([np.mean(v) for v in groups.values()])
    if len(values) < 2:
        raise ValueError("Need at least two independent test objects")
    rng = np.random.default_rng(seed)
    resampled = rng.choice(values, size=(draws,len(values)), replace=True).mean(1)
    return {"objects":len(values), "mean_object_delta_B_minus_A":float(values.mean()),
            "paired_bootstrap_95_ci":np.quantile(resampled,[.025,.975]).tolist(),
            "draws":draws,"seed":seed,"negative_favors":"B", "score":"object-macro standardized MAE"}


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("a");p.add_argument("b");p.add_argument("--draws",type=int,default=2000)
    x=p.parse_args();print(json.dumps(compare(x.a,x.b,x.draws),indent=2))
