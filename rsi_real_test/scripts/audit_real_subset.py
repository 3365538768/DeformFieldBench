from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np


def audit(root, output):
    results = []
    for pack in sorted(Path(root).rglob("sample_pack.npz")):
        gt = json.loads((pack.parent / "gt.json").read_text())
        params = gt.get("params") or gt.get("parameters") or {}
        with np.load(pack, allow_pickle=False) as z:
            arrays = {k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype)} for k in z.files}
            required = ["rgb", "flow", "stress", "force_mask"]
            if not all(k in z for k in required):
                raise ValueError(f"Missing arrays: {pack}")
            shape = z["rgb"].shape
            if not all(z[k].shape[:2] == shape[:2] for k in required):
                raise ValueError("Unsynchronized fields")
        damping = float(params.get("grid_v_damping_scale", 1))
        results.append({"path": str(pack), "object": gt.get("object"), "material": params.get("material"),
                        "sha256": hashlib.sha256(pack.read_bytes()).hexdigest(), "arrays": arrays,
                        "grid_v_damping_scale": damping, "rpic_damping": params.get("rpic_damping"),
                        "example_gamma_if_dt_1e_minus4": -math.log(damping)/1e-4 if 0<damping<1 else 0.,
                        "gamma_note": "Conditional calculation using template dt, NOT measured damping or verified per-sample dt"})
    receipt = {"samples": len(results), "purpose": "format and metadata audit, not physical validity certification", "records": results}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(receipt, indent=2));print(json.dumps({"audited_samples":len(results)}))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();audit(a.root,a.output)
