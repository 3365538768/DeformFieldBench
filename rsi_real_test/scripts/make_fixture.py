"""Deterministic synthetic tensors for software tests, NOT physical simulation."""
from pathlib import Path
import json
import numpy as np


def create(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    rows = []
    for i in range(8):
        path = root / f"fixture_{i}.npz"
        v, t, h = 1 + i % 2, 2 + i % 3, 32
        rgb = rng.integers(0, 256, size=(v, t, 3, h, h), dtype=np.uint8)
        contact = np.repeat((rgb[:, :, :1] > 200).astype(np.uint8) * 255, 3, axis=2)
        np.savez_compressed(path, rgb=rgb, flow=rgb, stress=255-rgb, force_mask=contact,
                            object_mask=np.full_like(rgb, 255))
        rows.append({"sample_id": f"fixture_{i}", "object_id": f"fixture_object_{i}", "source": "SOFTWARE_FIXTURE",
                     "split": "train" if i < 6 else "val" if i == 6 else "test", "pack": str(path),
                     "parameters": {"E": 1000. * (i+1), "nu": .25 + i*.02, "rho": 500.+100*i,
                                    "sigma_y": None if i%2 == 0 else 100.+i*20},
                     "law_id": i%3, "field_mode": "legacy_visual"})
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r)+"\n" for r in rows))
    return manifest


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True)
    print(create(p.parse_args().output))
