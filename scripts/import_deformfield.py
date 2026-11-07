"""Index original gt.json + sample_pack without importing legacy training code."""
import argparse
from pathlib import Path
import hashlib
import json
import random


def build(root, output, seed=17, test_objects=None, original_split=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    original = json.loads(Path(original_split).read_text()) if original_split else None
    train_ids = set(original["train_ids"]) if original else set()
    test_ids = set(original["test_ids"]) if original else set()
    if original and (train_ids & test_ids or not train_ids or not test_ids):
        raise ValueError("Invalid original split IDs")
    entries, excluded = [], []
    for gt in sorted(root.rglob("gt.json")):
        pack = gt.parent / "sample_pack.npz"
        if not pack.exists():
            if original and gt.parent.name in train_ids | test_ids:
                raise ValueError(f"Published split sample has no pack: {gt}")
            excluded.append(gt.parent.name)
            continue
        if original and gt.parent.name not in train_ids | test_ids:
            raise ValueError(f"Pack is outside the published split: {gt}")
        d = json.loads(gt.read_text())
        obj = str(d.get("object") or "").strip()
        if not obj:
            raise ValueError(f"Missing object id: {gt}")
        params = d.get("params") or d.get("parameters") or {}
        reg = d.get("regression") or {}
        def get(key):
            return reg[key] if key in reg else params.get(key)
        material = str(params.get("material", d.get("material", ""))).lower()
        y = {"E": get("E"), "nu": get("nu"), "rho": get("density"), "sigma_y": get("yield_stress")}
        # Only law-valid yield labels, never fabricated zeros for missing/non-yielding data.
        if material != "metal":
            y["sigma_y"] = None
        relative = gt.parent.relative_to(root).as_posix()
        entries.append({"sample_id": relative, "object_id": obj, "asset_group": "deformfield:" + obj,
                        "source": "DeformFieldBench", "pack": str(pack), "parameters": y,
                        "law_id": {"jelly": 0, "metal": 1, "plasticine": 2}.get(material, -1),
                        "action": d.get("action", "unknown"), "field_mode": "legacy_visual",
                        "label_provenance": "simulator_config", "field_warning": "visualized flow/stress; contact mask is not force"})
        run_path = gt.parent / "meta" / "run_parameters.json"
        if run_path.exists():
            run = json.loads(run_path.read_text())
            indices, dt = run.get("render_frame_indices"), run.get("frame_dt")
            if indices and dt and float(dt) > 0:
                entries[-1].update(timestamps_seconds=[(i-indices[0])*float(dt) for i in indices],
                                   time_provenance="render_frame_indices * frame_dt; relative elapsed seconds")
        if original:
            entries[-1]["published_split"] = "test" if gt.parent.name in test_ids else "train"
    if original and {Path(r["pack"]).parent.name for r in entries} != train_ids | test_ids:
        raise ValueError("Extracted sample IDs do not exactly match the published split")
    objects = sorted({r["object_id"] for r in entries})
    if len(objects) < 3:
        raise ValueError("Need at least three distinct objects for train/val/test")
    shuffled = objects.copy(); random.Random(seed).shuffle(shuffled)
    if original:
        tests = set(original["test_objects"])
        if tests != {r["object_id"] for r in entries if r["published_split"] == "test"}:
            raise ValueError("Published test object list and sample IDs disagree")
        if tests & {r["object_id"] for r in entries if r["published_split"] == "train"}:
            raise ValueError("Published split has object leakage")
    elif test_objects:
        tests = set(json.loads(Path(test_objects).read_text()))
        if not tests or not tests.issubset(objects):
            raise ValueError("Explicit test object list not in dataset")
    else:
        tests = set(shuffled[:max(1, len(objects)//10)])
    remaining = [o for o in shuffled if o not in tests]
    vals = set(remaining[:max(1, len(objects)//10)])
    for r in entries:
        r["split"] = "test" if r["object_id"] in tests else "val" if r["object_id"] in vals else "train"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Manifest is immutable; use a new filename")
    output.write_text("".join(json.dumps(r) + "\n" for r in entries))
    info = {"seed": seed, "test_objects": sorted(tests), "val_objects": sorted(vals),
            "objects": len(objects), "samples": len(entries),
            "split_counts": {s: sum(r["split"] == s for r in entries) for s in ("train", "val", "test")},
            "excluded_gt_without_pack": excluded,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "published_split_sha256": hashlib.sha256(Path(original_split).read_bytes()).hexdigest() if original else None,
            "protocol": "published test IDs unchanged; validation objects carved only from published train" if original else
                        "explicit test objects" if test_objects else "new deterministic object split; NOT paper split reproduction"}
    output.with_suffix(".split.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True); p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--test-objects", help="JSON array; use paper test objects for reproduction")
    p.add_argument("--original-split", help="Published train_test_split.cleaned.json; exact ID verification")
    a = p.parse_args(); build(a.root, a.output, a.seed, a.test_objects, a.original_split)
