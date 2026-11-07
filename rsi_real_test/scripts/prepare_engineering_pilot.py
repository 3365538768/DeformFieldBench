"""Create a fixed development pilot using published TRAIN IDs only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import tempfile
from import_deformfield import build
from fieldbridge.data import fit_stats


def run(root, split, output, base):
    original = json.loads(Path(split).read_text())
    objects = sorted(original["stats"]["object_counts_all"])
    random.Random(17).shuffle(objects)
    test_objects = set(original["test_objects"])
    val_objects = set([o for o in objects if o not in test_objects][:max(1, len(objects)//10)])
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Immutable pilot manifest already exists")
    with tempfile.TemporaryDirectory() as tmp:
        temporary = Path(tmp) / "all.jsonl"
        build(root, temporary)
        rows = [json.loads(l) for l in temporary.read_text().splitlines()]
    for r in rows:
        if Path(r["pack"]).parent.name not in original["train_ids"] or r["object_id"] in test_objects:
            raise ValueError("Pilot must not contain published test samples")
        r["split"] = "val" if r["object_id"] in val_objects else "train"
    if any(sum(r["split"] == s for r in rows) < 2 for s in ("train", "val")):
        raise ValueError("Insufficient train/val pilot coverage")
    output.write_text("".join(json.dumps(r)+"\n" for r in rows))
    stats = output.with_suffix(".stats.json")
    fit_stats(output, stats)
    cfg = json.loads(Path(base).read_text())
    # Engineering optimization check, NOT the final backbone or result protocol.
    cfg["model"].update(backbone="dinov2", hub_name="dinov2_vits14_reg",
        weights="weights/dinov2_vits14_reg4_pretrain.pth", width=256, field_size=56, frame_chunk=16)
    cfg["data"].update(manifest=str(output), stats=str(stats), frames=16)
    cfg["train"].update(epochs=12, field_epochs=2, workers=3, accumulation=1, max_steps=0, purpose="pilot")
    for name, feedback in (("C_aux_no_feedback", "none"), ("D_field_bridge", "fields")):
        c = json.loads(json.dumps(cfg))
        c["model"]["feedback"] = feedback
        c["train"]["output"] = "runs/engineering_pilot_240_" + name
        Path("configs/pilot_"+name+".json").write_text(json.dumps(c, indent=2))
    receipt = {"samples": len(rows), "objects": len({r["object_id"] for r in rows}),
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "val_objects": sorted(val_objects), "published_test_excluded": True,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "purpose": "Nonrandom partial-archive engineering pilot. DINOv2-S, one seed, validation only. Cannot substantiate final bridge superiority."}
    output.with_suffix(".pilot.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for name in ("root", "split", "output", "base"):
        p.add_argument("--"+name,required=True)
    a=p.parse_args();run(a.root,a.split,a.output,a.base)
