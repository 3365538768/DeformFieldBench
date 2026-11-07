"""Join independent static predictions by sample id; GT is not a fallback."""
import argparse
import json
from pathlib import Path


def join(manifest, predictions, output):
    rows = [json.loads(l) for l in Path(manifest).read_text().splitlines() if l.strip()]
    priors = [json.loads(l) for l in Path(predictions).read_text().splitlines() if l.strip()]
    indexed = {}
    for p in priors:
        for key in ("sample_id", "parameters", "model", "checkpoint_sha256", "input_sha256", "training_data_protocol"):
            if key not in p:
                raise ValueError(f"Prediction missing provenance {key}")
        if p["sample_id"] in indexed:
            raise ValueError("Duplicate prediction")
        if p["training_data_protocol"] not in {"external_pretrained", "train_split_only", "train_object_out_of_fold"}:
            raise ValueError("Unknown prior training provenance")
        indexed[p["sample_id"]] = p
    for r in rows:
        if r["sample_id"] not in indexed:
            raise ValueError(f"Missing prior: {r['sample_id']}; never fill with GT")
        p = indexed[r["sample_id"]]
        r["prior_parameters"] = p["parameters"]
        r["prior_provenance"] = {k: v for k,v in p.items() if k not in {"parameters", "sample_id"}}
        # Resolve pack paths relative to the original manifest, not the joined output.
        r["pack"] = str((Path(manifest).resolve().parent / r["pack"]).resolve())
    target = Path(output)
    if target.exists():
        raise FileExistsError("Use a new immutable manifest")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(r)+"\n" for r in rows))
    print(f"Joined {len(rows)} independent priors. Refit train statistics for this manifest hash.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--manifest", required=True)
    p.add_argument("--predictions", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); join(a.manifest, a.predictions, a.output)
