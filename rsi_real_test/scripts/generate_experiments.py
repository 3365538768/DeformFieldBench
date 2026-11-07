"""Generate matched controls; never silently modify architecture/data per variant."""
import argparse
import copy
import json
from pathlib import Path


def generate(base, output):
    cfg = json.loads(Path(base).read_text())
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    variants = {
        "B_rgb": ("none", False, 0.),
        "C_aux_no_feedback": ("none", False, 1.),
        "D_field_bridge": ("fields", False, 1.),
        "E_prior_direct": ("none", True, 1.),
        "F_prior_field": ("fields", True, 1.),
        "G_feature_control": ("features", False, 1.)}
    for seed in (17, 29, 43):
        for name, (feedback, prior, aux) in variants.items():
            c = copy.deepcopy(cfg)
            c["model"].update(feedback=feedback, use_prior=prior)
            c["train"].update(seed=seed, field_weight=aux, output=f"runs/{name}_seed{seed}")
            # B has no field-only warmup; primary causal contrasts C/D and E/F keep schedules matched.
            if aux == 0:
                c["train"]["field_epochs"] = 0
            (output / f"{name}_seed{seed}.json").write_text(json.dumps(c, indent=2))
    print(f"Wrote {len(variants)*3} configs. E/F require independently generated static priors.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--base", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); generate(a.base, a.output)
