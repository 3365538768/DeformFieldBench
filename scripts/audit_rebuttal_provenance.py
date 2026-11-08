"""Audit local rebuttal evidence without loading models or contacting a server.

Published configuration is source evidence, not proof of an actual training run.
Only hashes of optionally supplied manuscript/review files enter the receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


LEGACY_REF = "131c0b063d81b5413691e43ea78fe3ba64e53e9c"
LEGACY_FILES = (
    "configs/logic_model/final_logic_413_5000_new.json",
    "configs/my_model/train_dataset_5000_param_only.json",
    "logic_model/losses.py",
    "logic_model/model2.py",
    "logic_model/train.py",
    "eval/make_ablation_configs.py",
    "scripts/run_vlm_benchmark.sh",
    "vlm_benchmark/vlm_model_registry.py",
)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manuscript", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--legacy-ref", default=LEGACY_REF)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]

    def git(*parts: str) -> bytes:
        return subprocess.check_output(["git", *parts], cwd=repo, stderr=subprocess.PIPE)

    receipt: dict = {
        "schema_version": 1,
        "scope": "local_source_and_saved_diagnostics_only",
        "current_source_commit": git("rev-parse", "HEAD").decode().strip(),
        "legacy_source_commit": args.legacy_ref,
        "remote_contacted": False,
        "gpu_jobs_launched": False,
        "actual_paper_checkpoint_and_runtime_config_verified": False,
        "manuscript_matches_reviewer_seen_version_verified": False,
        "legacy_sources": {},
        "optional_documents": {},
        "diagnostics": {},
    }
    raw_sources: dict[str, bytes] = {}
    for relative in LEGACY_FILES:
        try:
            raw = git("show", f"{args.legacy_ref}:{relative}")
        except subprocess.CalledProcessError:
            receipt["legacy_sources"][relative] = {
                "status": "unavailable_in_local_git_history",
                "action": "Fetch the pinned history before auditing this source.",
            }
            continue
        raw_sources[relative] = raw
        receipt["legacy_sources"][relative] = {"status": "read", "sha256": digest(raw)}

    cfg_path = LEGACY_FILES[0]
    if cfg_path in raw_sources:
        cfg = json.loads(raw_sources[cfg_path])
        train = cfg["train"]
        model = cfg["model"]
        receipt["published_full_model_config_only"] = {
            "source": cfg_path,
            "not_an_actual_training_receipt": True,
            "model": {key: model.get(key) for key in (
                "arch", "num_views", "num_frames", "img_size", "dec_h", "dec_w",
                "dino_backbone_name", "freeze_backbone", "bottleneck_dim",
            )},
            "train": {key: train.get(key) for key in (
                "epochs", "batch_size", "lr", "use_amp", "eval_split",
                "eval_every_epochs", "lambda_stress", "lambda_flow", "lambda_force",
                "lambda_phys", "use_phys_loss", "stage1_use_phys_loss",
                "stage_flow_force_epochs", "stage_stress_epochs", "stage_joint_epochs",
                "lambda_stress_edge", "lambda_flow_edge", "force_bce_weight",
                "force_dice_weight", "use_param_groups",
            )},
            "inferred_final_param_stage_epochs_if_no_cli_override": (
                train["epochs"] - train["stage_flow_force_epochs"]
                - train["stage_stress_epochs"] - train["stage_joint_epochs"]
            ),
        }

    receipt["source_audit_findings"] = [
        {"finding": "Published default disables physics consistency.",
         "source": cfg_path, "runtime_verified": False},
        {"finding": "Legacy consistency aggregates V,T,H,W per sample after abs channel averaging; it is not a per-time mechanical balance.",
         "source": "logic_model/losses.py", "runtime_verified": False},
        {"finding": "Legacy ablation generator has no variant removing only feedback while retaining bottleneck and field supervision.",
         "source": "eval/make_ablation_configs.py", "runtime_verified": False},
        {"finding": "Default VLM wrapper selects qwen3.5-plus; this does not identify the model used for the manuscript table.",
         "source": "scripts/run_vlm_benchmark.sh", "runtime_verified": False},
    ]
    # These findings refer to the pinned source, never silently to another revision.
    if args.legacy_ref != LEGACY_REF:
        receipt["source_audit_findings"] = []
        receipt["manual_review_required"] = "Findings only apply to the default pinned legacy revision."

    for name, path in (("manuscript", args.manuscript), ("reviews", args.reviews)):
        if path:
            receipt["optional_documents"][name] = {
                "sha256": digest(path.read_bytes()), "content_not_republished": True,
            }

    for name, relative in (
        ("M2", "docs/M2_OBSERVED_RESULTS.json"),
        ("G1F", "docs/G1F_RESULTS.json"),
        ("NF", "docs/NF_RESULTS.json"),
    ):
        raw = (repo / relative).read_bytes()
        data = json.loads(raw)
        receipt["diagnostics"][name] = {
            "source": relative, "sha256": digest(raw),
            "saved_updated_unix": data.get("updated_unix"),
            "not_an_exact_original_paper_reproduction": True,
            "scores_lower_is_better": {
                key: value["best_material_mae"] for key, value in data["results"].items()
            },
        }
    m2 = receipt["diagnostics"]["M2"]["scores_lower_is_better"]
    receipt["m2_bridge_relative_error_increase_vs_aux_control"] = (
        m2["D_fields"] / m2["C_aux"] - 1
    )
    receipt["unresolved_required_artifacts"] = [
        "Manuscript submission/revision identity and exact associated tables.",
        "Original model checkpoint hashes, resolved runtime configs and training logs.",
        "Per-sample predictions, exact splits, label transforms and replay receipts.",
        "Actual VLM request/response metadata with credentials removed.",
        "Original simulator versions, effective runtime settings and calibration evidence.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "remote_contacted": False,
                      "actual_paper_run_verified": False,
                      "m2_bridge_relative_error_increase_vs_aux_control":
                      receipt["m2_bridge_relative_error_increase_vs_aux_control"]}))


if __name__ == "__main__":
    main()
