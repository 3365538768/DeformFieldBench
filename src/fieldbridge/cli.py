import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from .config import load_config, ModelConfig
from .data import read_manifest, fit_stats, MaterialDataset, ShuffledOracleDataset
from .evaluation import evaluate
from .local_model import build_model
from .runtime import load_weights, sha256


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("manifest")
    stats = sub.add_parser("fit-stats")
    stats.add_argument("manifest"); stats.add_argument("output")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--config", required=True); ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--split", choices=["val", "test"], default="val")
    ev.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "fit-stats":
        print(json.dumps(fit_stats(args.manifest, args.output), indent=2))
    elif args.command == "audit":
        rows = read_manifest(args.manifest)
        missing = [r["sample_id"] for r in rows if not Path(r["pack"]).is_file()]
        if missing:
            raise FileNotFoundError(str(missing[:20]))
        print(json.dumps({"samples": len(rows), "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
                          "objects": len({(r["source"], r["object_id"]) for r in rows}), "leakage": False}))
    else:
        cfg = load_config(args.config)
        device = torch.device(cfg.train.device)
        dataset = MaterialDataset(cfg.data, args.split, cfg.model.field_size, cfg.model.field_mode)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        motion_hash=sha256(Path(cfg.data.motion_targets_dir)/'index.json') if cfg.data.motion_targets_dir else None
        if ckpt.get('motion_target_index_sha256')!=motion_hash:
            raise ValueError('Evaluation observed-motion supervision mismatch')
        if ckpt["stats"] != dataset.codec.stats or asdict(ModelConfig(**ckpt["config"]["model"])) != cfg.dump()["model"]:
            raise ValueError("Evaluation checkpoint/config/statistics mismatch")
        backbone_sha = sha256(cfg.model.weights) if cfg.model.weights else None
        if dataset.prepared:
            if dataset.prepared.get('backbone_checkpoint_sha256') != backbone_sha or dataset.prepared.get('hub_name') != cfg.model.hub_name or dataset.prepared.get('encoder_source_sha256') != sha256(Path(__file__).with_name('model.py')):
                raise ValueError('Prepared feature backbone/implementation mismatch')
        if cfg.model.feedback == 'oracle_shuffled':
            dataset = ShuffledOracleDataset(dataset)
        model = build_model(cfg.model).to(device)
        load_weights(model, ckpt, backbone_sha)
        print(json.dumps(evaluate(model, dataset, device, args.output,
                         cfg.train.val_batch_size, cfg.train.val_workers, cfg.train.val_bf16), indent=2))


if __name__ == "__main__":
    main()
