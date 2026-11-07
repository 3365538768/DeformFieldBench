"""Audit predicted visual fields against a constant fitted on training samples."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from fieldbridge.config import load_config, ModelConfig
from fieldbridge.data import MaterialDataset, collate, model_inputs
from fieldbridge.evaluation import move_batch
from fieldbridge.model import MaterialFieldModel
from fieldbridge.runtime import atomic_json, load_weights, sha256


@torch.no_grad()
def run(config, checkpoint, output, fit_samples=64):
    cfg = load_config(config)
    if cfg.model.feedback.startswith('oracle'):
        raise ValueError('Audit deployable predictions only')
    train = MaterialDataset(cfg.data, 'train', cfg.model.field_size, cfg.model.field_mode)
    val = MaterialDataset(cfg.data, 'val', cfg.model.field_size, cfg.model.field_mode)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if ckpt['stats'] != val.codec.stats or asdict(ModelConfig(**ckpt['config']['model'])) != cfg.dump()['model']:
        raise ValueError('Checkpoint/validation contract mismatch')
    if ckpt['manifest_sha256'] != sha256(cfg.data.manifest):
        raise ValueError('Checkpoint/manifest mismatch')
    device = torch.device(cfg.train.device)
    model = MaterialFieldModel(cfg.model).to(device).eval()
    load_weights(model, ckpt, sha256(cfg.model.weights) if cfg.model.weights else None)
    if train.prepared and (train.prepared.get('backbone_checkpoint_sha256') != sha256(cfg.model.weights)
                           or train.prepared.get('encoder_source_sha256') != sha256(Path(__file__).parents[1] / 'src/fieldbridge/model.py')):
        raise ValueError('Cached encoder provenance mismatch')
    channels = 7 if cfg.model.field_mode == 'legacy_visual' else 4
    sums, counts = torch.zeros(channels), torch.zeros(channels)
    reduce = (0, 1, 2, 4, 5)
    chosen = torch.linspace(0, len(train)-1, min(fit_samples, len(train))).round().long().tolist()
    for b in DataLoader(Subset(train, chosen), batch_size=4, collate_fn=collate, num_workers=2):
        sums += (b['fields'] * b['field_mask']).sum(reduce)
        counts += b['field_mask'].sum(reduce)
    constant = (sums / counts.clamp_min(1)).to(device).view(1, 1, 1, channels, 1, 1)
    error = torch.zeros(4, channels, dtype=torch.float64, device=device)
    tp = fp = fn = positive = valid_count = 0
    for batch in DataLoader(val, batch_size=4, collate_fn=collate, num_workers=2):
        b = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.train.val_bf16 and device.type == 'cuda'):
            prediction = model(**model_inputs(b))
        pred, gt, mask = prediction['fields'].float(), b['fields'], b['field_mask']
        error[0] += ((pred - gt).abs() * mask).sum(reduce)
        error[1] += ((constant - gt).abs() * mask).sum(reduce)
        error[2] += mask.sum(reduce)
        error[3] += (pred.square() * mask).sum(reduce)
        support = mask[:, :, :, :1] > 0
        p, g = pred[:, :, :, :1] > .5, gt[:, :, :, :1] > .5
        tp += int((p & g & support).sum()); fp += int((p & ~g & support).sum())
        fn += int((~p & g & support).sum()); positive += int((g & support).sum())
        valid_count += int(support.sum())
    flow = 3 if channels == 7 else 2
    result = {'samples': len(val), 'constant_fit_split': 'train', 'constant_fit_samples': len(chosen),
              'constant_channel_means': constant.flatten().cpu().tolist(),
              'contact_positive_fraction': positive / max(valid_count, 1),
              'contact_iou': tp / max(tp+fp+fn, 1), 'contact_recall': tp / max(tp+fn, 1),
              'contact_precision': tp / max(tp+fp, 1),
              'manifest_sha256': ckpt['manifest_sha256'], 'checkpoint': checkpoint,
              'note': 'Validation-only diagnostic of visual encodings; constant fitted on train only. Not SI physics or test performance.'}
    for name, sl in [('flow', slice(1, 1+flow)), ('stress', slice(1+flow, channels))]:
        den = error[2, sl].sum().clamp_min(1)
        actual = float(error[0, sl].sum() / den)
        baseline = float(error[1, sl].sum() / den)
        result[name] = {'predicted_mae': actual, 'train_constant_mae': baseline,
                        'relative_improvement': 1 - actual / baseline if baseline else None}
    atomic_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ['config', 'checkpoint', 'output']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--fit-samples', type=int, default=64)
    args = parser.parse_args()
    if args.fit_samples < 1:
        parser.error('--fit-samples must be positive')
    run(args.config, args.checkpoint, args.output, args.fit_samples)
