"""Resumable frozen RGB feature/target cache for diagnostic training."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from fieldbridge.config import load_config
from fieldbridge.data import MaterialDataset, collate, model_inputs
from fieldbridge.model import FrameEncoder, MaterialFieldModel
from fieldbridge.runtime import atomic_json, sha256, paused
from fieldbridge.train import atomic_save


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); a = p.parse_args()
    cfg = load_config(a.config); directory = Path(cfg.data.prepared_dir); directory.mkdir(parents=True, exist_ok=True)
    cfg.data.prepared_dir = ''
    if not cfg.model.freeze_backbone or cfg.model.train_last_blocks:
        raise ValueError('Caching requires a fully frozen backbone')
    meta = {'schema': 1, 'manifest_sha256': sha256(cfg.data.manifest), 'frames': cfg.data.frames,
            'max_views': cfg.data.max_views, 'image_size': cfg.data.image_size,
            'field_size': cfg.model.field_size, 'field_mode': cfg.model.field_mode,
            'backbone_checkpoint_sha256': sha256(cfg.model.weights), 'hub_name': cfg.model.hub_name,
            'encoder_source_sha256': sha256(Path(__file__).resolve().parents[1] / 'src/fieldbridge/model.py'),
            'precision': 'bf16', 'files': {}}
    partial = directory / 'index.partial.json'
    existing = json.loads(partial.read_text()) if partial.exists() else None
    if existing and any(existing.get(k) != v for k, v in meta.items() if k != 'files'):
        raise ValueError('Partial cache key mismatch')
    if existing:
        meta['files'] = existing['files']
    datasets = [MaterialDataset(cfg.data, s, cfg.model.field_size, cfg.model.field_mode) for s in ['train', 'val']]
    encoder = FrameEncoder(cfg.model).cuda().eval(); total = sum(map(len, datasets)); done = 0; started = time.time()
    for ds in datasets:
        for index, row in enumerate(ds.rows):
            if paused('runs/g1_cache'):
                atomic_json(partial, meta)
                atomic_json('docs/G1_CACHE_STATUS.json', {'status': 'paused', 'completed': done, 'total': total})
                return
            sid = row['sample_id']; filename = sha256_text(sid) + '.pt'; path = directory / filename
            if sid not in meta['files'] or not path.exists():
                sample = ds[index]; rgb = sample['rgb']; shape = rgb.shape
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    features = encoder(rgb.flatten(0, 1).cuda()).cpu().reshape(*shape[:2], encoder.dim,
                                                                 shape[-2] // encoder.patch_size, shape[-1] // encoder.patch_size)
                sample['features'] = features
                sample['rgb'] = torch.zeros((*shape[:3], 1, 1), dtype=torch.float32)
                sample['field_mask'] = sample['field_mask'].bool()
                atomic_save(sample, path); meta['files'][sid] = filename
            done += 1
            atomic_json(partial, meta)
            atomic_json('docs/G1_CACHE_STATUS.json', {'status': 'building', 'completed': done, 'total': total,
                        'elapsed_seconds': time.time() - started, 'updated_unix': time.time(),
                        'scope': 'RGB-only frozen features; field targets stored separately for supervision/privileged diagnostics'})
    atomic_json(directory / 'index.json', meta)
    # Compare normal RGB inference and cached RGB-only features on real train/val samples.
    # Ground-truth fields remain targets and never enter these model inputs.
    torch.manual_seed(19)
    model = MaterialFieldModel(cfg.model).cuda().eval(); model.frame_encoder = encoder
    cfg.data.prepared_dir = str(directory)
    maximum = 0.
    for ds in datasets:
        cached = MaterialDataset(cfg.data, ds.split, cfg.model.field_size, cfg.model.field_mode)
        for i in range(min(2, len(ds))):
            raw, prepared = ds[i], cached[i]
            for key in ['fields', 'field_mask', 'target', 'target_mask', 'times', 'valid']:
                if not torch.equal(raw[key], prepared[key]):
                    raise ValueError('Cache target/mask/time mismatch: ' + key)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                inputs = [{k: v.cuda() if torch.is_tensor(v) else v for k, v in model_inputs(collate([s])).items()}
                          for s in [raw, prepared]]
                predictions = [model(**v)['mean'].float() for v in inputs]
            maximum = max(maximum, float((predictions[0] - predictions[1]).abs().max()))
    if maximum > .001:
        raise ValueError('Cached inference deviation exceeds .001 standardized units')
    atomic_json('docs/G1_CACHE_STATUS.json', {'status': 'complete', 'completed': done, 'total': total,
                'elapsed_seconds': time.time() - started, 'bytes': sum(p.stat().st_size for p in directory.glob('*.pt')),
                'parity_max_z_delta': maximum, 'parity_samples': 4,
                'updated_unix': time.time()})


def sha256_text(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


if __name__ == '__main__':
    main()
