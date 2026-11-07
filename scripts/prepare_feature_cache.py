"""Resumable frozen RGB feature/target cache for diagnostic training."""
import argparse
import json
import os
from pathlib import Path
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import torch
from fieldbridge.config import load_config
from fieldbridge.data import MaterialDataset, collate, model_inputs
from fieldbridge.model import FrameEncoder
from fieldbridge.local_model import build_model
from fieldbridge.runtime import atomic_json, sha256, paused
from fieldbridge.train import atomic_save


def samples(dataset, meta, directory, workers):
    """Ordered, bounded CPU prefetch; CUDA stays on the main thread."""
    pending = [i for i,row in enumerate(dataset.rows)
               if row['sample_id'] not in meta['files'] or not (directory/meta['files'][row['sample_id']]).exists()]
    if not workers:
        for i in pending: yield dataset.rows[i], dataset[i]
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator=iter(pending); queue=deque()
        for _ in range(workers):
            i=next(iterator,None)
            if i is not None: queue.append((i,pool.submit(dataset.__getitem__,i)))
        while queue:
            i,future=queue.popleft(); sample=future.result()
            following=next(iterator,None)
            if following is not None: queue.append((following,pool.submit(dataset.__getitem__,following)))
            yield dataset.rows[i], sample


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True)
    p.add_argument('--control-output', default='runs/g1_cache')
    p.add_argument('--prefetch-workers', type=int, default=0)
    p.add_argument('--status-output', default='docs/G1_CACHE_STATUS.json'); a = p.parse_args()
    cfg = load_config(a.config); directory = Path(cfg.data.prepared_dir); directory.mkdir(parents=True, exist_ok=True)
    cfg.data.prepared_dir = ''
    if not cfg.model.freeze_backbone or cfg.model.train_last_blocks:
        raise ValueError('Caching requires a fully frozen backbone')
    if cfg.data.view_dropout:
        raise ValueError('Immutable caches require view_dropout=0; apply augmentation at training load time')
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
    if not 0 <= a.prefetch_workers <= 8: raise ValueError('Prefetch workers must be between 0 and 8')
    encoder = FrameEncoder(cfg.model).cuda().eval(); total = sum(map(len, datasets))
    done = sum(row['sample_id'] in meta['files'] and (directory/meta['files'][row['sample_id']]).exists()
               for ds in datasets for row in ds.rows)
    resumed=done; started = time.time()
    for ds in datasets:
        for row, sample in samples(ds,meta,directory,a.prefetch_workers):
            if paused(a.control_output):
                atomic_json(partial, meta)
                atomic_json(a.status_output, {'status': 'paused', 'completed': done, 'total': total})
                return
            sid = row['sample_id']; filename = sha256_text(sid) + '.pt'; path = directory / filename
            if sid not in meta['files'] or not path.exists():
                rgb = sample['rgb']; shape = rgb.shape
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    features = encoder(rgb.flatten(0, 1).cuda()).cpu().reshape(*shape[:2], encoder.dim,
                                                                 shape[-2] // encoder.patch_size, shape[-1] // encoder.patch_size)
                sample['features'] = features
                sample['rgb'] = torch.zeros((*shape[:3], 1, 1), dtype=torch.float32)
                sample['field_mask'] = sample['field_mask'].bool()
                atomic_save(sample, path); meta['files'][sid] = filename
            done += 1
            atomic_json(partial, meta)
            atomic_json(a.status_output, {'status': 'building', 'completed': done, 'total': total,
                        'elapsed_seconds': time.time() - started, 'updated_unix': time.time(),
                        'prefetch_workers': a.prefetch_workers, 'resumed_samples': resumed,
                        'new_samples_per_second': (done-resumed)/max(time.time()-started,1e-6),
                        'scope': 'RGB-only frozen features; field targets stored separately for supervision/privileged diagnostics'})
    atomic_json(directory / 'index.json', meta)
    # Compare normal RGB inference and cached RGB-only features on real train/val samples.
    # Ground-truth fields remain targets and never enter these model inputs.
    torch.manual_seed(19)
    model = build_model(cfg.model).cuda().eval(); model.frame_encoder = encoder
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
    atomic_json(a.status_output, {'status': 'complete', 'completed': done, 'total': total,
                'elapsed_seconds': time.time() - started, 'bytes': sum(p.stat().st_size for p in directory.glob('*.pt')),
                'parity_max_z_delta': maximum, 'parity_samples': 4,
                'prefetch_workers': a.prefetch_workers, 'resumed_samples': resumed,
                'updated_unix': time.time()})


def sha256_text(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


if __name__ == '__main__':
    main()
