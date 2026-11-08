"""Rebuild from raw packs, verify, then publish cache aliases without deleting history."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from fieldbridge.config import load_config
from fieldbridge.data import MaterialDataset
from fieldbridge.runtime import atomic_json, paused, read_json, sha256


def verify_cache(directory, expected_ids, checkpoint):
    """Read every artifact before publishing; a partial index is never acceptance."""
    index = read_json(directory / 'index.json', {})
    if set(index.get('files', {})) != set(expected_ids):
        raise ValueError('Cache IDs differ from train/validation manifest')
    hashes = {}
    for sid, filename in index['files'].items():
        checkpoint(len(hashes), len(expected_ids))
        path = (directory / filename).resolve()
        if path.parent != directory.resolve():
            raise ValueError('Cache filename escapes cache directory')
        sample = torch.load(path, map_location='cpu', weights_only=False)
        if sample['sample_id'] != sid or not torch.isfinite(sample['features']).all():
            raise ValueError('Invalid cached identity/features: ' + sid)
        if sample['features'].shape[:2] != sample['valid'].shape:
            raise ValueError('Feature/view/time dimensions disagree: ' + sid)
        for key in ['fields', 'times', 'target', 'prior']:
            if not torch.isfinite(sample[key]).all():
                raise ValueError('Nonfinite cached supervision: ' + sid)
        hashes[filename] = sha256(path)
    return hashes


def activate_alias(alias, directory):
    """Preserve any copied directory and atomically replace only a cache alias."""
    alias, directory = Path(alias).absolute(), directory.resolve()
    prepared = Path('data/prepared').resolve()
    if alias.parent.resolve() != prepared or not directory.is_relative_to(prepared):
        raise ValueError('Cache activation must stay within data/prepared')
    if alias == directory:
        raise ValueError('Cache cannot alias itself')
    backup = None
    if alias.exists() and not alias.is_symlink():
        backup = alias.with_name(alias.name + '_migration_retained_' + str(time.time_ns()))
        alias.rename(backup)
    temporary = alias.with_name(alias.name + '.rebuild-link')
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(directory, target_is_directory=True)
    os.replace(temporary, alias)
    return {'alias': str(alias), 'target': str(directory),
            'retained_copy': str(backup) if backup else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--control-output', required=True)
    p.add_argument('--status-output', required=True)
    p.add_argument('--activate', action='append', default=[])
    p.add_argument('--prefetch-workers', type=int, default=2)
    a = p.parse_args()
    output = Path(a.control_output); output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.rebuild.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    cfg = load_config(a.config)
    directory = Path(cfg.data.prepared_dir)
    raw_status = output / 'extraction.json'
    directory.mkdir(parents=True, exist_ok=True)
    # Raw pack existence and train/validation IDs are checked before loading CUDA.
    raw_cfg = cfg.data; raw_cfg.prepared_dir = ''
    datasets = [MaterialDataset(raw_cfg, s, cfg.model.field_size, cfg.model.field_mode)
                for s in ['train', 'val']]
    ids = [r['sample_id'] for ds in datasets for r in ds.rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate cache IDs')
    if any(not Path(r['pack']).is_file() for ds in datasets for r in ds.rows):
        raise ValueError('Raw dataset packs are missing')

    def report(status, completed, detail, **extra):
        value = dict(status=status, completed=completed, total=len(ids), detail=detail,
                     updated_unix=time.time(), elapsed_seconds=time.time()-started,
                     directory=str(directory), generated_on_host=os.uname().nodename, **extra)
        atomic_json(a.status_output, value)
        return value

    def wait_for_resume(completed=0):
        while paused(output):
            report('paused', completed, '已在样本边界暂停；继续后复用本机已生成的完整样本。')
            time.sleep(2)

    try:
        while True:
            wait_for_resume(read_json(raw_status, {}).get('completed', 0))
            args = [sys.executable, 'scripts/prepare_feature_cache.py', '--config', a.config,
                    '--control-output', str(output), '--status-output', str(raw_status),
                    '--prefetch-workers', str(a.prefetch_workers)]
            child = subprocess.Popen(args)
            try:
                while child.poll() is None:
                    raw = read_json(raw_status, {})
                    report('paused' if paused(output) else 'running', raw.get('completed', 0),
                           '从原始视频重新提取冻结 RGB 特征；完成后校验并切换缓存。', extraction=raw)
                    time.sleep(2)
            finally:
                if child.poll() is None:
                    child.terminate(); child.wait()
            raw = read_json(raw_status, {})
            if child.returncode != 0:
                raise RuntimeError('Feature extraction failed, exit=' + str(child.returncode))
            if raw.get('status') == 'complete':
                break
            if raw.get('status') != 'paused':
                raise RuntimeError('Feature extraction has no successful parity receipt')

        def checkpoint(done, total):
            wait_for_resume(total)
            report('running', total, f'RGB/缓存预测一致性已通过；逐文件验收 {done}/{total}。',
                   verified_files=done, extraction=raw)
        hashes = verify_cache(directory, ids, checkpoint)
        atomic_json(output / 'artifact_hashes.json', hashes)
        wait_for_resume(len(ids))
        aliases = [activate_alias(path, directory) for path in a.activate]
        value = report('complete', len(ids), '本机重建完成；所有文件验收通过，已切换训练缓存路径。',
                       verified_files=len(hashes), extraction=raw, aliases=aliases,
                       manifest_sha256=sha256(cfg.data.manifest),
                       artifact_hash_receipt=str(output / 'artifact_hashes.json'))
        atomic_json(output / 'status.json', value)
    except Exception as error:
        value = report('failed', read_json(raw_status, {}).get('completed', 0), str(error), reason=str(error))
        atomic_json(output / 'status.json', value)
        raise


if __name__ == '__main__':
    main()
