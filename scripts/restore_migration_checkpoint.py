"""Rebuild our trusted legacy checkpoints from verified frozen-backbone exports.

Tensor/optimizer/RNG content is preserved; torch serialization bytes may differ.
Never pass downloads from untrusted sources: legacy state uses pickle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reconstruct(export: Path, backbone: Path) -> tuple[dict, dict]:
    receipt = json.loads(export.with_suffix('.migration.json').read_text())
    if sha256(export) != receipt['export_sha256']:
        raise ValueError('Export SHA256 mismatch')
    if sha256(backbone) != receipt['frozen_backbone_sha256']:
        raise ValueError('Backbone SHA256 mismatch')
    value = torch.load(export, map_location='cpu', mmap=True, weights_only=False)
    embedded = value.pop('migration_frozen_backbone_export')
    if any(receipt.get(key) != val for key, val in embedded.items()):
        raise ValueError('Embedded export receipt disagrees with sidecar')
    if embedded.get('frozen_backbone_exact_tensor_equality') is not True:
        raise ValueError('Source did not verify exact frozen tensor equality')
    cfg = value['config']['model']
    if not cfg['freeze_backbone'] or cfg.get('train_last_blocks'):
        raise ValueError('Backbone was trainable')
    base = torch.load(backbone, map_location='cpu', mmap=True, weights_only=True)
    frozen = {'frame_encoder.net.' + key: val for key, val in base.items()}
    frozen['frame_encoder.rgb_mean'] = torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1)
    frozen['frame_encoder.rgb_std'] = torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1)
    if set(frozen) != set(embedded['excluded_keys']):
        raise ValueError('Frozen key set mismatch')
    state = value['model']
    order = embedded['original_model_key_order']
    if set(frozen) & set(state) or set(order) != set(frozen) | set(state) or len(order) != len(set(order)):
        raise ValueError('Original model key coverage mismatch')
    state.update(frozen)
    value['model'] = {key: state[key] for key in order}
    return value, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only and args.output.exists():
        raise FileExistsError(args.output)
    value, receipt = reconstruct(args.export, args.backbone)
    if not args.verify_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents replacing another process's checkpoint.
        with args.output.open('xb') as stream:
            torch.save(value, stream)
    print(json.dumps({'status': 'verified' if args.verify_only else 'restored',
                      'original_relative_path': receipt['original_relative_path'],
                      'model_tensor_count': len(value['model']),
                      'byte_identical_serialization_claimed': False}))


if __name__ == '__main__':
    main()
