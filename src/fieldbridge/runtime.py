"""Checkpoint, control and plateau contracts shared by trainer and diagnostics."""
import hashlib
import json
import os
from pathlib import Path
import time
import torch


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def paused(output):
    own = read_json(Path(output) / 'control.json', {})
    global_control = read_json('docs/OPTIMIZATION_CONTROL.json', {})
    return own.get('desired') == 'paused' or global_control.get('desired') == 'paused'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


class Plateau:
    def __init__(self, patience=5, relative_delta=.002, state=None):
        self.patience, self.relative_delta = patience, relative_delta
        self.state = state or {'best': None, 'bad_epochs': 0, 'observations': 0}

    def observe(self, score, min_epochs=1):
        if score is None:
            return False
        s = self.state; s['observations'] += 1
        if s['best'] is None or score < s['best'] - max(abs(s['best']) * self.relative_delta, 1e-8):
            s.update(best=score, bad_epochs=0)
        else:
            s['bad_epochs'] += 1
        return bool(self.patience and s['bad_epochs'] >= self.patience and s['observations'] >= min_epochs)


def frozen_keys(model):
    if model.cfg.backbone == 'tiny' or any(p.requires_grad for p in model.frame_encoder.parameters()):
        return set()
    return {k for k in model.state_dict() if k.startswith('frame_encoder.')}


def checkpoint_weights(model, compact=True):
    excluded = frozen_keys(model) if compact else set()
    return {k: v for k, v in model.state_dict().items() if k not in excluded}, bool(excluded)


def load_weights(model, checkpoint, backbone_sha=None):
    if checkpoint.get('compact_model'):
        if checkpoint.get('backbone_checkpoint_sha256') != backbone_sha:
            raise ValueError('Frozen backbone SHA mismatch')
        incompatible = model.load_state_dict(checkpoint['model'], strict=False)
        if set(incompatible.missing_keys) != frozen_keys(model) or incompatible.unexpected_keys:
            raise ValueError('Incomplete or incompatible compact checkpoint')
    else:
        model.load_state_dict(checkpoint['model'], strict=True)


def forward_batch(model, batch, feedback):
    from .data import model_inputs
    inputs = model_inputs(batch)
    if feedback.startswith('oracle'):
        # Deliberately absent from normal inference inputs. Config gates these runs.
        inputs['diagnostic_fields'] = batch['diagnostic_fields'] if feedback == 'oracle_shuffled' else batch['fields'] * batch['field_mask']
    return model(**inputs)
