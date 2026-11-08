"""Strict whitelist reader: legacy cached fields never leave the disk payload."""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .data import read_manifest, ParameterCodec
from .memory_cache import TensorLRU
from .runtime import sha256


class FieldFreeDataset(Dataset):
    def __init__(self, cfg, split, field_size=None, field_mode='none'):
        self.cfg, self.split = cfg, split
        self.rows = [r for r in read_manifest(cfg.manifest) if r['split'] == split]
        if not self.rows:
            raise ValueError('Empty direct-video split')
        digest = sha256(cfg.manifest)
        stats = json.loads(Path(cfg.stats).read_text())
        if stats.get('fit_split') != 'train' or stats.get('manifest_sha256') != digest:
            raise ValueError('Direct-video statistics must match the train manifest')
        self.codec = ParameterCodec(stats)
        self.prepared = json.loads((Path(cfg.prepared_dir)/'index.json').read_text())
        expected = dict(manifest_sha256=digest, frames=cfg.frames, max_views=cfg.max_views, image_size=cfg.image_size)
        if any(self.prepared.get(k) != v for k, v in expected.items()):
            raise ValueError('Direct-video cache contract mismatch')
        if any(r['sample_id'] not in self.prepared['files'] for r in self.rows):
            raise ValueError('Direct-video cache coverage mismatch')
        self.memory_cache = TensorLRU(int(cfg.prepared_cache_gib*1024**3))
        self.priors = None
        training_rows = {r['sample_id']: r for r in read_manifest(cfg.manifest) if r['split'] == 'train'}
        training_objects = {r['object_id'] for r in training_rows.values()}
        if cfg.material_prior_file:
            self.priors = json.loads(Path(cfg.material_prior_file).read_text())
            expected = dict(schema='fieldfree_static_prior_v1', manifest_sha256=digest,
                            stats_sha256=sha256(cfg.stats), feature_index_sha256=sha256(Path(cfg.prepared_dir)/'index.json'))
            if any(self.priors.get(k) != v for k, v in expected.items()):
                raise ValueError('Static prior provenance mismatch')
            for r in self.rows:
                p = self.priors['samples'].get(r['sample_id'], {})
                if p.get('object_id') != r['object_id']:
                    raise ValueError('Missing or mismatched static prior')
                if split == 'train' and r['object_id'] in self.priors['fits'][p['fit_id']]['training_objects']:
                    raise ValueError('Static prior leaks its held-out object')
                if set(self.priors['fits'][p['fit_id']]['training_objects']) - training_objects:
                    raise ValueError('Static prior fitted on non-training objects')
        self.teacher = None
        if cfg.material_teacher_file:
            self.teacher = json.loads(Path(cfg.material_teacher_file).read_text())
            expected = dict(schema='fieldfree_offline_teacher_v1', manifest_sha256=digest,
                            stats_sha256=sha256(cfg.stats), physical_admission_passed=True)
            if any(self.teacher.get(k) != v for k, v in expected.items()):
                raise ValueError('Teacher physical admission/provenance mismatch')
            for sid, p in self.teacher['samples'].items():
                if sid not in training_rows or p.get('object_id') != training_rows[sid]['object_id']:
                    raise ValueError('Teacher contains non-training or mismatched sample')
                if not all(p.get(k) for k in ['solver', 'input_sha256', 'fit_protocol', 'response_validation']):
                    raise ValueError('Teacher lacks solver/input/fit/response provenance')
            if split == 'train' and any(r['sample_id'] not in self.teacher['samples'] for r in self.rows):
                raise ValueError('Teacher coverage incomplete; no GT fallback')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(self.cfg.prepared_dir)/self.prepared['files'][row['sample_id']]
        def load():
            payload = torch.load(path, weights_only=True)
            # No RGB proxy, field, law or parameter values from the payload are used.
            return {k: payload[k] for k in ['sample_id', 'features', 'valid', 'times']}
        source = self.memory_cache.get(str(path), load)
        if source['sample_id'] != row['sample_id']:
            raise ValueError('Direct-video sample identity mismatch')
        valid = source['valid']
        cameras = torch.where(valid.any(1))[0]
        if not len(cameras):
            raise ValueError('No valid camera')
        chosen = int(cameras[torch.randint(len(cameras), ())] if self.split == 'train' else cameras[0])
        features, valid, times = [source[k][chosen:chosen+1].clone() for k in ['features','valid','times']]
        if not torch.isfinite(features).all() or not torch.isfinite(times).all():
            raise ValueError('Non-finite direct-video inputs')
        target, mask = self.codec.encode(row['parameters'])
        sample = dict(features=features, valid=valid, times=times, rgb=torch.zeros(*valid.shape, 3, 1, 1),
                      target=target, target_mask=mask, prior=torch.zeros(4), prior_mask=torch.zeros(4, dtype=torch.bool),
                      prior_logvar=torch.zeros(4), law=torch.tensor(row.get('law_id', -1)),
                      sample_id=row['sample_id'], object_id=row['object_id'], source=row['source'],
                      action=row.get('action', 'unknown'), camera_index=chosen)
        if self.priors:
            p = self.priors['samples'][row['sample_id']]
            for key, stored in [('prior','mean_z'),('prior_logvar','logvar_z'),('prior_mask','mask')]:
                sample[key] = torch.tensor(p[stored][chosen], dtype=torch.bool if key=='prior_mask' else torch.float32)
                if sample[key].shape != (4,) or not torch.isfinite(sample[key]).all():
                    raise ValueError('Invalid static prior values')
        if self.teacher:
            p = self.teacher['samples'][row['sample_id']] if self.split == 'train' else None
            values = torch.tensor(p['solutions_z'], dtype=torch.float32) if p else torch.zeros(1,4)
            weights = torch.tensor(p['weights'], dtype=torch.float32) if p else torch.ones(1)
            observed = torch.tensor(p['parameter_mask'], dtype=torch.bool) if p else torch.zeros(4,dtype=torch.bool)
            if (values.shape != (len(weights),4) or not 1<=len(weights)<=16 or observed.shape!=(4,)
                    or not torch.isfinite(values).all() or not torch.isfinite(weights).all() or (weights<0).any() or not weights.sum()>0):
                raise ValueError('Teacher needs 1-16 finite solutions and nonnegative normalized weights')
            padded = torch.zeros(16,4); padded[:len(weights)] = values
            importance = torch.zeros(16); importance[:len(weights)] = weights/weights.sum()
            sample.update(teacher_solutions=padded,teacher_weights=importance,teacher_mask=observed)
        return sample
