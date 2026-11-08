"""Independent reader for PG4_SAMPLE_PACK_v1 and explicit metric-v2 packs.

Legacy RGB flow/stress remain VISUAL supervision. No HSV inversion or SI claims.
All returned tensors use [V,T,C,H,W]. Model inputs are selected separately.
"""
from pathlib import Path
import hashlib
import json
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from .engine_domains import validate_single_parameter_domain
from .paths import path_map, relocate_path
from .memory_cache import TensorLRU

PARAMETERS = ("E", "nu", "rho", "sigma_y")
LAWS = {"jelly": 0, "metal": 1, "plasticine": 2}


def read_manifest(path):
    path = Path(path).resolve()
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    mappings = path_map()
    seen, groups = set(), {}
    for r in rows:
        for k in ("sample_id", "object_id", "split", "source", "pack", "parameters"):
            if k not in r:
                raise ValueError(f"Missing {k}: {r.get('sample_id')}")
        if r["sample_id"] in seen:
            raise ValueError("Duplicate sample id")
        seen.add(r["sample_id"])
        if r["split"] not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        group = r.get("asset_group") or f"{r['source']}:{r['object_id']}"
        groups.setdefault(group, set()).add(r["split"])
        r["pack"] = relocate_path(str((path.parent / r["pack"]).resolve()), mappings)
    leaks = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    if leaks:
        raise ValueError(f"Object leakage across splits: {leaks}")
    if not rows:
        raise ValueError("Empty manifest")
    validate_single_parameter_domain(rows)
    return rows


def transform_parameters(params):
    values, valid = [], []
    for k in PARAMETERS:
        v = params.get(k)
        ok = v is not None and math.isfinite(float(v))
        if ok and k == "nu" and not -1 < float(v) < 0.5:
            raise ValueError("nu must be in (-1, 0.5)")
        if ok and k != "nu" and float(v) < 0:
            raise ValueError("Negative physical parameter")
        values.append((float(v) if k == "nu" else math.log1p(float(v))) if ok else 0.)
        valid.append(ok)
    return np.asarray(values, np.float32), np.asarray(valid, bool)


class ParameterCodec:
    def __init__(self, stats):
        self.stats = stats
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.std = torch.tensor(stats["std"], dtype=torch.float32)
        if (self.std <= 0).any():
            raise ValueError("Invalid train standard deviation")

    def encode(self, params):
        y, valid = transform_parameters(params)
        y, valid = torch.from_numpy(y), torch.from_numpy(valid)
        return torch.where(valid, (y - self.mean) / self.std, 0.), valid

    def unstandardize(self, y):
        return y * self.std.to(y.device) + self.mean.to(y.device)

    def decode(self, y):
        z = self.unstandardize(y).clone()
        for j in (0, 2, 3):
            z[..., j] = torch.expm1(z[..., j].clamp(0, 40))
        z[..., 1] = z[..., 1].clamp(-0.999, 0.4999)
        return z


def fit_stats(manifest, output):
    rows = read_manifest(manifest)
    ys = [transform_parameters(r["parameters"]) for r in rows if r["split"] == "train"]
    if not ys:
        raise ValueError("No training labels")
    mean, std, counts = [], [], []
    for j in range(4):
        x = np.array([y[j] for y, m in ys if m[j]])
        mean.append(float(x.mean()) if len(x) else 0.)
        std.append(max(float(x.std()), 0.05) if len(x) else 1.)
        counts.append(len(x))
    stats = {"mean": mean, "std": std, "count": counts, "transform": "ln1p_E_rho_sigma_y;linear_nu",
             "fit_split": "train", "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest()}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(stats, indent=2))
    return stats


def _image_tensor(a):
    if a.ndim != 5:
        raise ValueError("Expected [V,T,C,H,W]")
    x = torch.from_numpy(np.array(a, copy=True)).float()
    return x / 255. if a.dtype == np.uint8 else x


def _resize(x, size, nearest=False):
    shape = x.shape
    kw = {} if nearest else {"align_corners": False}
    return F.interpolate(x.flatten(0, 1), size=(size, size),
                         mode="nearest" if nearest else "bilinear", **kw).reshape(*shape[:3], size, size)


class MaterialDataset(Dataset):
    def __init__(self, cfg, split, field_size, field_mode):
        self.rows = [r for r in read_manifest(cfg.manifest) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No {split} samples")
        self.cfg, self.split = cfg, split
        self.field_size, self.field_mode = field_size, field_mode
        stats = json.loads(Path(cfg.stats).read_text())
        digest = hashlib.sha256(Path(cfg.manifest).read_bytes()).hexdigest()
        if stats.get("fit_split") != "train" or stats.get("manifest_sha256") != digest:
            raise ValueError("Statistics must be fitted on the current manifest's train split")
        self.codec = ParameterCodec(stats)
        self.prepared = None
        self.motion = None
        if field_mode == 'observed_motion':
            if not cfg.prepared_dir or not cfg.motion_targets_dir:
                raise ValueError('Observed motion needs cached features and explicit pseudo-labels')
            self.motion = json.loads((Path(cfg.motion_targets_dir)/'index.json').read_text())
            expected = dict(schema='observed_motion_v1',manifest_sha256=digest,frames=cfg.frames,
                max_views=cfg.max_views,image_size=cfg.image_size,field_size=field_size,field_mode=field_mode,
                displacement_scale_px=8.,units=['binary_reliability_proxy','pixel','pixel','unavailable'],
                direction='current_to_first_frame; values live on current-frame pixels')
            if any(self.motion.get(k)!=v for k,v in expected.items()) or any(r['sample_id'] not in self.motion['files'] for r in self.rows):
                raise ValueError('Observed-motion contract/coverage mismatch')
        self.memory_cache = TensorLRU(int(cfg.prepared_cache_gib * 1024**3))
        self.rgb_cache=None
        if getattr(cfg,'rgb_cache_dir',''):
            self.rgb_cache=json.loads((Path(cfg.rgb_cache_dir)/'index.json').read_text())
            contract=dict(schema='uint8_rgb_v1',manifest_sha256=digest,frames=cfg.frames,max_views=cfg.max_views,image_size=cfg.image_size)
            if not cfg.prepared_dir or any(self.rgb_cache.get(k)!=v for k,v in contract.items()) or any(r['sample_id'] not in self.rgb_cache['files'] for r in self.rows):
                raise ValueError('RGB cache contract/coverage mismatch')
        if cfg.prepared_dir:
            self.prepared = json.loads((Path(cfg.prepared_dir) / 'index.json').read_text())
            expected = {'manifest_sha256': digest, 'frames': cfg.frames, 'max_views': cfg.max_views,
                        'image_size': cfg.image_size, 'field_size': field_size,
                        'field_mode': 'legacy_visual' if self.motion else field_mode}
            if any(self.prepared.get(k) != v for k, v in expected.items()):
                raise ValueError('Prepared cache/data contract mismatch')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        if self.prepared is not None:
            path = Path(self.cfg.prepared_dir) / self.prepared['files'][r['sample_id']]
            sample = self.memory_cache.get(str(path), lambda: torch.load(path, weights_only=True))
            if sample['sample_id'] != r['sample_id']:
                raise ValueError('Prepared sample identity mismatch')
            if getattr(self,'motion',None):
                target_path = Path(self.cfg.motion_targets_dir)/self.motion['files'][r['sample_id']]
                def load_motion():
                    if hashlib.sha256(target_path.read_bytes()).hexdigest()!=self.motion['sha256'][r['sample_id']]:
                        raise ValueError('Observed-motion file SHA mismatch')
                    return torch.load(target_path,weights_only=True)
                target=self.memory_cache.get('motion:'+str(target_path),load_motion)
                if target['sample_id']!=r['sample_id'] or target['fields'].shape!=(*sample['valid'].shape,4,self.field_size,self.field_size):
                    raise ValueError('Observed-motion identity/shape mismatch')
                if target['field_mask'].shape!=target['fields'].shape or target['field_mask'][:,:,3:].any() or not torch.isfinite(target['fields']).all():
                    raise ValueError('Observed motion cannot provide stress labels or non-finite values')
                sample=dict(sample,fields=target['fields'],field_mask=target['field_mask'])
            # Augmentation must never modify retained samples across epochs.
            sample = dict(sample, valid=sample['valid'].clone(), field_mask=sample['field_mask'].float().clone())
            if self.rgb_cache:
                rgb_path=Path(self.cfg.rgb_cache_dir)/self.rgb_cache['files'][r['sample_id']]
                payload=rgb_path.read_bytes()
                if hashlib.sha256(payload).hexdigest()!=self.rgb_cache['sha256'][r['sample_id']]:raise ValueError('RGB cache SHA mismatch')
                import io
                cached=torch.load(io.BytesIO(payload),weights_only=True)
                if cached['sample_id']!=r['sample_id'] or cached['rgb'].dtype!=torch.uint8 or not torch.equal(cached['valid'],sample['valid']) or not torch.equal(cached['times'],sample['times']):
                    raise ValueError('RGB cache identity/time/mask mismatch')
                if cached['rgb'].shape!=(*sample['valid'].shape,3,self.cfg.image_size,self.cfg.image_size):raise ValueError('RGB cache image shape mismatch')
                sample['rgb']=cached['rgb'].float()/255.
            if getattr(self.cfg,'view_policy','all')=='single':
                available=torch.where(sample['valid'].any(1))[0]
                if not len(available):raise ValueError('No available camera for single-view policy')
                chosen=int(available[torch.randint(len(available),())] if self.split=='train' else available[0])
                # Copy just the selected view so worker IPC/GPU transfer and model
                # computation do not carry two masked cameras or their storage.
                if getattr(self.cfg,'single_view_layout','masked')=='compact':
                    for key in ['rgb','features','fields','field_mask','valid','times','diagnostic_fields']:
                        if key in sample:sample[key]=sample[key][chosen:chosen+1].clone()
                else:
                    keep=torch.zeros_like(sample['valid']);keep[chosen]=sample['valid'][chosen]
                    sample['valid']=keep;sample['field_mask']*=keep[:,:,None,None,None]
                sample['camera_index']=chosen
            if self.split == 'train' and self.cfg.view_dropout:
                v = sample['valid'].shape[0]
                keep = torch.rand(v) >= self.cfg.view_dropout; keep[torch.randint(v, ())] = True
                sample['valid'] &= keep[:, None]
                sample['field_mask'] *= sample['valid'][:, :, None, None, None]
            return sample
        mode = r.get("field_mode", "legacy_visual")
        if mode != self.field_mode:
            raise ValueError("Mixed field encodings; use separate experiments")
        if r.get("prior_parameters") and not r.get("prior_provenance"):
            raise ValueError("Static priors require model/input provenance")
        with np.load(r["pack"], allow_pickle=False) as p:
            raw_rgb = p["rgb"]
            v, t = min(raw_rgb.shape[0], self.cfg.max_views), raw_rgb.shape[1]
            n = min(t, self.cfg.frames)
            idx = torch.linspace(0, t - 1, n).round().long()
            idx_np = idx.numpy()
            # Slice uint8 arrays before conversion: avoid making float copies of unused frames.
            rgb = _resize(_image_tensor(raw_rgb[:v, idx_np]), self.cfg.image_size)
            del raw_rgb
            if "timestamps_seconds" in r:
                times = torch.tensor(r["timestamps_seconds"], dtype=torch.float32)
                if times.ndim != 1 or len(times) != t or not torch.isfinite(times).all() or not (times[1:] > times[:-1]).all():
                    raise ValueError("Invalid physical timestamps in manifest")
                times = times[idx].repeat(v, 1)
            elif "timestamps" in p:
                times = torch.tensor(p["timestamps"], dtype=torch.float32)
                times = times[:v, idx] if times.ndim == 2 else times[idx].repeat(v, 1)
            else:
                # Normalized sequence time, explicitly NOT seconds.
                times = (idx.float() / max(t - 1, 1)).repeat(v, 1)
            if mode == "legacy_visual":
                flow = _image_tensor(p["flow"][:v, idx_np])
                stress = _image_tensor(p["stress"][:v, idx_np])
                if flow.shape[2] != 3 or stress.shape[2] != 3:
                    raise ValueError("Legacy visual fields must have three channels")
                force = _image_tensor(p["force_mask"][:v, idx_np])
                if (force.max(2).values - force.min(2).values).max() < 1e-5:
                    contact = force[:, :, :1]
                else:
                    contact = (force[:, :, :1] - force[:, :, 1:].amax(2, keepdim=True)).clamp(0, 1)
                contact = (contact >= .5).float()
                fields = torch.cat((contact, flow, stress), dim=2)
                mask = torch.ones_like(fields)
                if "object_mask" in p:
                    fg = (_image_tensor(p["object_mask"][:v, idx_np]).amax(2, keepdim=True) > .5).float()
                    mask[:, :, 1:] *= fg
            else:
                if r.get("field_units") != ["binary", "pixel_per_second", "pixel_per_second", "pascal"]:
                    raise ValueError("Metric fields need explicit units")
                fields = torch.tensor(p["fields"][:v][:, idx.numpy()], dtype=torch.float32)
                mask = torch.tensor(p["field_mask"][:v][:, idx.numpy()], dtype=torch.float32)
                if fields.shape[2] != 4:
                    raise ValueError("Metric fields: contact, vx, vy, stress")
                scales = r.get("field_scales")
                if not scales or min(scales) <= 0 or len(scales) != 2:
                    raise ValueError("Set fixed train-derived flow/stress scales")
                fields[:, :, 1:3] /= scales[0]
                fields[:, :, 3:] = torch.log1p(fields[:, :, 3:].clamp_min(0)) / scales[1]
            # Values retain original input-image coordinate units during sampling.
            contact_target = fields[:, :, :1].clone()
            fields = _resize(fields, self.field_size)
            fields[:, :, :1] = _resize(contact_target, self.field_size, True)
            mask = _resize(mask, self.field_size, True)
        if not torch.isfinite(fields).all() or not torch.isfinite(rgb).all():
            raise ValueError(f"Non-finite arrays: {r['sample_id']}")
        valid = torch.ones(v, n, dtype=torch.bool)
        if self.split == "train" and self.cfg.view_dropout:
            keep = torch.rand(v) >= self.cfg.view_dropout
            keep[torch.randint(v, ())] = True
            valid &= keep[:, None]
        mask *= valid[:, :, None, None, None]
        target, target_mask = self.codec.encode(r["parameters"])
        # Priors must come from inference-only static inputs, never GT parameters.
        prior, prior_mask = self.codec.encode(r.get("prior_parameters", {}))
        return {"rgb": rgb, "valid": valid, "times": times, "fields": fields, "field_mask": mask,
                "target": target, "target_mask": target_mask, "prior": prior, "prior_mask": prior_mask,
                "law": torch.tensor(r.get("law_id", -1)), "sample_id": r["sample_id"],
                "object_id": r["object_id"], "source": r["source"], "action": r.get("action", "unknown")}


def collate(samples):
    v = max(x["rgb"].shape[0] for x in samples)
    t = max(x["rgb"].shape[1] for x in samples)
    result = {}
    variable = {"rgb", "fields", "field_mask", "valid", "times", "features", "diagnostic_fields"}
    for k in samples[0]:
        if k in variable:
            tensors = []
            for s in samples:
                x = s[k]
                if x.shape[:2] == (v, t):
                    tensors.append(x)
                    continue
                y = x.new_zeros((v, t, *x.shape[2:]))
                y[:x.shape[0], :x.shape[1]] = x
                tensors.append(y)
            result[k] = torch.stack(tensors)
        elif torch.is_tensor(samples[0][k]):
            result[k] = torch.stack([s[k] for s in samples])
        else:
            result[k] = [s[k] for s in samples]
    return result


def model_inputs(batch):
    keys = ("rgb", "valid", "times", "prior", "prior_mask")
    return {k: batch[k] for k in (*keys, 'features') if k in batch}


class ShuffledOracleDataset(Dataset):
    """Fixed split-local donor fields; requires prepared cache to avoid double decode."""
    def __init__(self, dataset, seed=1729):
        self.dataset = dataset; self.rows, self.codec = dataset.rows, dataset.codec
        indices = list(range(len(dataset))); random.Random(seed).shuffle(indices)
        self.donors = {}
        for pos, i in enumerate(indices):
            for offset in range(1, len(indices)):
                donor = indices[(pos + offset) % len(indices)]
                if self.rows[i]['object_id'] != self.rows[donor]['object_id']:
                    self.donors[i] = donor; break
            if i not in self.donors:
                raise ValueError('Shuffled oracle needs multiple objects in each split')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        sample, donor = self.dataset[i], self.dataset[self.donors[i]]
        fields = donor['fields'] * donor['field_mask']
        # The diagnostic protocol uses fixed V/T. Reject silent shape truncation.
        if fields.shape != sample['fields'].shape:
            raise ValueError('Oracle donor shape mismatch')
        sample['diagnostic_fields'] = fields
        return sample
