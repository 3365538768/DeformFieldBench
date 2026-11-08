"""Video system identification with no field heads, targets or feedback route."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import FrameEncoder
from .motion_model import temporal_differences


def mixture_moments(logits, means, logvars):
    weights = logits.float().softmax(-1)[..., None]
    mean = (weights * means.float()).sum(-2)
    variance = (weights * (logvars.float().exp() + means.float().square())).sum(-2) - mean.square()
    return mean, variance.clamp_min(1e-8).log()


def mixture_log_prob(logits, means, logvars, target, mask):
    """Joint diagonal-mixture density, exactly marginalizing missing dimensions."""
    logvars = logvars.float()
    density = -.5 * ((target.float()[..., None, :] - means.float()).square() * (-logvars).exp()
                       + logvars + math.log(2 * math.pi))
    return torch.logsumexp(logits.float().log_softmax(-1) + (density * mask[..., None, :]).sum(-1), -1)


class DirectVideoModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.frame_encoder = FrameEncoder(cfg)
        d = cfg.width
        self.project = nn.Conv2d(self.frame_encoder.dim, d, 1)
        self.motion_mix = nn.Sequential(nn.Conv2d(3*d, d, 1), nn.GELU(),
                                       nn.Conv2d(d, d, 3, padding=1, groups=d))
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4*d, batch_first=True,
                                           norm_first=True, dropout=0., activation='gelu')
        self.temporal_views = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.spatial_position = nn.Parameter(torch.zeros(1, 16, 1, d))
        self.evidence_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))
        self.uncertainty_head = nn.Linear(d, 8*cfg.posterior_components + cfg.posterior_components)
        self.prior_encoder = nn.Sequential(nn.Linear(12, d), nn.GELU(), nn.Linear(d, 4))
        self.prior_gate = nn.Linear(d, 4)
        self.law_head = nn.Linear(d, 3)

    def forward(self, rgb, valid, times, prior, prior_mask, features=None, prior_logvar=None):
        b, v, t = valid.shape
        if not valid.flatten(1).any(1).all() or not torch.isfinite(times).all():
            raise ValueError('Need valid video observations and finite times')
        if features is None:
            frame = self.frame_encoder(rgb.flatten(0, 2))
        else:
            if any(p.requires_grad for p in self.frame_encoder.parameters()):
                raise ValueError('Cached features require frozen backbone')
            if features.shape[:4] != (b, v, t, self.frame_encoder.dim):
                raise ValueError('Direct-video feature shape mismatch')
            frame = features.flatten(0, 2).to(self.project.weight.dtype)
        projected = self.project(frame)
        d, h, w = projected.shape[1:]
        clean, departure, adjacent = temporal_differences(projected.reshape(b, v, t, d, h, w), valid)
        local = clean.flatten(0, 2) + self.motion_mix(torch.cat((clean, departure, adjacent), 3).flatten(0, 2))
        local = local * valid.flatten()[:, None, None, None]
        grid = F.adaptive_avg_pool2d(local, (4, 4)).reshape(b, v, t, d, 16)
        sequence = grid.permute(0, 1, 4, 2, 3).reshape(b*v*16, t, d)
        frequency = torch.exp(torch.arange(0, d, 2, device=times.device).float() * (-math.log(10000.)/d))
        phase = times[..., None] * frequency
        position = torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)
        position = (position[:, :, None] + self.spatial_position[:, None]).expand(b, v, 16, t, d)
        mask = valid[:, :, None].expand(b, v, 16, t).reshape(b*v*16, t)
        safe = mask.clone(); safe[~safe.any(1), 0] = True
        encoded = self.temporal_views(sequence + position.reshape_as(sequence).to(sequence.dtype),
                                      src_key_padding_mask=~safe)
        pooled = (encoded * mask[..., None]).reshape(b, v, 16, t, d).sum((1, 2, 3))
        pooled = pooled / (valid.sum((1, 2)).clamp_min(1)*16)[:, None]
        pm = prior_mask.to(pooled.dtype) * float(self.cfg.use_prior)
        safe_prior = torch.where(pm.bool(), prior, 0.).to(pooled.dtype)
        lv = torch.zeros_like(prior) if prior_logvar is None else prior_logvar
        safe_lv = torch.where(pm.bool(), lv, 0.).to(pooled.dtype)
        # Static evidence enters only the parameter estimator, never a field branch.
        correction = self.prior_encoder(torch.cat((safe_prior, safe_lv, pm), -1))
        correction = correction * pm
        center = self.evidence_head(pooled)
        gate = self.prior_gate(pooled).sigmoid() * pm
        center = center + gate * (safe_prior + correction - center)
        k = self.cfg.posterior_components
        raw = self.uncertainty_head(pooled)
        logits = raw[:, :k]
        offsets, logvars = raw[:, k:].reshape(b, k, 8).chunk(2, -1)
        # Center offsets to keep the point prediction independent of NLL training.
        offsets = offsets - (logits.softmax(-1)[..., None]*offsets).sum(1, keepdim=True)
        means = center[:, None] + offsets
        logvars = logvars.clamp(-6, 4)
        mean, logvar = mixture_moments(logits, means, logvars)
        return dict(mean=mean, logvar=logvar, law_logits=self.law_head(pooled),
                    mixture_logits=logits, mixture_means=means, mixture_logvars=logvars)
