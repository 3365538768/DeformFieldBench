"""Local temporal residual branch beside a frozen RGB estimator.

The legacy implementation remains unchanged to preserve historical checkpoints and
the byte-level frozen-encoder cache contract. New modules share its named heads.
"""
from dataclasses import asdict, replace
import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import ModelConfig
from .model import MaterialFieldModel
from .runtime import load_weights


class LocalResidualModel(MaterialFieldModel):
    FROZEN = ('frame_encoder', 'project', 'temporal_views', 'context', 'prior_encoder',
              'gate', 'param_head', 'law_head')

    def __init__(self, cfg):
        super().__init__(cfg)
        d = cfg.width
        for name in self.FROZEN:
            getattr(self, name).requires_grad_(False)
        self.spatial_adapter = nn.Sequential(nn.Conv2d(d, d, 3, padding=1, groups=d),
                                             nn.GELU(), nn.Conv2d(d, d, 1))
        self.motion_adapter = nn.Sequential(nn.Conv1d(d, d, 3, padding=1, groups=d),
                                            nn.GELU(), nn.Conv1d(d, d, 1))
        self.local_attention = nn.MultiheadAttention(d, cfg.heads, batch_first=True, dropout=0.)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 2*d, dropout=0., batch_first=True,
                                           norm_first=True, activation='gelu')
        self.bridge_temporal = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.confidence = nn.Linear(d, 1)
        self.residual_head = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, 8))
        self.residual_law = nn.Linear(2*d, 3)
        nn.init.zeros_(self.residual_head[-1].weight); nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.residual_law.weight); nn.init.zeros_(self.residual_law.bias)
        nn.init.constant_(self.confidence.bias, -2.)

    def train(self, mode=True):
        super().train(mode)
        for name in self.FROZEN:
            getattr(self, name).eval()
        return self

    def _position(self, times, dtype):
        freq = torch.exp(torch.arange(0, self.cfg.width, 2, device=times.device).float()
                         * (-math.log(10000.) / self.cfg.width))
        phase = times[..., None] * freq
        return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2).to(dtype)

    def forward(self, rgb, valid, times, prior, prior_mask, features=None, diagnostic_fields=None):
        b, v, t, c, h, w = rgb.shape
        if not valid.flatten(1).any(1).all() or not torch.isfinite(times).all():
            raise ValueError('Every sample needs valid observations and finite times')
        flat_mask = valid.reshape(b, v*t)
        with torch.no_grad():
            if features is not None:
                if any(p.requires_grad for p in self.frame_encoder.parameters()):
                    raise ValueError('Cached features require a frozen backbone')
                if features.shape[:4] != (b, v, t, self.frame_encoder.dim):
                    raise ValueError('Cached feature shape mismatch')
                frame = features.flatten(0, 2).to(self.project.weight.dtype)
            else:
                frame = self.frame_encoder(rgb.reshape(-1, c, h, w))
            projected = self.project(frame)
            tokens = projected.mean((-1, -2)).reshape(b, v*t, -1)
            position = self._position(times.reshape(b, v*t), tokens.dtype)
            encoded = self.temporal_views(tokens + position, src_key_padding_mask=~flat_mask)
            weights = flat_mask[..., None].to(encoded.dtype)
            pooled = (encoded*weights).sum(1) / weights.sum(1).clamp_min(1)
            local = projected + self.context(encoded).reshape(b*v*t, -1, 1, 1)
            pm = prior_mask.to(prior.dtype) * float(self.cfg.use_prior)
            prior_input = prior * pm
            prior_embedding = self.prior_encoder(torch.cat((prior_input, pm), -1))
            base_joint = torch.cat((pooled, torch.zeros_like(pooled), prior_embedding), -1)
            base_mu, base_lv = self.param_head(base_joint).chunk(2, -1)
            gate = self.gate(pooled).sigmoid()
            base_mu = prior_input + base_mu * (1-pm+pm*gate)
            base_law = self.law_head(base_joint)
        # Mask before local temporal convolutions: padded frames cannot affect neighbours.
        d, hh, ww = local.shape[1:]
        masked = projected.reshape(b, v, t, d, hh, ww) * valid[..., None, None, None]
        temporal = masked.permute(0, 1, 4, 5, 3, 2).reshape(-1, d, t)
        motion = self.motion_adapter(temporal).reshape(b, v, hh, ww, d, t)
        motion = motion.permute(0, 1, 5, 4, 2, 3).reshape(b*v*t, d, hh, ww)
        local = local + self.spatial_adapter(local) + motion
        low = self.contact_flow(local)
        contact = low[:, :1].sigmoid()
        flow = low[:, 1:].sigmoid() if self.cfg.field_mode == 'legacy_visual' else low[:, 1:]
        stress = self.stress(torch.cat((local, contact, flow), 1))
        stress = stress.sigmoid() if self.cfg.field_mode == 'legacy_visual' else F.softplus(stress)
        fields = torch.cat((contact, flow, stress), 1)
        control = self.feature_control(local)
        if self.cfg.feedback.startswith('oracle'):
            if diagnostic_fields is None:
                raise ValueError('Oracle is a privileged diagnostic only')
            bridge = F.interpolate(diagnostic_fields.flatten(0, 2).float(), fields.shape[-2:],
                                   mode='bilinear', align_corners=False) + fields*0. + control*0.
        elif diagnostic_fields is not None:
            raise ValueError('GT fields rejected by deployment models')
        else:
            bridge = control + fields*0. if self.cfg.feedback == 'features' else fields + control*0.
        if self.cfg.detach_feedback:
            bridge = bridge.detach()
        embedding = self._bridge_embedding(bridge, encoded, position, valid, weights)
        if self.cfg.bridge_pooling == 'normalized_anchor':
            reference = self._bridge_embedding(torch.zeros_like(bridge), encoded, position, valid, weights)
            embedding = embedding - reference
        # G1d isolates the explicit direct-RGB shortcut in the residual head.
        residual_rgb = pooled if self.cfg.residual_rgb_input else torch.zeros_like(pooled)
        joint = torch.cat((residual_rgb, embedding), -1)
        residual = self.residual_head(joint)
        delta_law = self.residual_law(joint)
        if self.cfg.bridge_pooling == 'normalized_anchor':
            # Cancel learned intercepts too: a zero bridge must reproduce RGB,
            # including after optimizer updates. Do not detach either reference.
            residual = residual - self.residual_head(torch.zeros_like(joint))
            delta_law = delta_law - self.residual_law(torch.zeros_like(joint))
        delta, delta_lv = residual.chunk(2, -1)
        delta = delta + control.mean()*0.  # Keep the matched control connected under DDP/detach.
        if self.cfg.feedback == 'none':
            delta, delta_lv, delta_law = delta*0., delta_lv*0., delta_law*0.
        size = (self.cfg.field_size, self.cfg.field_size)
        return {'mean': base_mu + delta, 'logvar': (base_lv+delta_lv).clamp(-6, 4),
                'law_logits': base_law + delta_law,
                'fields': F.interpolate(fields, size, mode='bilinear', align_corners=False).reshape(b, v, t, -1, *size),
                'contact_logits': F.interpolate(low[:, :1], size, mode='bilinear', align_corners=False).reshape(b, v, t, 1, *size)}

    def _bridge_embedding(self, bridge, encoded, position, valid, weights):
        b, v, t = valid.shape
        d = self.cfg.width
        # Preserve a 4x4 spatial grid; RGB context queries its local field tokens.
        spatial = F.adaptive_avg_pool2d(self.field_encoder[:-1](bridge), (4, 4))
        coords = torch.linspace(-1, 1, 4, device=spatial.device)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        coordinate = torch.stack((xx, yy), -1).reshape(16, 2)
        freq = torch.exp(torch.arange(d//4, device=spatial.device).float() * (-math.log(10000.) / max(d//4, 1)))
        xy = coordinate[..., None] * freq
        xy = torch.cat((xy.sin(), xy.cos()), -1).flatten(1).to(spatial.dtype)
        spatial_tokens = spatial.flatten(2).transpose(1, 2) + xy
        query = encoded.reshape(b*v*t, 1, d)
        attended = self.local_attention(query, spatial_tokens, spatial_tokens, need_weights=False)[0]
        sequences = attended.reshape(b*v, t, d) + position.reshape(b*v, t, d)
        mask = valid.reshape(b*v, t)
        # Fully padded views need one harmless key to avoid all-masked attention NaNs.
        safe = mask.clone(); safe[~safe.any(1), 0] = True
        sequences = self.bridge_temporal(sequences, src_key_padding_mask=~safe).reshape(b, v*t, d)
        logits = self.confidence(sequences)
        if self.cfg.bridge_pooling == 'normalized_anchor':
            # log(sigmoid) + masked softmax is a stable normalized sigmoid gate.
            # A shared tiny gate cannot suppress the entire bridge anymore.
            log_weight = F.logsigmoid(logits.float()).masked_fill(~valid.reshape(b, v*t, 1), -torch.inf)
            confidence = log_weight.softmax(1)
            return (sequences.float() * confidence).sum(1).to(sequences.dtype)
        confidence = logits.sigmoid() * weights
        return (sequences * confidence).sum(1) / weights.sum(1).clamp_min(1)


def build_model(cfg):
    if cfg.architecture == 'motion_temporal':
        from .temporal_bridge import TemporalBridgeModel
        return TemporalBridgeModel(cfg)
    if cfg.architecture == 'motion_field':
        from .motion_model import MotionFieldModel
        return MotionFieldModel(cfg)
    return LocalResidualModel(cfg) if cfg.architecture == 'local_residual' else MaterialFieldModel(cfg)


def initialize_rgb_baseline(model, checkpoint, backbone_sha, manifest_sha, stats):
    old = ModelConfig(**checkpoint['config']['model'])
    comparable = replace(model.cfg, architecture='legacy', feedback='none', detach_feedback=False,
                         residual_rgb_input=True, bridge_pooling='legacy')
    if old.architecture != 'legacy' or old.feedback != 'none' or old.use_prior:
        raise ValueError('Initialization requires a legacy RGB-only estimator without priors')
    if asdict(old) != asdict(comparable):
        raise ValueError('RGB baseline model contract mismatch')
    if checkpoint['manifest_sha256'] != manifest_sha or checkpoint['stats'] != stats:
        raise ValueError('RGB baseline data/statistics mismatch')
    # Strictly load the complete historical model, then copy exactly its named tensors.
    legacy = MaterialFieldModel(old)
    load_weights(legacy, checkpoint, backbone_sha)
    own = model.state_dict()
    for key, tensor in legacy.state_dict().items():
        own[key].copy_(tensor)
