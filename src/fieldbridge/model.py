import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class FrameEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.backbone == "tiny":
            self.net = nn.Sequential(nn.Conv2d(3, 32, 7, 4, 3), nn.GELU(),
                                     nn.Conv2d(32, 64, 3, 2, 1), nn.GELU())
            self.dim, self.patch_size = 64, 8
        else:
            repo, weights = Path(cfg.repo), Path(cfg.weights)
            if not repo.is_dir() or not weights.is_file():
                raise FileNotFoundError("Explicit pretrained repo/checkpoint not found; no random fallback")
            if cfg.backbone == "dinov3":
                self.net = torch.hub.load(str(repo), cfg.hub_name, source="local", weights=str(weights))
            elif cfg.backbone == "dinov2":
                self.net = torch.hub.load(str(repo), cfg.hub_name, source="local", pretrained=False)
                state = torch.load(weights, map_location="cpu", weights_only=True)
                self.net.load_state_dict(state, strict=True)
            else:
                raise ValueError("Unsupported backbone")
            self.dim = self.net.embed_dim
            patch = self.net.patch_embed.patch_size
            self.patch_size = patch[0] if isinstance(patch, tuple) else patch
        if cfg.freeze_backbone:
            self.net.requires_grad_(False)
        if cfg.train_last_blocks:
            if cfg.backbone == "tiny":
                raise ValueError("Last-block tuning applies to ViT backbones")
            self.net.requires_grad_(False)
            for block in list(self.net.blocks)[-cfg.train_last_blocks:]:
                block.requires_grad_(True)
            self.net.norm.requires_grad_(True)
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.net.parameters()):
            self.net.eval()
        return self

    def forward(self, x):
        x = (x - self.rgb_mean) / self.rgb_std
        parts = []
        for chunk in x.split(self.cfg.frame_chunk):
            if self.cfg.backbone == "tiny":
                y = self.net(chunk)
            else:
                if chunk.shape[-1] % self.patch_size or chunk.shape[-2] % self.patch_size:
                    raise ValueError("Image resolution must be divisible by backbone patch size")
                tokens = self.net.forward_features(chunk)["x_norm_patchtokens"]
                h, w = chunk.shape[-2] // self.patch_size, chunk.shape[-1] // self.patch_size
                y = tokens.transpose(1, 2).reshape(len(chunk), self.dim, h, w)
            parts.append(y)
        return torch.cat(parts)


class MaterialFieldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.frame_encoder = FrameEncoder(cfg)
        d = cfg.width
        self.project = nn.Conv2d(self.frame_encoder.dim, d, 1)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, d * 4, dropout=0.,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.temporal_views = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        # No fixed view embedding: masked attention is invariant to view permutation.
        self.context = nn.Linear(d, d)
        flow_c, stress_c = (3, 3) if cfg.field_mode == "legacy_visual" else (2, 1)
        self.flow_c, self.stress_c = flow_c, stress_c
        self.contact_flow = nn.Sequential(nn.Conv2d(d, d // 2, 3, padding=1), nn.GELU(),
                                          nn.Conv2d(d // 2, 1 + flow_c, 1))
        self.stress = nn.Sequential(nn.Conv2d(d + 1 + flow_c, d // 2, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(d // 2, stress_c, 1))
        field_channels = 1 + flow_c + stress_c
        self.feature_control = nn.Conv2d(d, field_channels, 1)
        self.field_encoder = nn.Sequential(nn.Conv2d(field_channels, 32, 3, 2, 1), nn.GELU(),
                                            nn.Conv2d(32, d, 3, 2, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1))
        self.prior_encoder = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        self.gate = nn.Linear(d, 4)
        self.param_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 8))
        self.law_head = nn.Linear(3 * d, 3)

    def forward(self, rgb, valid, times, prior, prior_mask, features=None, diagnostic_fields=None):
        b, v, t, c, h, w = rgb.shape
        if not valid.flatten(1).any(1).all():
            raise ValueError("Every sample needs a valid observation")
        if not torch.isfinite(times).all():
            raise ValueError("Invalid observation times")
        if features is not None:
            if any(p.requires_grad for p in self.frame_encoder.parameters()):
                raise ValueError("Cached features cannot be used with a trainable backbone")
            if features.shape[:3] != (b, v, t) or features.shape[3] != self.frame_encoder.dim:
                raise ValueError("Cached feature shape mismatch")
            frame_features = features.flatten(0, 2).to(self.project.weight.dtype)
        else:
            frame_features = self.frame_encoder(rgb.reshape(-1, c, h, w))
        local = self.project(frame_features)
        tokens = local.mean((-1, -2)).reshape(b, v * t, -1)
        freq = torch.exp(torch.arange(0, self.cfg.width, 2, device=rgb.device).float()
                         * (-math.log(10000.) / self.cfg.width))
        phase = times.reshape(b, v * t, 1) * freq
        position = torch.stack((phase.sin(), phase.cos()), -1).flatten(-2).to(tokens.dtype)
        flat_mask = valid.reshape(b, v * t)
        encoded = self.temporal_views(tokens + position, src_key_padding_mask=~flat_mask)
        weights = flat_mask[..., None].to(encoded.dtype)
        pooled = (encoded * weights).sum(1) / weights.sum(1).clamp_min(1)
        ctx = self.context(encoded).reshape(b * v * t, -1, 1, 1)
        local = local + ctx
        low = self.contact_flow(local)
        contact = low[:, :1].sigmoid()
        flow = low[:, 1:].sigmoid() if self.cfg.field_mode == "legacy_visual" else low[:, 1:]
        stress = self.stress(torch.cat((local, contact, flow), 1))
        stress = stress.sigmoid() if self.cfg.field_mode == "legacy_visual" else F.softplus(stress)
        fields = torch.cat((contact, flow, stress), 1)
        # Same modules and parameter count for all feedback controls.
        control = self.feature_control(local)
        if self.cfg.feedback.startswith("oracle"):
            if diagnostic_fields is None:
                raise ValueError("Oracle is a privileged diagnostic, not deployable inference")
            oracle = F.interpolate(diagnostic_fields.flatten(0, 2).float(), fields.shape[-2:], mode="bilinear", align_corners=False)
            bridge = oracle + fields * 0. + control * 0.
        elif diagnostic_fields is not None:
            raise ValueError("GT fields rejected by deployment models")
        elif self.cfg.feedback == "features":
            bridge = control + fields * 0.
        else:
            bridge = fields + control * 0.
        if self.cfg.detach_feedback:
            bridge = bridge.detach()
        field_embedding = self.field_encoder(bridge).flatten(1).reshape(b, v * t, -1)
        field_embedding = (field_embedding * weights).sum(1) / weights.sum(1).clamp_min(1)
        if self.cfg.feedback == "none":
            field_embedding = field_embedding * 0.
        pm = prior_mask.to(prior.dtype) * float(self.cfg.use_prior)
        # Invalid prior values are erased before entering the network.
        prior_input = prior * pm
        prior_embedding = self.prior_encoder(torch.cat((prior_input, pm), -1))
        joint = torch.cat((pooled, field_embedding, prior_embedding), -1)
        mu, logvar = self.param_head(joint).chunk(2, -1)
        gate = self.gate(pooled).sigmoid()
        # With no prior, gate is identity. With a prior, learn a bounded residual gate.
        mu = prior_input + mu * (1 - pm + pm * gate)
        field_out = F.interpolate(fields, (self.cfg.field_size, self.cfg.field_size), mode="bilinear", align_corners=False)
        contact_logits = F.interpolate(low[:, :1], (self.cfg.field_size, self.cfg.field_size), mode="bilinear", align_corners=False)
        return {"mean": mu, "logvar": logvar.clamp(-6, 4), "law_logits": self.law_head(joint),
                "fields": field_out.reshape(b, v, t, -1, self.cfg.field_size, self.cfg.field_size),
                "contact_logits": contact_logits.reshape(b, v, t, 1, self.cfg.field_size, self.cfg.field_size)}
