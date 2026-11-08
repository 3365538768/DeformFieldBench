"""Patch-local temporal evidence before spatial pooling; legacy encoder stays fixed."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import MaterialFieldModel


def temporal_differences(projected, valid):
    """Appearance, departure from first valid frame, and adjacent differences.

    These are feature differences, not metric velocities. Invalid observations
    are erased before any mixing and adjacent differences need two valid frames.
    """
    b, v, t, d, h, w = projected.shape
    clean = projected * valid[..., None, None, None]
    first = valid.long().argmax(2)
    reference = clean.gather(2, first[..., None, None, None, None].expand(b, v, 1, d, h, w))
    departure = (clean-reference) * valid[..., None, None, None]
    adjacent = torch.zeros_like(clean)
    adjacent[:, :, 1:] = (clean[:, :, 1:]-clean[:, :, :-1]) * (
        valid[:, :, 1:] & valid[:, :, :-1])[..., None, None, None]
    return clean, departure, adjacent


class MotionFieldModel(MaterialFieldModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.motion_mix = nn.Sequential(nn.Conv2d(3*cfg.width, cfg.width, 1), nn.GELU(),
                                        nn.Conv2d(cfg.width, cfg.width, 3, padding=1, groups=cfg.width))

    def _bridge_embedding(self, bridge, valid, times):
        b,v,t=valid.shape
        weights=valid.reshape(b,v*t,1).to(bridge.dtype)
        embedding=self.field_encoder(bridge).flatten(1).reshape(b,v*t,self.cfg.width)
        return (embedding*weights).sum(1)/weights.sum(1).clamp_min(1)

    def forward(self, rgb, valid, times, prior, prior_mask, features=None, diagnostic_fields=None):
        b, v, t, c, h, w = rgb.shape
        if not valid.flatten(1).any(1).all() or not torch.isfinite(times).all():
            raise ValueError('Need valid observations and finite times')
        if features is not None:
            if any(p.requires_grad for p in self.frame_encoder.parameters()):
                raise ValueError('Cached features require a frozen backbone')
            if features.shape[:4] != (b, v, t, self.frame_encoder.dim):
                raise ValueError('Cached feature shape mismatch')
            frame = features.flatten(0, 2).to(self.project.weight.dtype)
        else:
            frame = self.frame_encoder(rgb.flatten(0, 2))
        projected = self.project(frame)
        d, hh, ww = projected.shape[1:]
        clean, departure, adjacent = temporal_differences(projected.reshape(b,v,t,d,hh,ww), valid)
        local = clean.flatten(0, 2) + self.motion_mix(torch.cat((clean,departure,adjacent), 3).flatten(0, 2))
        local = local * valid.flatten()[:, None, None, None]

        # Attention follows each of 16 spatial cells through time before pooling.
        grid = F.adaptive_avg_pool2d(local, (4,4)).reshape(b,v,t,d,16)
        sequence = grid.permute(0,1,4,2,3).reshape(b*v*16,t,d)
        frequency = torch.exp(torch.arange(0,d,2,device=times.device).float()*(-math.log(10000.)/d))
        phase = times[...,None] * frequency
        position = torch.stack((phase.sin(),phase.cos()),-1).flatten(-2)
        position = position[:,:,None].expand(b,v,16,t,d).reshape(b*v*16,t,d).to(sequence.dtype)
        mask = valid[:,:,None].expand(b,v,16,t).reshape(b*v*16,t)
        safe = mask.clone(); safe[~safe.any(1),0] = True
        encoded = self.temporal_views(sequence+position,src_key_padding_mask=~safe)
        encoded = encoded * mask[...,None]
        denominator = valid.sum((1,2)).clamp_min(1).to(encoded.dtype)*16
        pooled = encoded.reshape(b,v,16,t,d).sum((1,2,3))/denominator[:,None]
        context = self.context(encoded).reshape(b,v,16,t,d).permute(0,1,3,4,2).reshape(b*v*t,d,4,4)
        local = local + F.interpolate(context,(hh,ww),mode='bilinear',align_corners=False)
        low = self.contact_flow(local)
        contact = low[:,:1].sigmoid()
        flow = low[:,1:].sigmoid() if self.cfg.field_mode=='legacy_visual' else low[:,1:]
        stress = self.stress(torch.cat((local,contact,flow),1))
        stress = stress.sigmoid() if self.cfg.field_mode=='legacy_visual' else F.softplus(stress)
        if self.cfg.field_mode=='observed_motion':
            stress=stress*0  # No free latent channel masquerading as missing stress.
        fields = torch.cat((contact,flow,stress),1)
        control = self.feature_control(local)
        if self.cfg.field_mode=='observed_motion':
            control=torch.cat((control[:,:3],control[:,3:]*0),1)
        if self.cfg.feedback.startswith('oracle'):
            if diagnostic_fields is None:
                raise ValueError('Oracle requires explicit privileged fields')
            bridge = F.interpolate(diagnostic_fields.flatten(0,2).float(),(hh,ww),mode='bilinear',align_corners=False)+fields*0+control*0
        elif diagnostic_fields is not None:
            raise ValueError('GT fields rejected by deployment models')
        else:
            bridge = control+fields*0 if self.cfg.feedback=='features' else fields+control*0
        if self.cfg.detach_feedback:
            bridge = bridge.detach()
        field_embedding = self._bridge_embedding(bridge,valid,times)
        if self.cfg.feedback=='none':
            field_embedding = field_embedding*0
        pm = prior_mask.to(prior.dtype)*float(self.cfg.use_prior)
        prior_input = prior*pm
        prior_embedding = self.prior_encoder(torch.cat((prior_input,pm),-1))
        joint = torch.cat((pooled,field_embedding,prior_embedding),-1)
        mean,logvar = self.param_head(joint).chunk(2,-1)
        mean = prior_input+mean*(1-pm+pm*self.gate(pooled).sigmoid())
        size = (self.cfg.field_size,self.cfg.field_size)
        return dict(mean=mean,logvar=logvar.clamp(-6,4),law_logits=self.law_head(joint),
                    material_embedding=torch.cat((pooled,field_embedding),-1),
                    fields=F.interpolate(fields,size,mode='bilinear',align_corners=False).reshape(b,v,t,-1,*size),
                    contact_logits=F.interpolate(low[:,:1],size,mode='bilinear',align_corners=False).reshape(b,v,t,1,*size))
