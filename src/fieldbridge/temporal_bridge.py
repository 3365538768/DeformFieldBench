"""Order-aware spatial field tokens with an exact zero-field embedding anchor."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .motion_model import MotionFieldModel


class TemporalBridgeModel(MotionFieldModel):
    def __init__(self,cfg):
        super().__init__(cfg)
        layer=nn.TransformerEncoderLayer(cfg.width,cfg.heads,2*cfg.width,dropout=0.,
            batch_first=True,norm_first=True,activation='gelu')
        self.bridge_temporal=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)

    def _bridge_embedding(self,bridge,valid,times):
        b,v,t=valid.shape;d=self.cfg.width
        frequency=torch.exp(torch.arange(0,d,2,device=times.device).float()*(-math.log(10000.)/d))
        phase=times[...,None]*frequency
        position=torch.stack((phase.sin(),phase.cos()),-1).flatten(-2)
        position=position[:,:,None].expand(b,v,16,t,d).reshape(b*v*16,t,d)
        mask=valid[:,:,None].expand(b,v,16,t).reshape(b*v*16,t)
        safe=mask.clone();safe[~safe.any(1),0]=True
        def encode(x):
            # Erase invalid frames before local convolutions or temporal mixing.
            x=x*valid.flatten()[:,None,None,None]
            spatial=F.adaptive_avg_pool2d(self.field_encoder[:-1](x),(4,4))
            tokens=spatial.reshape(b,v,t,d,16).permute(0,1,4,2,3).reshape(b*v*16,t,d)
            encoded=self.bridge_temporal(tokens+position.to(tokens.dtype),src_key_padding_mask=~safe)
            encoded=encoded*mask[...,None]
            return encoded.reshape(b,v,16,t,d).sum((1,2,3))/(valid.sum((1,2)).clamp_min(1)[:,None]*16)
        return encode(bridge)-encode(torch.zeros_like(bridge))
