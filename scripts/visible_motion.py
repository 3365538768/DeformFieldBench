"""Image correspondence teacher with foreground/endpoint/FB consistency masks.

Outputs source-to-target displacement in pixels, never 3D velocity or stress.
Forward/backward consistency is a reliability proxy, not an occlusion GT label.
"""
import cv2
import numpy as np


def estimate(source, target, source_mask, target_mask):
    if source.shape != target.shape or source.shape[:2] != source_mask.shape or source_mask.shape != target_mask.shape:
        raise ValueError('Image/mask shape mismatch')
    a=cv2.cvtColor(np.ascontiguousarray(source),cv2.COLOR_RGB2GRAY)
    b=cv2.cvtColor(np.ascontiguousarray(target),cv2.COLOR_RGB2GRAY)
    solver=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    forward=solver.calc(a,b,None)
    backward=solver.calc(b,a,None)
    h,w=a.shape;yy,xx=np.mgrid[:h,:w].astype(np.float32)
    u=xx+forward[...,0];v=yy+forward[...,1]
    inside=(u>=0)&(u<=w-1)&(v>=0)&(v<=h-1)
    back=cv2.remap(backward,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    endpoint=cv2.remap(target_mask.astype(np.uint8),u,v,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
    residual=np.linalg.norm(forward+back,axis=-1)
    valid=(source_mask>0)&endpoint&inside&(residual<=.5+.05*np.linalg.norm(forward,axis=-1))
    if not np.isfinite(forward).all():raise ValueError('Non-finite displacement teacher')
    return forward,valid,residual
