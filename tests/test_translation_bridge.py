import copy
import torch
from fieldbridge.config import ModelConfig
from fieldbridge.local_model import build_model
from fieldbridge.temporal_bridge import translation_banks


def test_translation_banks_keep_rigid_motion_separate_from_relative_evidence():
    field=torch.zeros(2,4,8,8);field[:,0]=1;field[:,1]=3;field[:,2]=-2
    local,global_motion=translation_banks(field)
    assert not local[:,1:].any()
    torch.testing.assert_close(global_motion,field)
    field[0,1,0,0]+=1
    local,global_motion=translation_banks(field)
    assert local[0,1].abs().max()>0
    torch.testing.assert_close(local[:,1:3]+global_motion[:,1:3],field[:,1:3])
    for control in [False,True]:
        a,b=translation_banks(torch.zeros_like(field),control)
        assert not a.any() and not b.any()
    field[:,0]=0
    assert not translation_banks(field)[0][:,1:].any()
    assert not translation_banks(field)[1][:,1:].any()


def test_decomposition_preserves_capacity_zero_anchor_and_masking():
    torch.manual_seed(17)
    cfg=ModelConfig(architecture='motion_temporal',field_mode='observed_motion',width=32,heads=4,layers=1,field_size=8)
    raw=build_model(cfg).eval();cfg=copy.deepcopy(cfg);cfg.bridge_decomposition='translation'
    model=build_model(cfg).eval();model.load_state_dict(raw.state_dict())
    assert sum(p.numel() for p in raw.parameters())==sum(p.numel() for p in model.parameters())
    valid=torch.tensor([[[True,True,True],[False,False,False]]]);times=torch.arange(3).float()[None,None].expand(1,2,3)
    fields=torch.randn(6,4,8,8);fields[:,0]=fields[:,0].sigmoid();fields[:,3]=0
    a=model._bridge_embedding(fields,valid,times);fields[3:]=1000
    torch.testing.assert_close(a,model._bridge_embedding(fields,valid,times))
    assert not model._bridge_embedding(torch.zeros_like(fields),valid,times).any()


def test_no_feedback_controls_have_identical_outputs_and_material_gradients():
    torch.manual_seed(17)
    cfg=ModelConfig(architecture='motion_temporal',field_mode='observed_motion',feedback='none',width=32,heads=4,layers=1,field_size=8)
    a=build_model(cfg).train();cfg=copy.deepcopy(cfg);cfg.bridge_decomposition='translation'
    b=build_model(cfg).train();b.load_state_dict(a.state_dict())
    inputs=dict(rgb=torch.zeros(1,1,3,3,1,1),features=torch.randn(1,1,3,64,4,4),valid=torch.ones(1,1,3,dtype=torch.bool),
        times=torch.arange(3).float().reshape(1,1,3),prior=torch.zeros(1,4),prior_mask=torch.zeros(1,4))
    torch.manual_seed(29);x=a(**inputs);torch.manual_seed(29);y=b(**inputs)
    for key in ['mean','fields','law_logits']:torch.testing.assert_close(x[key],y[key],atol=0,rtol=0)
    for output in [x,y]:(output['mean'].square().sum()+output['fields'].square().mean()).backward()
    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        assert (p.grad is None)==(q.grad is None),name
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
