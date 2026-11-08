import torch
from fieldbridge.config import ModelConfig
from fieldbridge.local_model import build_model


def test_temporal_bridge_zero_anchor_survives_updates_and_retains_order():
    torch.set_num_threads(1);torch.manual_seed(17)
    cfg=ModelConfig(architecture='motion_temporal',field_mode='observed_motion',width=32,heads=4,layers=1)
    model=build_model(cfg)
    valid=torch.tensor([[[True,True,True],[False,False,False]]])
    times=torch.tensor([[[0.,.3,1.],[0.,0.,0.]]])
    x=torch.randn(6,4,8,8)
    a=model._bridge_embedding(x,valid,times)
    altered=x.reshape(1,2,3,4,8,8).flip(2).reshape_as(x)
    b=model._bridge_embedding(altered,valid,times)
    assert (a-b).abs().max()>1e-6
    padded=x.clone();padded[3:]=100000
    torch.testing.assert_close(a,model._bridge_embedding(padded,valid,times))
    opt=torch.optim.AdamW(model.parameters(),lr=.01)
    a.square().mean().backward();opt.step()
    assert not model._bridge_embedding(torch.zeros_like(x),valid,times).any()


def test_temporal_models_match_capacity_and_no_gt_enters_deployment():
    torch.set_num_threads(1);counts=[]
    for feedback in ['none','fields','features','oracle']:
        cfg=ModelConfig(architecture='motion_temporal',field_mode='observed_motion',width=32,heads=4,layers=1,field_size=8,feedback=feedback)
        model=build_model(cfg)
        counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
    assert len(set(counts))==1


def test_observed_motion_cannot_learn_an_unsupervised_stress_channel():
    torch.set_num_threads(1)
    cfg=ModelConfig(architecture='motion_temporal',field_mode='observed_motion',width=32,heads=4,layers=1,field_size=8)
    model=build_model(cfg)
    inputs=dict(rgb=torch.zeros(1,1,3,3,1,1),features=torch.randn(1,1,3,64,4,4),
        valid=torch.ones(1,1,3,dtype=torch.bool),times=torch.arange(3).float().reshape(1,1,3),
        prior=torch.zeros(1,4),prior_mask=torch.zeros(1,4))
    out=model(**inputs)
    assert not out['fields'][:,:,:,3:].any()
    out['mean'].square().sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in model.stress.parameters())
