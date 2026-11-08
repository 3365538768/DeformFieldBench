import pytest
import torch

from fieldbridge.config import ModelConfig
from fieldbridge.local_model import build_model
from fieldbridge.motion_model import temporal_differences


def test_feature_differences_do_not_create_motion_from_padding_or_static_frames():
    x = torch.ones(1,2,4,8,3,3); valid = torch.tensor([[[True,True,False,False],[False]*4]])
    x[:,:,2:] = 1000
    clean, departure, adjacent = temporal_differences(x,valid)
    assert not departure.any() and not adjacent.any()
    assert not clean[~valid].any()


def test_motion_model_preserves_view_symmetry_and_rejects_gt():
    torch.manual_seed(17)
    cfg = ModelConfig(architecture='motion_field',width=32,heads=4,layers=1,field_size=8)
    model = build_model(cfg).eval()
    inputs = dict(rgb=torch.zeros(1,2,4,3,1,1),features=torch.randn(1,2,4,64,4,4),
                  valid=torch.tensor([[[True,True,False,False],[True]*4]]),
                  times=torch.arange(4).float()[None,None].expand(1,2,4),prior=torch.zeros(1,4),prior_mask=torch.zeros(1,4))
    with torch.no_grad():
        a=model(**inputs)
        changed={**inputs,'features':inputs['features'].clone()};changed['features'][~inputs['valid']]=10000
        b=model(**changed)
        swapped={k:(x.flip(1) if k in ['rgb','features','valid','times'] else x) for k,x in inputs.items()}
        c=model(**swapped)
    torch.testing.assert_close(a['mean'],b['mean'])
    torch.testing.assert_close(a['mean'],c['mean'],atol=1e-6,rtol=1e-5)
    assert a['fields'].shape==(1,2,4,7,8,8)
    with pytest.raises(ValueError,match='GT fields rejected'):
        model(**inputs,diagnostic_fields=torch.ones(1,2,4,7,8,8))
    counts=[]
    for feedback in ['none','features','fields']:
        cfg.feedback=feedback;counts.append(sum(p.numel() for p in build_model(cfg).parameters()))
    assert len(set(counts))==1
