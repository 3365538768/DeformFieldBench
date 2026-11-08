import copy
import hashlib
import json
import torch
import pytest
from fieldbridge.config import ModelConfig,Config
from fieldbridge.data import MaterialDataset
from fieldbridge.local_model import build_model
from test_observed_motion import overlay


def test_rgb_overlay_checks_hash_and_keeps_selected_camera_alignment(overlay,tmp_path):
    cfg,_,_=overlay;base=MaterialDataset(cfg,'train',8,'observed_motion');folder=tmp_path/'rgb';folder.mkdir()
    files={};hashes={}
    for i in range(len(base)):
        sample=base[i];name=str(i)+'.pt';sid=sample['sample_id']
        torch.save(dict(sample_id=sid,rgb=(sample['rgb']*255).round().to(torch.uint8),valid=sample['valid'],times=sample['times']),folder/name)
        files[sid]=name;hashes[sid]=hashlib.sha256((folder/name).read_bytes()).hexdigest()
    index=dict(schema='uint8_rgb_v1',manifest_sha256=hashlib.sha256(open(cfg.manifest,'rb').read()).hexdigest(),frames=4,max_views=3,image_size=32,files=files,sha256=hashes)
    (folder/'index.json').write_text(json.dumps(index));cfg.rgb_cache_dir=str(folder);cfg.view_policy='single';cfg.single_view_layout='compact'
    ds=MaterialDataset(cfg,'train',8,'observed_motion');sample=ds[0]
    assert sample['rgb'].shape==(1,base[0]['valid'].shape[1],3,32,32) and sample['features'].shape[0]==1
    sid=sample['sample_id'];(folder/files[sid]).write_bytes(b'bad')
    with pytest.raises(ValueError,match='RGB cache SHA'):ds[0]


def test_pixel_branch_has_matched_capacity_and_rejects_placeholder_rgb():
    torch.manual_seed(17)
    cfg=ModelConfig(architecture='motion_temporal',field_mode='observed_motion',pixel_branch=True,width=32,heads=4,layers=1,field_size=8)
    enabled=build_model(cfg).eval();cfg=copy.deepcopy(cfg);cfg.pixel_enabled=False
    disabled=build_model(cfg).eval();disabled.load_state_dict(enabled.state_dict())
    assert sum(p.numel() for p in enabled.parameters())==sum(p.numel() for p in disabled.parameters())
    inputs=dict(rgb=torch.rand(1,1,3,3,32,32),features=torch.randn(1,1,3,64,4,4),valid=torch.ones(1,1,3,dtype=torch.bool),
        times=torch.arange(3).float().reshape(1,1,3),prior=torch.zeros(1,4),prior_mask=torch.zeros(1,4))
    a=enabled(**inputs);b=disabled(**inputs)
    assert (a['mean']-b['mean']).abs().max()>1e-7
    a['mean'].square().sum().backward();assert any(p.grad is not None and p.grad.any() for p in enabled.pixel_encoder.parameters())
    changed=dict(inputs,rgb=torch.rand_like(inputs['rgb']))
    torch.testing.assert_close(disabled(**changed)['mean'],b['mean'])
    with pytest.raises(ValueError,match='real RGB'):enabled(**dict(inputs,rgb=torch.zeros(1,1,3,3,1,1)))
    broken=Config(model=cfg)
    with pytest.raises(ValueError,match='Pixel branch'):broken.validate()
