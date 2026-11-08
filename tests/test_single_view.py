import torch
from fieldbridge.config import Config
from fieldbridge.data import MaterialDataset
from test_observed_motion import overlay


def test_single_view_training_uses_one_camera_and_masks_all_other_fields(overlay):
    cfg,_,_=overlay
    ds=MaterialDataset(cfg,'train',8,'observed_motion')
    index=next(i for i in range(len(ds)) if ds[i]['valid'].any(1).sum()>1)
    cfg.view_policy='single';cfg.single_view_layout='compact';torch.manual_seed(17)
    chosen=[]
    for _ in range(24):
        sample=ds[index];active=sample['valid'].any(1)
        assert active.sum()==1
        assert not sample['field_mask'][~active].any()
        assert sample['features'].shape[0]==sample['fields'].shape[0]==1
        assert sample['features'].untyped_storage().nbytes()==sample['features'].numel()*sample['features'].element_size()
        chosen.append(sample['camera_index'])
    assert len(set(chosen))>1
    cfg.view_policy='all'
    assert ds[index]['valid'].any(1).sum()>1


def test_single_view_validation_is_deterministic(overlay):
    cfg,_,_=overlay;cfg.view_policy='single';ds=MaterialDataset(cfg,'train',8,'observed_motion')
    ds.split='val'
    for seed in [17,29]:
        torch.manual_seed(seed);sample=ds[0]
        assert sample['valid'][0].all() and not sample['valid'][1:].any()
        assert sample['camera_index']==0


def test_single_view_rejects_incompatible_augmentation():
    cfg=Config();cfg.data.view_policy='single';cfg.data.prepared_dir='cache';cfg.data.view_dropout=.5
    import pytest
    with pytest.raises(ValueError,match='Single-view'):cfg.validate()


def test_compact_view_matches_masked_camera_prediction():
    from fieldbridge.local_model import build_model
    from fieldbridge.config import ModelConfig
    torch.manual_seed(17)
    model=build_model(ModelConfig(architecture='motion_temporal',field_mode='observed_motion',width=32,heads=4,layers=1,field_size=8)).eval()
    inputs=dict(rgb=torch.zeros(1,3,4,3,1,1),features=torch.randn(1,3,4,64,4,4),
        valid=torch.tensor([[[False]*4,[True]*4,[False]*4]]),times=torch.arange(4).float()[None,None].expand(1,3,4),
        prior=torch.zeros(1,4),prior_mask=torch.zeros(1,4))
    compact={k:(v[:,1:2].clone() if k in ['rgb','features','valid','times'] else v) for k,v in inputs.items()}
    with torch.no_grad():a=model(**inputs);b=model(**compact)
    torch.testing.assert_close(a['mean'],b['mean'],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(a['fields'][:,1:2],b['fields'],atol=1e-6,rtol=1e-5)
