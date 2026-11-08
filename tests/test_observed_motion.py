import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
import torch
from fieldbridge.config import DataConfig
from fieldbridge.data import MaterialDataset, fit_stats, model_inputs


@pytest.fixture
def overlay(tmp_path):
    spec=importlib.util.spec_from_file_location('fixture',Path(__file__).parents[1]/'scripts/make_fixture.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    manifest=module.create(tmp_path/'raw');stats=tmp_path/'stats.json';fit_stats(manifest,stats)
    cfg=DataConfig(manifest=str(manifest),stats=str(stats),frames=4,image_size=32)
    ds=MaterialDataset(cfg,'train',8,'legacy_visual')
    base=tmp_path/'base';motion=tmp_path/'motion';base.mkdir();motion.mkdir();files={};hashes={}
    for i,row in enumerate(ds.rows):
        sample=ds[i];sample['features']=torch.ones(*sample['valid'].shape,64,2,2)
        name=str(i)+'.pt';torch.save(sample,base/name)
        f=torch.zeros(*sample['valid'].shape,4,8,8);f[:,:,:1]=1
        mask=torch.ones_like(f,dtype=torch.bool);mask[:,:,3:]=False
        torch.save(dict(sample_id=row['sample_id'],fields=f,field_mask=mask),motion/name)
        files[row['sample_id']]=name;hashes[row['sample_id']]=hashlib.sha256((motion/name).read_bytes()).hexdigest()
    common=dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),frames=4,
        max_views=3,image_size=32,field_size=8)
    (base/'index.json').write_text(json.dumps(dict(**common,field_mode='legacy_visual',files=files)))
    index=dict(**common,schema='observed_motion_v1',field_mode='observed_motion',files=files,sha256=hashes,
        displacement_scale_px=8.,units=['binary_reliability_proxy','pixel','pixel','unavailable'],
        direction='current_to_first_frame; values live on current-frame pixels')
    (motion/'index.json').write_text(json.dumps(index))
    cfg.prepared_dir=str(base);cfg.motion_targets_dir=str(motion)
    return cfg,motion,index


def test_overlay_preserves_features_and_excludes_targets_from_inputs(overlay):
    cfg,motion,index=overlay
    ds=MaterialDataset(cfg,'train',8,'observed_motion');sample=ds[0]
    assert sample['fields'].shape[2]==4 and not sample['field_mask'][:,:,3:].any()
    assert sample['features'].eq(1).all()
    assert 'fields' not in model_inputs(sample) and 'field_mask' not in model_inputs(sample)
    (motion/index['files'][sample['sample_id']]).write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='SHA'):ds[0]


def test_displacement_is_not_relabelled_as_velocity(overlay):
    cfg,motion,index=overlay;index['units'][1]='pixel_per_second'
    (motion/'index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError,match='contract'):MaterialDataset(cfg,'train',8,'observed_motion')


def test_deployment_intervention_selects_fixed_view_and_preserves_padding():
    import sys
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    from audit_observed_models import observation_intervention
    valid=torch.ones(1,3,16,dtype=torch.bool);valid[:,2]=False
    batch={'valid':valid,'features':torch.randn(1,3,16,4,2,2)}
    single=observation_intervention(batch,'single_view_0')
    assert single['valid'][:,0].all() and not single['valid'][:,1:].any()
    sparse=observation_intervention(batch,'sparse_eight_frames')
    assert sparse['valid'].sum()==16 and not sparse['valid'][:,2].any()
    assert sparse['valid'][:,:2,0].all() and sparse['valid'][:,:2,-1].all()
    assert torch.equal(batch['valid'],valid) and batch['valid'].sum()==32
