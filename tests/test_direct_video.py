import importlib.util
import json
from pathlib import Path
import pytest
import torch
from fieldbridge.config import Config,ModelConfig
from fieldbridge.data import MaterialDataset,fit_stats,collate,model_inputs,read_manifest
from fieldbridge.direct_data import FieldFreeDataset
from fieldbridge.direct_video import mixture_log_prob
from fieldbridge.local_model import build_model
from fieldbridge.losses import compute_loss
from fieldbridge.runtime import sha256


@pytest.fixture
def direct_config(tmp_path):
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('direct_fixture',Path(__file__).parents[1]/'scripts/make_fixture.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    manifest=fixture.create(tmp_path/'raw');stats=tmp_path/'stats.json';fit_stats(manifest,stats)
    cfg=Config(model=ModelConfig(architecture='direct_video',field_mode='none',feedback='none',width=16,heads=4,layers=1,field_size=8))
    cfg.data.manifest=str(manifest);cfg.data.stats=str(stats);cfg.data.frames=4;cfg.data.image_size=32
    base=tmp_path/'features';base.mkdir();files={}
    for split in ['train','val']:
        ds=MaterialDataset(cfg.data,split,8,'legacy_visual')
        for row in ds.rows:
            i=next(i for i,x in enumerate(ds.rows) if x['sample_id']==row['sample_id']);p=ds[i]
            p['features']=torch.randn(*p['valid'].shape,64,4,4)
            p['fields'].fill_(float('nan'))  # A poisoned field must never be consumed.
            name=row['sample_id']+'.pt';torch.save(p,base/name);files[row['sample_id']]=name
    (base/'index.json').write_text(json.dumps(dict(manifest_sha256=sha256(manifest),frames=4,max_views=3,image_size=32,
        files=files,hub_name=cfg.model.hub_name,backbone_checkpoint_sha256=None,
        encoder_source_sha256=sha256(Path(__file__).parents[1]/'src/fieldbridge/model.py'))))
    cfg.data.prepared_dir=str(base);cfg.data.view_policy='single';cfg.data.single_view_layout='compact'
    cfg.train.field_weight=0.;cfg.train.field_epochs=0;cfg.train.device='cpu';cfg.train.epochs=1
    cfg.train.workers=cfg.train.val_workers=0;cfg.train.purpose='smoke';cfg.train.output=str(tmp_path/'run')
    return cfg


def write_priors(cfg,path,leak=False):
    rows=[r for r in read_manifest(cfg.data.manifest) if r['split'] in ['train','val']]
    training=[r['object_id'] for r in rows if r['split']=='train'];fits={};samples={}
    for row in rows:
        sid=row['sample_id'];fits[sid]=dict(training_objects=training if leak or row['split']=='val' else [o for o in training if o!=row['object_id']])
        samples[sid]=dict(object_id=row['object_id'],fit_id=sid,mean_z=[[0.]*4]*3,logvar_z=[[0.]*4]*3,mask=[[True]*4]*3)
    path.write_text(json.dumps(dict(schema='fieldfree_static_prior_v1',manifest_sha256=sha256(cfg.data.manifest),
        stats_sha256=sha256(cfg.data.stats),feature_index_sha256=sha256(Path(cfg.data.prepared_dir)/'index.json'),fits=fits,samples=samples)))
    cfg.data.material_prior_file=str(path)


def test_no_fields_in_data_model_or_training(direct_config):
    cfg=direct_config;cfg.validate();ds=FieldFreeDataset(cfg.data,'train');batch=collate([ds[0],ds[1]])
    assert not {'fields','field_mask','diagnostic_fields'} & set(batch)
    model=build_model(cfg.model)
    assert not any(any(x in n for x in ['field_encoder','contact_flow','stress','feature_control']) for n,_ in model.named_parameters())
    output=model(**model_inputs(batch));assert 'fields' not in output
    loss,parts=compute_loss(output,batch,cfg);loss.backward()
    assert torch.isfinite(loss) and 'contact' not in parts
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_direct_prior_is_masked_and_can_change_prediction(direct_config):
    cfg=direct_config;sample=FieldFreeDataset(cfg.data,'val')[0];batch=collate([sample]);cfg.model.use_prior=True
    model=build_model(cfg.model).eval();inputs=model_inputs(batch)
    with torch.no_grad():
        expected=model(**inputs)['mean']
        erased=model(**dict(inputs,prior=torch.full_like(inputs['prior'],float('nan')),prior_logvar=torch.full_like(inputs['prior'],float('nan'))))['mean']
        changed=model(**dict(inputs,prior=torch.ones_like(inputs['prior']),prior_mask=torch.ones_like(inputs['prior_mask'])))['mean']
    torch.testing.assert_close(expected,erased,rtol=0,atol=0)
    assert not torch.allclose(expected,changed)


def test_static_prior_rejects_own_object_and_wrong_feature_index(direct_config,tmp_path):
    cfg=direct_config;path=tmp_path/'prior.json';write_priors(cfg,path,True)
    with pytest.raises(ValueError,match='leaks'):FieldFreeDataset(cfg.data,'train')
    write_priors(cfg,path);ds=FieldFreeDataset(cfg.data,'train');assert ds[0]['prior_mask'].all()
    value=json.loads(path.read_text());value['feature_index_sha256']='wrong';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='provenance'):FieldFreeDataset(cfg.data,'train')


def test_mixture_missing_parameter_likelihood_and_quantiles():
    from fieldbridge.posterior import mixture_quantile
    logits=torch.tensor([[.2,-.3]],requires_grad=True);means=torch.tensor([[[-3.,2.],[3.,-1.]]],requires_grad=True)
    lv=torch.zeros_like(means);target=torch.tensor([[1.,900.]])
    mask=torch.tensor([[True,False]])
    actual=mixture_log_prob(logits,means,lv,target,mask)
    expected=torch.logsumexp(logits.log_softmax(-1)+torch.distributions.Normal(means[:,:,0],1.).log_prob(target[:,:1]),-1)
    torch.testing.assert_close(actual,expected)
    (-actual.sum()).backward();assert torch.isfinite(means.grad).all() and not means.grad[:,:,1].any()
    q=mixture_quantile(torch.zeros(1,1),torch.zeros(1,1,1),torch.zeros(1,1,1),.975)
    assert abs(float(q)-1.9599639845)<1e-6


def test_teacher_rejects_unadmitted_or_nontraining_solutions(direct_config,tmp_path):
    cfg=direct_config;path=tmp_path/'teacher.json'
    value=dict(schema='fieldfree_offline_teacher_v1',manifest_sha256=sha256(cfg.data.manifest),stats_sha256=sha256(cfg.data.stats),physical_admission_passed=False,samples={})
    path.write_text(json.dumps(value));cfg.data.material_teacher_file=str(path)
    with pytest.raises(ValueError,match='admission'):FieldFreeDataset(cfg.data,'train')
    value.update(physical_admission_passed=True,samples={'fixture_6':dict(object_id='fixture_object_6')});path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='non-training'):FieldFreeDataset(cfg.data,'train')


def test_direct_training_resume_and_prior_hash_guard(direct_config,tmp_path,monkeypatch):
    import fieldbridge.train as trainer
    cfg=direct_config;path=tmp_path/'prior.json';write_priors(cfg,path);cfg.train.nll_weight=.1
    monkeypatch.chdir(tmp_path);config=tmp_path/'config.json';config.write_text(json.dumps(cfg.dump()));trainer.run(config)
    last=torch.load(Path(cfg.train.output)/'last.pt',weights_only=False)
    assert last['best_field'] is None and not (Path(cfg.train.output)/'best_field.pt').exists()
    assert last['material_overlay_sha256']['material_prior_file']==sha256(path)
    summary=json.loads((Path(cfg.train.output)/'val_epoch_0000/summary.json').read_text())
    assert summary['field_absolute_error_encoded'] is None
    cfg.train.epochs=2;cfg.train.resume=str(Path(cfg.train.output)/'last.pt');config.write_text(json.dumps(cfg.dump()));trainer.run(config)
    assert torch.load(cfg.train.resume,weights_only=False)['epoch']==1
    value=json.loads(path.read_text());value['annotation']='changed';path.write_text(json.dumps(value))
    config.write_text(json.dumps(cfg.dump()))
    with pytest.raises(ValueError,match='provenance'):trainer.run(config)


def test_configuration_rejects_hidden_field_supervision(direct_config):
    cfg=direct_config;cfg.train.field_weight=.1
    with pytest.raises(ValueError,match='no fields'):cfg.validate()
    cfg.train.field_weight=0;cfg.model.feedback='features'
    with pytest.raises(ValueError,match='no fields'):cfg.validate()


def test_multisolution_teacher_enters_loss_but_not_model_inputs(direct_config,tmp_path):
    cfg=direct_config;path=tmp_path/'teacher.json';solutions={}
    for row in read_manifest(cfg.data.manifest):
        if row['split']=='train':
            solutions[row['sample_id']]=dict(object_id=row['object_id'],solver='software_test_only',input_sha256='fixture',
                fit_protocol='fixture',response_validation={'fixture':True},solutions_z=[[.2]*4,[-.2]*4],weights=[.4,.6],parameter_mask=[True,True,True,False])
    path.write_text(json.dumps(dict(schema='fieldfree_offline_teacher_v1',manifest_sha256=sha256(cfg.data.manifest),
        stats_sha256=sha256(cfg.data.stats),physical_admission_passed=True,samples=solutions)))
    cfg.data.material_teacher_file=str(path);cfg.train.distillation_weight=.2;cfg.validate()
    batch=collate([FieldFreeDataset(cfg.data,'train')[0]])
    assert batch['teacher_weights'].shape==(1,16) and batch['teacher_weights'].sum()==1
    assert not any(k.startswith('teacher') for k in model_inputs(batch))
    model=build_model(cfg.model);output=model(**model_inputs(batch))
    loss,parts=compute_loss(output,batch,cfg);loss.backward()
    assert torch.isfinite(loss) and parts['distillation']!=0 and model.uncertainty_head.weight.grad.abs().sum()>0


def test_single_video_cli_needs_no_fields_or_labels(direct_config,tmp_path,monkeypatch):
    cv2=pytest.importorskip('cv2')
    import numpy as np
    import fieldbridge.train as trainer
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from infer_fieldfree_video import run,read_video
    cfg=direct_config;monkeypatch.chdir(tmp_path);config=tmp_path/'config.json'
    config.write_text(json.dumps(cfg.dump()));trainer.run(config)
    video=tmp_path/'video.avi';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10.,(32,32))
    if not writer.isOpened():pytest.skip('MJPG video encoder unavailable')
    try:
        for i in range(6):writer.write(np.full((32,32,3),i*20,dtype=np.uint8))
    finally:writer.release()
    output=tmp_path/'inference.json'
    result=run(Path(cfg.train.output)/'best_material.pt',video,output,'cpu')
    assert result['uses_fields'] is False and result['uses_online_optimization'] is False
    assert set(result['parameters'])=={'E','nu','rho','sigma_y'}
    assert result['video']['sampled_frame_indices']==[0,2,3,5] and output.exists()
    assert result['video']['timing_source']=='constant_container_fps'
    timestamps=tmp_path/'times.json';timestamps.write_text(json.dumps([2.,2.1,2.4,2.8,3.,3.5]))
    _,times,metadata=read_video(video,4,32,timestamps)
    torch.testing.assert_close(times,torch.tensor([0.,.4,.8,1.5]))
    assert metadata['timing_source']=='explicit_timestamps_seconds' and metadata['source_time_origin_seconds']==2.
    timestamps.write_text(json.dumps([0.,1.,1.,2.,3.,4.]))
    with pytest.raises(ValueError,match='strictly increasing'):read_video(video,4,32,timestamps)
