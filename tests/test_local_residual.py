import copy
from dataclasses import replace
import torch
import pytest
import importlib.util
import json
from pathlib import Path
from fieldbridge.config import Config, ModelConfig
from fieldbridge.model import MaterialFieldModel
from fieldbridge.local_model import LocalResidualModel, initialize_rgb_baseline
from fieldbridge.losses import compute_loss


def setup():
    torch.set_num_threads(1); torch.manual_seed(17)
    cfg = ModelConfig(width=16, heads=4, layers=1, field_size=8, feedback='none')
    legacy = MaterialFieldModel(cfg).eval()
    checkpoint = {'config': {'model': cfg.__dict__}, 'model': legacy.state_dict(),
                  'manifest_sha256': 'manifest', 'stats': {'train': True}}
    model = LocalResidualModel(replace(cfg, architecture='local_residual', feedback='fields'))
    initialize_rgb_baseline(model, checkpoint, None, 'manifest', checkpoint['stats'])
    inputs = {'rgb':torch.rand(1,2,3,3,32,32), 'valid':torch.ones(1,2,3,dtype=torch.bool),
              'times':torch.tensor([[[0.,.1,.2],[0.,.1,.2]]]),'prior':torch.zeros(1,4),
              'prior_mask':torch.zeros(1,4)}
    return legacy, model, inputs, checkpoint


def test_zero_residual_reproduces_frozen_rgb_and_can_learn():
    legacy, model, inputs, _ = setup()
    model.eval()
    with torch.no_grad():
        expected = legacy(**inputs)['mean']
        actual = model(**inputs)['mean']
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    frozen = {k:v.clone() for k,v in model.state_dict().items() if k.startswith(('project.','param_head.','temporal_views.'))}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    # First update opens the initially zero residual; subsequent gradients must reach fields.
    for _ in range(2):
        optimizer.zero_grad();model(**inputs)['mean'].sum().backward();optimizer.step()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.contact_flow.parameters())
    assert not torch.equal(model(**inputs)['mean'], expected)
    for k,v in frozen.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)


@pytest.mark.parametrize('pooling',['legacy','normalized_anchor'])
def test_local_bridge_padding_views_and_gt_isolation(pooling):
    _,model,inputs,_ = setup();model.eval()
    model.cfg=replace(model.cfg,bridge_pooling=pooling,residual_rgb_input=pooling=='legacy')
    with torch.no_grad():model.residual_head[-1].weight.normal_(0,.05)
    baseline=model(**inputs)['mean'].detach()
    reversed_views={k:x.flip(1) if k in {'rgb','valid','times'} else x for k,x in inputs.items()}
    torch.testing.assert_close(model(**reversed_views)['mean'],baseline,rtol=1e-4,atol=1e-5)
    padded=copy.deepcopy(inputs)
    for key in ['rgb','valid','times']:
        shape=list(padded[key].shape);shape[1]=1
        extra=torch.randn(shape) if key!='valid' else torch.zeros(shape,dtype=torch.bool)
        padded[key]=torch.cat((padded[key],extra),1)
    torch.testing.assert_close(model(**padded)['mean'],baseline,rtol=1e-4,atol=1e-5)
    for key in ['rgb','valid','times']:
        shape=list(inputs[key].shape);shape[2]=2
        extra=torch.randn(shape) if key!='valid' else torch.zeros(shape,dtype=torch.bool)
        padded[key]=torch.cat((inputs[key],extra),2)
    torch.testing.assert_close(model(**padded)['mean'],baseline,rtol=1e-4,atol=1e-5)
    with pytest.raises(ValueError,match='GT fields rejected'):
        model(**inputs,diagnostic_fields=torch.rand(1,2,3,7,8,8))


def test_controls_have_equal_capacity_and_baseline_init_rejects_bad_provenance():
    _,model,_,checkpoint=setup()
    counts=[]
    for feedback in ['none','features','fields','oracle','oracle_shuffled']:
        cfg=replace(model.cfg,feedback=feedback)
        counts.append(sum(p.numel() for p in LocalResidualModel(cfg).parameters()))
    assert len(set(counts))==1
    bad=copy.deepcopy(checkpoint);bad['manifest_sha256']='other'
    with pytest.raises(ValueError,match='data/statistics'):
        initialize_rgb_baseline(model,bad,None,'manifest',checkpoint['stats'])


def test_local_none_cannot_change_parameter_outputs_when_fields_train():
    legacy,model,inputs,_=setup();model.cfg=replace(model.cfg,feedback='none')
    cfg=Config(model=model.cfg);cfg.train.contact_loss='balanced_bce'
    batch={'target':torch.ones(1,4),'target_mask':torch.ones(1,4),'law':torch.zeros(1,dtype=torch.long),
           'fields':torch.rand(1,2,3,7,8,8),'field_mask':torch.ones(1,2,3,7,8,8)}
    before=model(**inputs)['mean'].detach()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    loss,_=compute_loss(model(**inputs),batch,cfg);loss.backward();optimizer.step()
    torch.testing.assert_close(model(**inputs)['mean'],before,rtol=0,atol=0)


def test_epoch_zero_baseline_checkpoint_can_resume(tmp_path, monkeypatch):
    from fieldbridge.data import fit_stats
    import fieldbridge.train as trainer
    spec=importlib.util.spec_from_file_location('local_fixtures',Path(__file__).parents[1]/'scripts/make_fixture.py')
    fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
    monkeypatch.chdir(tmp_path);torch.set_num_threads(1)
    cfg=Config(model=ModelConfig(width=16,heads=4,layers=1,field_size=8,feedback='none'))
    cfg.data.manifest=str(fixtures.create(tmp_path/'data'));cfg.data.stats=str(tmp_path/'stats.json')
    fit_stats(cfg.data.manifest,cfg.data.stats);cfg.data.frames=4;cfg.data.image_size=32
    cfg.train.device='cpu';cfg.train.purpose='smoke';cfg.train.epochs=1;cfg.train.early_stop_patience=0
    cfg.train.output=str(tmp_path/'base')
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg.dump()));trainer.run(config)
    cfg.model.architecture='local_residual';cfg.train.output=str(tmp_path/'local')
    cfg.train.initialize_rgb_baseline=str(tmp_path/'base/best_material.pt');cfg.train.epochs=2;cfg.train.lr=0.
    config.write_text(json.dumps(cfg.dump()))
    monkeypatch.setattr(trainer,'paused',lambda output:(Path(output)/'last.pt').exists())
    trainer.run(config)
    initial=torch.load(tmp_path/'local/last.pt',weights_only=False)
    assert initial['epoch']==-1 and initial['epoch_complete']
    cfg.train.initialize_rgb_baseline='';cfg.train.resume=str(tmp_path/'local/last.pt')
    config.write_text(json.dumps(cfg.dump()));monkeypatch.setattr(trainer,'paused',lambda _:False)
    trainer.run(config)
    best=torch.load(tmp_path/'local/best_material.pt',weights_only=False)
    assert best['epoch']==-1
    last=torch.load(tmp_path/'local/last.pt',weights_only=False)
    assert last['step']==12
    assert last['best_material']==initial['best_material']


def test_direct_rgb_shortcut_can_be_removed_without_changing_baseline():
    legacy,model,inputs,_=setup()
    model.cfg=replace(model.cfg,residual_rgb_input=False);model.eval()
    captured=[]
    handle=model.residual_head[0].register_forward_pre_hook(lambda _,values:captured.append(values[0].detach()))
    with torch.no_grad():
        prediction=model(**inputs)['mean'];expected=legacy(**inputs)['mean']
    handle.remove()
    assert torch.count_nonzero(captured[0][:,:model.cfg.width])==0
    assert torch.count_nonzero(captured[0][:,model.cfg.width:])>0
    torch.testing.assert_close(prediction,expected,rtol=0,atol=0)


def test_anchor_zero_bridge_stays_rgb_after_learning_and_keeps_field_gradients():
    legacy,model,inputs,_=setup()
    model.cfg=replace(model.cfg,residual_rgb_input=False,bridge_pooling='normalized_anchor')
    model.train()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    for _ in range(3):
        optimizer.zero_grad();model(**inputs)['mean'].sum().backward();optimizer.step()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.contact_flow.parameters())
    model.eval()
    # Nonzero learned biases cannot change the zero-field reference prediction.
    with torch.no_grad():model.residual_head[-1].bias.fill_(.7);model.residual_law.bias.fill_(.8)
    handle=model.field_encoder[0].register_forward_pre_hook(lambda _,values:(torch.zeros_like(values[0]),))
    with torch.no_grad():
        actual=model(**inputs);expected=legacy(**inputs)
    handle.remove()
    for name in ['mean','logvar','law_logits']:
        torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0)


def test_normalized_gate_cannot_suppress_embedding_with_shared_negative_bias():
    _,model,inputs,_=setup();model.eval()
    model.cfg=replace(model.cfg,residual_rgb_input=False,bridge_pooling='normalized_anchor')
    with torch.no_grad():
        model.residual_head[-1].weight.normal_(0,.1)
        model.confidence.weight.zero_();model.confidence.bias.fill_(-2)
        expected=model(**inputs)['mean']
        model.confidence.bias.fill_(-100)
        actual=model(**inputs)['mean']
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('positive',[False,True])
def test_contact_dice_ignores_padding_and_backpropagates_finite_gradients(positive):
    from fieldbridge.losses import contact_bce_dice
    logits=torch.zeros(1,1,2,1,4,4,requires_grad=True)
    target=torch.zeros_like(logits);mask=torch.ones_like(logits);mask[:,:,1]=0
    if positive:target[0,0,0,0,1,1]=1
    loss=contact_bce_dice(logits,target,mask);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[:,:,:1].abs().sum()>0 and logits.grad[:,:,1:].abs().sum()==0
    changed=target.clone();changed[:,:,1]=1
    torch.testing.assert_close(contact_bce_dice(logits,changed,mask),loss)
