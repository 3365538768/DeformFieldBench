"""Training-only local gradient diagnostic at the fixed M2 D checkpoint."""
import copy
import json
import os
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader, Subset
from fieldbridge.config import load_config
from fieldbridge.data import MaterialDataset, collate
from fieldbridge.evaluation import move_batch
from fieldbridge.local_model import build_model
from fieldbridge.losses import compute_loss
from fieldbridge.runtime import forward_batch, load_weights, sha256
from engine_fixture import atomic, wait_if_paused

ROOT=Path(__file__).resolve().parents[1]


def summarize_gradients(a,b,weight):
    vectors=[(x.flatten().float(),y.flatten().float()) for x,y in zip(a,b) if x is not None and y is not None]
    if not vectors:raise ValueError('No common gradient support')
    dot=sum((x*y).sum() for x,y in vectors)
    na=sum(x.square().sum() for x,y in vectors).sqrt();nb=sum(y.square().sum() for x,y in vectors).sqrt()
    return dict(cosine=float(dot/(na*nb).clamp_min(1e-20)),material_norm=float(na),
        field_norm=float(nb),weighted_field_to_material_norm=float(weight*nb/na.clamp_min(1e-20)))


def main():
    os.chdir(ROOT);torch.set_num_threads(1)
    output=ROOT/'runs/m2_gradient_audit';output.mkdir(parents=True,exist_ok=True)
    cfg=load_config('configs/runtime/m2v2_D_fields_seed17.json')
    ds=MaterialDataset(cfg.data,'train',cfg.model.field_size,cfg.model.field_mode)
    checkpoint=ROOT/'runs/m2v2_D_fields_seed17/best_material.pt'
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ckpt['manifest_sha256']!=sha256(cfg.data.manifest) or ckpt['stats']!=ds.codec.stats or ckpt.get('motion_target_index_sha256')!=sha256(Path(cfg.data.motion_targets_dir)/'index.json'):
        raise ValueError('Gradient diagnostic checkpoint/data mismatch')
    model=build_model(cfg.model).cuda().eval();load_weights(model,ckpt,sha256(cfg.model.weights))
    groups={'motion_projection':list(model.project.parameters())+list(model.motion_mix.parameters()),
            'displacement_reliability_decoder':list(model.contact_flow.parameters())}
    material_cfg=copy.deepcopy(cfg);material_cfg.train.field_weight=0.
    report=dict(status='running',completed=0,total=8,batches=[],checkpoint_sha256=sha256(checkpoint),
        checkpoint_epoch=ckpt['epoch']+1,split='train',sample_ids=[],test_untouched=True,
        scope='First 32 deterministic training samples at one fixed checkpoint, eval mode, FP32; local gradient association, not proof of optimization causality.',
        detail='固定训练样本上比较材参损失与场损失的梯度夹角、范数；不更新权重。')
    try:
        for batch in DataLoader(Subset(ds,range(32)),batch_size=4,num_workers=4,collate_fn=collate):
            wait_if_paused(output);b=move_batch(batch,'cuda');report['sample_ids'].extend(b['sample_id'])
            prediction=forward_batch(model,b,cfg.model.feedback)
            total,_=compute_loss(prediction,b,cfg);material,_=compute_loss(prediction,b,material_cfg)
            field=(total-material)/cfg.train.field_weight
            result={}
            for name,parameters in groups.items():
                a=torch.autograd.grad(material,parameters,retain_graph=True,allow_unused=True)
                z=torch.autograd.grad(field,parameters,retain_graph=True,allow_unused=True)
                result[name]=summarize_gradients(a,z,cfg.train.field_weight)
            report['batches'].append(result);report.update(completed=len(report['batches']),updated_unix=time.time())
            atomic(ROOT/'docs/M2_GRADIENT_AUDIT.json',report)
        report['summary']={name:dict(negative_cosine_batches=sum(r[name]['cosine']<0 for r in report['batches']),
            mean_cosine=sum(r[name]['cosine'] for r in report['batches'])/report['total'],
            mean_weighted_field_to_material_norm=sum(r[name]['weighted_field_to_material_norm'] for r in report['batches'])/report['total']) for name in groups}
        report.update(status='complete',updated_unix=time.time());atomic(ROOT/'docs/M2_GRADIENT_AUDIT.json',report)
        atomic(output/'status.json',dict(status='complete',reason='training_only_fixed_checkpoint_gradient_audit'))
    except Exception as error:
        report.update(status='failed',error=str(error));atomic(ROOT/'docs/M2_GRADIENT_AUDIT.json',report)
        atomic(output/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
