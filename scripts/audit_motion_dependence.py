"""Matched M1 validation diagnostics, grouped uncertainty and temporal interventions.

Checkpoint selection and diagnosis both use validation: intervals are exploratory,
conditional on these checkpoints, and exclude training-seed uncertainty.
"""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from fieldbridge.config import load_config
from fieldbridge.data import MaterialDataset, collate
from fieldbridge.evaluation import move_batch
from fieldbridge.local_model import build_model
from fieldbridge.runtime import forward_batch, load_weights, sha256
from engine_fixture import atomic, wait_if_paused

ROOT=Path(__file__).resolve().parents[1]
LABELS=['B_motion_rgb','C_motion_aux','D_motion_fields','G_motion_features']
CONDITIONS=['original','repeat_first_frame','reverse_observations','zero_bridge','reverse_time_bridge']


def intervene(batch, condition):
    result=dict(batch)
    if condition not in {'repeat_first_frame','reverse_observations'}:
        return result
    # Valid-length observations only: padding stays outside the intervention.
    for key in ['rgb','features']:
        if key not in batch:continue
        result[key]=batch[key].clone()
        for i in range(len(batch['valid'])):
            for view in range(batch['valid'].shape[1]):
                indices=torch.where(batch['valid'][i,view])[0]
                if not len(indices):continue
                donor=indices[:1].expand_as(indices) if condition=='repeat_first_frame' else indices.flip(0)
                result[key][i,view,indices]=batch[key][i,view,donor]
    # Keep capture positions fixed, deliberately breaking observation chronology.
    return result


def paired_bootstrap(left,right):
    if set(left)!=set(right):raise ValueError('Unmatched sample IDs')
    groups={}
    for identifier,a in left.items():
        b=right[identifier]
        for key in ['object_id','target_z','target_mask']:
            if a[key]!=b[key]:raise ValueError('Unmatched labels or object identity')
        mask=np.asarray(a['target_mask'],bool);target=np.asarray(a['target_z'])
        ea=np.abs(np.asarray(a['prediction_z'])-target)[mask].sum()
        eb=np.abs(np.asarray(b['prediction_z'])-target)[mask].sum()
        row=groups.setdefault(a['object_id'],np.zeros(3));row += [ea,eb,mask.sum()]
    values=np.stack(list(groups.values()))
    rng=np.random.default_rng(17);sampled=values[rng.integers(len(values),size=(10000,len(values)))].sum(1)
    difference=(sampled[:,0]-sampled[:,1])/sampled[:,2]
    return dict(objects=len(values),samples=len(left),bootstrap_replicates=10000,
        delta_mae_left_minus_right=float((values[:,0].sum()-values[:,1].sum())/values[:,2].sum()),
        exploratory_95_cluster_interval=np.quantile(difference,[.025,.975]).tolist(),
        object_rows={name:dict(left_mae=float(v[0]/v[2]),right_mae=float(v[1]/v[2]),label_count=int(v[2]))
                     for name,v in groups.items()},
        scope='Validation-selected checkpoints; object-cluster bootstrap only; no seed uncertainty or held-out test claim.')


@torch.no_grad()
def main():
    os.chdir(ROOT);torch.set_num_threads(1)
    output=ROOT/'runs/m1_motion_dependence';output.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',completed=0,total=4,models={},test_untouched=True,
        detail='固定最佳验证检查点，检查时间顺序/静帧/场桥依赖；按物体配对 bootstrap，非最终测试。')
    selected={}
    try:
        for label in LABELS:
            wait_if_paused(output)
            folder=ROOT/('runs/m1_'+label+'_seed17')
            cfg=load_config(ROOT/('configs/runtime/m1_'+label+'_seed17.json'))
            ckpt=torch.load(folder/'best_material.pt',map_location='cpu',weights_only=False)
            ds=MaterialDataset(cfg.data,'val',cfg.model.field_size,cfg.model.field_mode)
            backbone=sha256(cfg.model.weights)
            if ckpt['manifest_sha256']!=sha256(cfg.data.manifest) or ckpt['stats']!=ds.codec.stats:
                raise ValueError('Checkpoint/manifest/statistics mismatch')
            if (ds.prepared['backbone_checkpoint_sha256']!=backbone or
                ds.prepared['encoder_source_sha256']!=sha256(ROOT/'src/fieldbridge/model.py')):
                raise ValueError('Frozen feature provenance mismatch')
            device=torch.device('cuda');model=build_model(cfg.model).to(device).eval()
            load_weights(model,ckpt,backbone)
            records=[json.loads(x) for x in (folder/f"val_epoch_{ckpt['epoch']:04d}/predictions.jsonl").read_text().splitlines()]
            selected[label]={r['sample_id']:r for r in records}
            metrics={c:dict(error=0.,change=0.,count=0) for c in CONDITIONS};ids=[]
            begin=time.perf_counter()
            for batch in DataLoader(ds,batch_size=4,num_workers=2,collate_fn=collate):
                wait_if_paused(output);batch=move_batch(batch,device);ids.extend(batch['sample_id'])
                shape=tuple(batch['valid'].shape);reference=None
                for condition in CONDITIONS:
                    def hook(module,inputs):
                        x=inputs[0]
                        if condition=='zero_bridge':return (torch.zeros_like(x),)
                        return (x.reshape(*shape,*x.shape[1:]).flip(2).reshape_as(x),)
                    handle=model.field_encoder[0].register_forward_pre_hook(hook) if condition in {'zero_bridge','reverse_time_bridge'} else None
                    try:
                        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg.train.val_bf16):
                            prediction=forward_batch(model,intervene(batch,condition),cfg.model.feedback)['mean'].float()
                    finally:
                        if handle:handle.remove()
                    if reference is None:reference=prediction
                    mask=batch['target_mask'];m=metrics[condition]
                    m['error']+=float(((prediction-batch['target']).abs()*mask).sum())
                    m['change']+=float(((prediction-reference).abs()*mask).sum());m['count']+=int(mask.sum())
            if len(set(ids))!=len(ds) or set(ids)!=set(selected[label]):raise ValueError('Coverage mismatch')
            for value in metrics.values():
                value['standardized_mae']=value.pop('error')/value['count']
                value['prediction_change_z']=value.pop('change')/value['count']
            timing=[json.loads(x) for x in (folder/'timing.jsonl').read_text().splitlines()]
            report['models'][label]=dict(checkpoint_epoch=ckpt['epoch']+1,
                parameter_mae_z={name:dict(count=sum(r['target_mask'][j] for r in records),
                    mae=float(np.mean([abs(r['prediction_z'][j]-r['target_z'][j]) for r in records if r['target_mask'][j]])))
                    for j,name in enumerate(['E','nu','rho','sigma_y']) if any(r['target_mask'][j] for r in records)},
                checkpoint_sha256=sha256(folder/'best_material.pt'),metrics=metrics,
                audit_seconds=time.perf_counter()-begin,
                epoch_timing_median={key:float(np.median([x[key] for x in timing])) for key in
                    ['train_seconds','data_wait_seconds','validation_seconds','cycle_seconds']})
            report.update(completed=len(report['models']),updated_unix=time.time())
            atomic(ROOT/'docs/M1_MOTION_DEPENDENCE.json',report)
            del model;torch.cuda.empty_cache()
        report['paired_comparisons']={f'D_minus_{label[0]}':paired_bootstrap(selected['D_motion_fields'],selected[label])
                                      for label in ['B_motion_rgb','C_motion_aux','G_motion_features']}
        report.update(status='complete',updated_unix=time.time(),
            interpretation='Dependency interventions may be out of distribution; sensitivity does not establish physical causality. Bridge time reversal can be structurally invariant under mean pooling.')
        atomic(ROOT/'docs/M1_MOTION_DEPENDENCE.json',report)
        atomic(output/'status.json',dict(status='complete',reason='matched_validation_dependency_diagnostic_complete'))
    except Exception as error:
        report.update(status='failed',error=str(error));atomic(ROOT/'docs/M1_MOTION_DEPENDENCE.json',report)
        atomic(output/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
