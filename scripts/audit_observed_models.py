"""Complete-coverage validation audit: motion teacher fit, controls and interventions."""
import json
import os
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from fieldbridge.config import load_config
from fieldbridge.data import MaterialDataset, collate
from fieldbridge.evaluation import move_batch
from fieldbridge.local_model import build_model
from fieldbridge.runtime import forward_batch, load_weights, sha256
from audit_motion_dependence import intervene, paired_bootstrap
from engine_fixture import atomic, wait_if_paused

ROOT=Path(__file__).resolve().parents[1]
LABELS=['B_rgb','C_aux','D_fields','G_features','O_pseudo_oracle']


def observation_intervention(batch,condition):
    result=intervene(batch,condition)
    if condition in ['single_view_0','sparse_eight_frames']:
        result['valid']=batch['valid'].clone()
        if condition=='single_view_0':
            if not result['valid'][:,0].any(1).all():raise ValueError('View zero missing')
            result['valid'][:,1:]=False
        else:
            for i in range(len(result['valid'])):
                for v in range(result['valid'].shape[1]):
                    indices=torch.where(result['valid'][i,v])[0]
                    chosen=torch.linspace(0,len(indices)-1,min(8,len(indices)),device=indices.device).round().long() if len(indices) else indices
                    result['valid'][i,v]=False
                    result['valid'][i,v,indices[chosen]]=True
    return result


def records_at_best(folder,epoch):
    records=[json.loads(x) for x in (folder/f'val_epoch_{epoch:04d}/predictions.jsonl').read_text().splitlines()]
    if len({r['sample_id'] for r in records})!=len(records):raise ValueError('Duplicate validation ID')
    return {r['sample_id']:r for r in records}


def constants(dataset):
    # Only training samples, per-channel median approximated by a fixed bounded histogram.
    histogram=torch.zeros(2,2049,dtype=torch.float64);count=torch.zeros(2,dtype=torch.float64)
    for b in DataLoader(dataset,batch_size=4,num_workers=4,collate_fn=collate):
        for j in range(2):
            value=b['fields'][:,:,:,j+1];mask=b['field_mask'][:,:,:,j+1].bool()
            selected=value[mask].float();count[j]+=len(selected)
            bins=((selected.clamp(-32,32)+32)/64*2048).round().long()
            histogram[j]+=torch.bincount(bins,minlength=2049)
    if not count.gt(0).all():raise ValueError('No train displacement labels')
    median=torch.stack([torch.searchsorted(row.cumsum(0),row.sum()/2) for row in histogram]).float()/2048*64-32
    return median


@torch.no_grad()
def main(report_path='docs/M2_MOTION_DEPENDENCE.json',output_path='runs/m2_model_audit'):
    os.chdir(ROOT);torch.set_num_threads(1)
    output=ROOT/output_path;output.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',completed=0,total=5,models={},test_untouched=True,
        detail='最佳检查点全验证覆盖：位移EPE/常量基线、时序与零桥干预、物体配对区间。',
        scope='RGB-derived pseudo-correspondence; no independent real GT, SI force/stress, or final test claim.')
    selected={}
    try:
        cfg=load_config('configs/runtime/m2v2_C_aux_seed17.json')
        train=MaterialDataset(cfg.data,'train',cfg.model.field_size,cfg.model.field_mode)
        median=constants(train);report['train_component_median_px']=(median*8).tolist()
        atomic(ROOT/report_path,report)
        for label in LABELS:
            wait_if_paused(output);folder=ROOT/('runs/m2v2_'+label+'_seed17')
            cfg=load_config('configs/runtime/m2v2_'+label+'_seed17.json')
            ds=MaterialDataset(cfg.data,'val',cfg.model.field_size,cfg.model.field_mode)
            ckpt=torch.load(folder/'best_material.pt',map_location='cpu',weights_only=False)
            backbone=sha256(cfg.model.weights)
            if ckpt['manifest_sha256']!=sha256(cfg.data.manifest) or ckpt['stats']!=ds.codec.stats or ckpt.get('motion_target_index_sha256')!=sha256(Path(cfg.data.motion_targets_dir)/'index.json'):
                raise ValueError('Audit checkpoint/data contract mismatch')
            if ds.prepared['backbone_checkpoint_sha256']!=backbone or ds.prepared['encoder_source_sha256']!=sha256(ROOT/'src/fieldbridge/model.py'):
                raise ValueError('Encoder provenance mismatch')
            selected[label]=records_at_best(folder,ckpt['epoch'])
            model=build_model(cfg.model).cuda().eval();load_weights(model,ckpt,backbone)
            conditions=['original'] if label.startswith('O_') else ['original','repeat_first_frame','reverse_observations','zero_bridge','reverse_time_bridge','single_view_0','sparse_eight_frames']
            metrics={c:dict(error=0.,change=0.,count=0) for c in conditions}
            flow=dict(predicted_epe_sum=0.,zero_epe_sum=0.,constant_epe_sum=0.,pixels=0,correct=0,positive=0,total=0)
            object_rows={};ids=[]
            for batch in DataLoader(ds,batch_size=4,num_workers=4,collate_fn=collate):
                wait_if_paused(output);b=move_batch(batch,'cuda');ids.extend(b['sample_id'])
                shape=tuple(b['valid'].shape);reference=None
                for condition in conditions:
                    def hook(module,inputs):
                        x=inputs[0]
                        if condition=='zero_bridge':return (torch.zeros_like(x),)
                        return (x.reshape(*shape,*x.shape[1:]).flip(2).reshape_as(x),)
                    handle=model.field_encoder[0].register_forward_pre_hook(hook) if condition in ['zero_bridge','reverse_time_bridge'] else None
                    try:
                        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg.train.val_bf16):
                            out=forward_batch(model,observation_intervention(b,condition),cfg.model.feedback)
                        pred=out['mean'].float()
                    finally:
                        if handle:handle.remove()
                    if reference is None:
                        reference=pred
                        # First-frame displacement is identically zero: exclude it from EPE.
                        mask=b['field_mask'][:,:,:,1:3].all(3)&b['valid'][...,None,None]
                        mask[:,:,0]=False
                        target=b['fields'][:,:,:,1:3].float();prediction=out['fields'][:,:,:,1:3].float()
                        epe=(prediction-target).norm(dim=3)*8
                        zero=target.norm(dim=3)*8
                        const=(target-median.to(target.device)[None,None,None,:,None,None]).norm(dim=3)*8
                        for key,value in [('predicted_epe_sum',epe),('zero_epe_sum',zero),('constant_epe_sum',const)]:flow[key]+=float(value[mask].sum())
                        flow['pixels']+=int(mask.sum())
                        reliability=out['fields'][:,:,:,:1]>.5;gold=b['fields'][:,:,:,:1]>.5;support=b['field_mask'][:,:,:,:1].bool()
                        flow['correct']+=int((reliability&gold&support).sum());flow['positive']+=int((reliability&support).sum());flow['total']+=int((gold&support).sum())
                        for i,identifier in enumerate(b['sample_id']):
                            row=object_rows.setdefault(b['object_id'][i],dict(samples=0,valid_pixels=0,epe_sum=0.,zero_sum=0.))
                            row['samples']+=1;row['valid_pixels']+=int(mask[i].sum());row['epe_sum']+=float(epe[i][mask[i]].sum());row['zero_sum']+=float(zero[i][mask[i]].sum())
                    m=metrics[condition];label_mask=b['target_mask']
                    m['error']+=float(((pred-b['target']).abs()*label_mask).sum());m['change']+=float(((pred-reference).abs()*label_mask).sum());m['count']+=int(label_mask.sum())
            if len(set(ids))!=len(ds) or set(ids)!=set(selected[label]):raise ValueError('Audit validation coverage mismatch')
            for m in metrics.values():
                m['standardized_mae']=m.pop('error')/m['count'];m['mean_prediction_change_z']=m.pop('change')/m['count']
            n=max(flow['pixels'],1)
            quality=dict(predicted_mean_epe_px=flow['predicted_epe_sum']/n,zero_mean_epe_px=flow['zero_epe_sum']/n,
                train_constant_mean_epe_px=flow['constant_epe_sum']/n,valid_pixels=flow['pixels'],
                reliability_precision=flow['correct']/max(flow['positive'],1),reliability_recall=flow['correct']/max(flow['total'],1),
                first_frame_excluded=True,scope='Agreement with RGB teacher only; reliability target is FB consistency, not physical contact or calibrated visibility.')
            report['models'][label]=dict(checkpoint_epoch=ckpt['epoch']+1,checkpoint_sha256=sha256(folder/'best_material.pt'),metrics=metrics,flow_quality=quality,per_object_flow=object_rows)
            report.update(completed=len(report['models']),updated_unix=time.time());atomic(ROOT/report_path,report)
            del model;torch.cuda.empty_cache()
        report['paired_comparisons']={f'D_minus_{label[0]}':paired_bootstrap(selected['D_fields'],selected[label]) for label in ['B_rgb','C_aux','G_features']}
        report.update(status='complete',updated_unix=time.time())
        atomic(ROOT/report_path,report)
        atomic(output/'status.json',dict(status='complete',reason='complete_coverage_motion_and_bridge_audit'))
    except Exception as error:
        report.update(status='failed',error=str(error));atomic(ROOT/report_path,report)
        atomic(output/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--report',default='docs/M2_MOTION_DEPENDENCE.json')
    parser.add_argument('--output',default='runs/m2_model_audit');args=parser.parse_args()
    main(args.report,args.output)
