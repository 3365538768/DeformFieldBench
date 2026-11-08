"""Bounded train/validation audit of motion content, sampling and model errors.

Translation residual is an image-space diagnostic, not measured material strain.
The same reliable teacher pixels define total and residual displacement energy.
"""
import json
import os
from pathlib import Path
import time
from collections import Counter
import numpy as np
import torch
from fieldbridge.data import read_manifest
from fieldbridge.runtime import sha256
from engine_fixture import atomic, wait_if_paused

ROOT=Path(__file__).resolve().parents[1]
LABELS=['B_rgb','C_aux','D_fields','G_features','O_pseudo_oracle']


def translation_energy(vector):
    center=vector.mean(0)
    energy=vector.square().sum()
    residual=(vector-center).square().sum()
    return float(energy),float(residual),float(vector.norm(dim=1).mean()),float((vector-center).norm(dim=1).mean())


def quantiles(values):
    return dict(zip(['p10','p50','p90'],map(float,np.quantile(values,[.1,.5,.9])))) if values else None


def main():
    os.chdir(ROOT);torch.set_num_threads(1)
    assert translation_energy(torch.ones(4,2))[1]==0
    assert translation_energy(torch.tensor([[1.,0.],[-1.,0.]]))[:2]==(2.,2.)
    out=ROOT/'runs/m2_bottleneck_audit';out.mkdir(parents=True,exist_ok=True)
    index_path=ROOT/'data/prepared/m2_observed_motion/index.json'
    index=json.loads(index_path.read_text());rows=read_manifest('data/manifests/g1_diagnostic_512_128.jsonl')
    if any(r['split']=='test' for r in rows):raise ValueError('Test data excluded from tuning audits')
    report=dict(status='running',completed=0,total=len(rows),manifest_sha256=sha256('data/manifests/g1_diagnostic_512_128.jsonl'),
        target_index_sha256=sha256(index_path),splits={},models={},test_untouched=True,
        detail='分离整体平移与空间变化，检查动作覆盖、逐参数/动作误差和训练耗时。',
        scope='Teacher-visible pixels only; image translation residual also includes rotation, perspective and tracking error, not strain GT.')
    summaries={s:dict(samples=0,objects=set(),actions=Counter(),laws=Counter(),pixels=0,
        energy=0.,residual_energy=0.,per_clip_residual_fraction=[],motion_px=[],translation_residual_px=[],
        reliable_cells_per_frame=[],frames_with_at_least_32_cells=0,noninitial_frames=0) for s in ['train','val']}
    try:
        for row in rows:
            wait_if_paused(out);file=index_path.parent/index['files'][row['sample_id']]
            if sha256(file)!=index['sha256'][row['sample_id']]:raise ValueError('Motion label hash mismatch')
            pack=torch.load(file,map_location='cpu',weights_only=True)
            vector=pack['fields'][:,:,1:3].float()*index['displacement_scale_px']
            mask=pack['field_mask'][:,:,1:3].all(2)
            summary=summaries[row['split']];summary['samples']+=1;summary['objects'].add(row['object_id'])
            summary['actions'][row.get('action','missing')]+=1;summary['laws'][str(row.get('law_id'))]+=1
            energy=residual=0.
            for view in range(vector.shape[0]):
                for t in range(1,vector.shape[1]):
                    summary['noninitial_frames']+=1
                    n=int(mask[view,t].sum());summary['reliable_cells_per_frame'].append(n)
                    if n<32:continue
                    summary['frames_with_at_least_32_cells']+=1;summary['pixels']+=n
                    values=vector[view,t].permute(1,2,0)[mask[view,t]]
                    e,r,m,d=translation_energy(values);energy+=e;residual+=r
                    summary['motion_px'].append(m);summary['translation_residual_px'].append(d)
            summary['energy']+=energy;summary['residual_energy']+=residual
            if energy>0:summary['per_clip_residual_fraction'].append(residual/energy)
            report.update(completed=report['completed']+1,updated_unix=time.time())
            if report['completed']%32==0:atomic(ROOT/'docs/M2_BOTTLENECK_AUDIT.json',report)
        for split,s in summaries.items():
            report['splits'][split]=dict(samples=s['samples'],objects=len(s['objects']),actions=dict(s['actions']),laws=dict(s['laws']),
                reliable_cells_per_noninitial_frame=quantiles(s['reliable_cells_per_frame']),
                sufficient_support_fraction=s['frames_with_at_least_32_cells']/s['noninitial_frames'],
                total_displacement_mean_per_frame_px=quantiles(s['motion_px']),
                translation_residual_mean_per_frame_px=quantiles(s['translation_residual_px']),
                residual_to_total_squared_energy=s['residual_energy']/max(s['energy'],1e-12),
                per_clip_residual_energy_fraction=quantiles(s['per_clip_residual_fraction']),
                scope='Frames with >=32 reliable 56x56 cells; first frame excluded. Values remain in original 224px coordinates.')
        by_id={r['sample_id']:r for r in rows}
        for label in LABELS:
            folder=ROOT/('runs/m2v2_'+label+'_seed17')
            status=json.loads((folder/'status.json').read_text())
            checkpoint=torch.load(folder/'best_material.pt',map_location='cpu',weights_only=False)
            predictions=[json.loads(x) for x in (folder/f"val_epoch_{checkpoint['epoch']:04d}/predictions.jsonl").read_text().splitlines()]
            groups={}
            for prediction in predictions:
                row=by_id[prediction['sample_id']];mask=np.asarray(prediction['target_mask'],bool)
                error=np.abs(np.asarray(prediction['prediction_z'])-prediction['target_z'])
                for j,name in enumerate(['E','nu','rho','sigma_y']):
                    if mask[j]:groups.setdefault('parameter:'+name,[]).append(float(error[j]))
                for key in ['action','law_id']:
                    groups.setdefault(key+':'+str(row[key]),[]).extend(error[mask].tolist())
            history=[json.loads(x) for x in (folder/'timing.jsonl').read_text().splitlines()]
            timing={k:quantiles([r[k] for r in history if k in r]) for k in ['train_seconds','data_wait_seconds','validation_seconds','cycle_seconds']}
            report['models'][label]=dict(best_epoch=checkpoint['epoch']+1,stop_reason=status.get('reason'),
                errors={k:dict(standardized_mae=float(np.mean(v)),label_count=len(v)) for k,v in groups.items()},timing_seconds=timing)
        report.update(status='complete',updated_unix=time.time());atomic(ROOT/'docs/M2_BOTTLENECK_AUDIT.json',report)
        atomic(out/'status.json',dict(status='complete',reason='bounded_motion_content_and_error_audit'))
    except Exception as error:
        report.update(status='failed',error=str(error));atomic(ROOT/'docs/M2_BOTTLENECK_AUDIT.json',report)
        atomic(out/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
