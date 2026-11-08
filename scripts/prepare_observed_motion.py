"""Bounded RGB-derived displacement targets beside immutable DINO features."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from fieldbridge.config import load_config
from fieldbridge.data import read_manifest
from fieldbridge.runtime import sha256
from engine_fixture import atomic, wait_if_paused
from visible_motion import estimate

ROOT=Path(__file__).resolve().parents[1]


def prepare(row,frames,max_views,size,directory):
    filename=hashlib.sha256(row['sample_id'].encode()).hexdigest()+'.pt'
    target=directory/filename
    with np.load(row['pack'],allow_pickle=False) as z:
        raw=z['rgb'];views=min(max_views,len(raw));total=raw.shape[1]
        indices=torch.linspace(0,total-1,min(frames,total)).round().long().numpy()
        rgb=raw[:views,indices].transpose(0,1,3,4,2)
        masks=z['object_mask'][:views,indices].max(2)>127
    fields=np.zeros((views,len(indices),4,size,size),np.float32)
    support=np.zeros_like(fields);support[:,:,0]=1
    magnitudes=[];fractions=[]
    for v in range(views):
        for t in range(len(indices)):
            if t==0:flow=np.zeros((*masks[v,t].shape,2),np.float32);valid=masks[v,t]
            else:flow,valid,_=estimate(rgb[v,t],rgb[v,0],masks[v,t],masks[v,0])
            coverage=cv2.resize(valid.astype(np.float32),(size,size),interpolation=cv2.INTER_AREA)
            weighted=cv2.resize(flow*valid[...,None],(size,size),interpolation=cv2.INTER_AREA)
            displacement=weighted/np.maximum(coverage[...,None],1e-8)
            usable=coverage>=.5
            fields[v,t,0]=usable;fields[v,t,1:3]=displacement.transpose(2,0,1)/8.
            support[v,t,1:3]=usable
            if t and valid.any():magnitudes.append(float(np.linalg.norm(flow[valid],axis=-1).mean()))
            if t:fractions.append(float(valid.sum()/max(masks[v,t].sum(),1)))
    sample=dict(sample_id=row['sample_id'],fields=torch.from_numpy(fields),field_mask=torch.from_numpy(support).bool())
    if not np.isfinite(fields).all() or not fractions:raise ValueError('Invalid motion pseudo-label')
    temp=target.with_suffix('.pt.tmp');torch.save(sample,temp);temp.replace(target)
    return dict(sample_id=row['sample_id'],filename=filename,sha256=sha256(target),split=row['split'],
        object_id=row['object_id'],mean_reference_displacement_px=float(np.mean(magnitudes)),
        mean_valid_object_fraction=float(np.mean(fractions)),frame_indices=indices.tolist())


def main():
    os.chdir(ROOT);torch.set_num_threads(1);cv2.setNumThreads(1)
    output=ROOT/'runs/m2_observed_motion';output.mkdir(parents=True,exist_ok=True)
    gate=json.loads((ROOT/'docs/M2_TEACHER_AUDIT.json').read_text())
    if not gate.get('teacher_accepted'):raise ValueError('Motion teacher did not pass diagnostic gate')
    cfg=load_config('configs/runtime/m1_B_motion_rgb_seed17.json')
    rows=read_manifest(cfg.data.manifest)
    if any(r['split']=='test' for r in rows):raise ValueError('No test pseudo-label generation during tuning')
    directory=ROOT/'data/prepared/m2_observed_motion';directory.mkdir(parents=True,exist_ok=True)
    contract=dict(schema='observed_motion_v1',manifest_sha256=sha256(cfg.data.manifest),frames=cfg.data.frames,
        max_views=cfg.data.max_views,image_size=cfg.data.image_size,field_size=cfg.model.field_size,
        field_mode='observed_motion',displacement_scale_px=8.,units=['binary_reliability_proxy','pixel','pixel','unavailable'],
        direction='current_to_first_frame; values live on current-frame pixels',
        provenance='RGB DIS medium + GT object masks + forward/backward endpoint checks; pseudo labels, not simulator fields',
        source_sha256=hashlib.sha256((ROOT/'scripts/visible_motion.py').read_bytes()+(ROOT/'scripts/prepare_observed_motion.py').read_bytes()).hexdigest())
    report=dict(status='running',completed=0,total=len(rows),rows=[],contract=contract,workers=6,
        detail='同一 512/128 划分：当前帧到首帧的像素位移、对应可靠性；不复制冻结特征，不提供应力标签。')
    def update():
        report.update(completed=len(report['rows']),updated_unix=time.time());atomic(ROOT/'docs/M2_OBSERVED_MOTION.json',report)
    try:
        update()
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            pending={};iterator=iter(rows)
            while True:
                wait_if_paused(output)
                while len(pending)<6:
                    row=next(iterator,None)
                    if row is None:break
                    future=pool.submit(prepare,row,cfg.data.frames,cfg.data.max_views,cfg.model.field_size,directory)
                    pending[future]=row['sample_id']
                if not pending:break
                done,_=concurrent.futures.wait(pending,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    report['rows'].append(future.result());del pending[future];update()
        files={r['sample_id']:r['filename'] for r in report['rows']}
        if len(files)!=len(rows):raise ValueError('Motion target sample coverage mismatch')
        atomic(directory/'index.json',dict(**contract,files=files,sha256={r['sample_id']:r['sha256'] for r in report['rows']}))
        report.update(status='complete',index=str((directory/'index.json').relative_to(ROOT)),
            train_samples=sum(r['split']=='train' for r in rows),val_samples=sum(r['split']=='val' for r in rows),
            used_for_training=False,stress_gt_available=False,reaction_force_gt_available=False)
        update();atomic(output/'status.json',dict(status='complete',reason='bounded_rgb_motion_targets_prepared'))
    except Exception as error:
        report.update(status='failed',error=str(error));update()
        atomic(output/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
