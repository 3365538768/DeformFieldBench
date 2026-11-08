"""Extract only existing RGB, preserving frame indices and original resolution.

No new renderer, material labels, external dataset, or target inputs are used.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from fieldbridge.config import load_config
from fieldbridge.data import read_manifest,_image_tensor,_resize
from fieldbridge.runtime import sha256
from engine_fixture import atomic,wait_if_paused

ROOT=Path(__file__).resolve().parents[1]
CACHE='data/prepared/b1_pixel_rgb'


def main():
    os.chdir(ROOT);torch.set_num_threads(1);cfg=load_config('configs/single_video_base.json')
    out=ROOT/'runs/b1_pixel_prepare';out.mkdir(parents=True,exist_ok=True);directory=ROOT/CACHE;directory.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',completed=0,total=640,max_quantization_error=0.,
        detail='只解压现有RGB，保留16帧、三台源相机、原时间与像素，不读取旧RGB场数组；训练再选一台。',
        source='existing DeformFieldBench RGB only',new_material_labels=False,test_untouched=True)
    def update(**kwargs):
        report.update(kwargs,updated_unix=time.time());atomic(ROOT/'docs/B1_PIXEL_PREPARE.json',report)
    try:
        base=ROOT/cfg.data.prepared_dir;old=json.loads((base/'index.json').read_text())
        index=dict(schema='uint8_rgb_v1',manifest_sha256=sha256(cfg.data.manifest),frames=cfg.data.frames,max_views=cfg.data.max_views,
            image_size=cfg.data.image_size,files={},sha256={},source_feature_index_sha256=sha256(base/'index.json'))
        if index['manifest_sha256']!=old['manifest_sha256']:raise ValueError('RGB/feature manifest mismatch')
        shapes=set();rows=[r for r in read_manifest(cfg.data.manifest) if r['split'] in ['train','val']]
        if len(rows)!=640:raise ValueError('B1 requires original fixed 512/128 protocol')
        for row in rows:
            wait_if_paused(out);sid=row['sample_id']
            with np.load(row['pack'],allow_pickle=False) as archive:
                raw=archive['rgb'];shapes.add(tuple(raw.shape));v,t=min(raw.shape[0],cfg.data.max_views),raw.shape[1]
                positions=torch.linspace(0,t-1,min(t,cfg.data.frames)).round().long()
                rgb=_resize(_image_tensor(raw[:v,positions.numpy()]),cfg.data.image_size)
            original=torch.load(base/old['files'][sid],map_location='cpu',weights_only=True)
            if original['sample_id']!=sid or original['valid'].shape!=rgb.shape[:2]:raise ValueError('RGB/feature identity/shape mismatch')
            expected=torch.tensor(row['timestamps_seconds'],dtype=torch.float32)[positions].repeat(v,1)
            if not torch.equal(expected,original['times']):raise ValueError('RGB sampling time mismatch')
            pixels=(rgb*255).round().clamp(0,255).to(torch.uint8)
            delta=float((rgb-pixels.float()/255).abs().max());report['max_quantization_error']=max(report['max_quantization_error'],delta)
            if delta>.5/255+1e-6:raise ValueError('RGB cache quantization error exceeds half a pixel intensity step')
            name=hashlib.sha256(sid.encode()).hexdigest()+'.pt';tmp=directory/(name+'.tmp')
            torch.save(dict(sample_id=sid,rgb=pixels,times=original['times'],valid=original['valid']),tmp);tmp.replace(directory/name)
            index['files'][sid]=name;index['sha256'][sid]=sha256(directory/name)
            update(completed=report['completed']+1)
        atomic(directory/'index.json',index)
        update(status='complete',rgb_cache=CACHE,index_sha256=sha256(directory/'index.json'),source_rgb_shapes=[list(s) for s in sorted(shapes)],
            limitation='Original packed RGB only; upsampling cannot recover native missing detail. Frozen DINO remains unchanged in B1 pixel probe.')
        atomic(out/'status.json',dict(status='complete',reason='rgb_cache_sampling_identity_hash_verified'))
    except Exception as error:
        update(status='failed',error=str(error));atomic(out/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
