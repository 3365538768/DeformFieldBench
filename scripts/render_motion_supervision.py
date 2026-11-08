"""Train-only RGB/teacher examples at low, median and high correspondence coverage."""
import hashlib
import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from fieldbridge.config import load_config
from fieldbridge.data import read_manifest
from engine_fixture import atomic, digest, wait_if_paused

ROOT=Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT);torch.set_num_threads(1);cv2.setNumThreads(1)
    output=ROOT/'runs/m2_motion_preview';output.mkdir(parents=True,exist_ok=True)
    cfg=load_config('configs/runtime/m2v2_B_rgb_seed17.json')
    rows={r['sample_id']:r for r in read_manifest(cfg.data.manifest)}
    report=json.loads((ROOT/'docs/M2_OBSERVED_MOTION.json').read_text())
    training=sorted([r for r in report['rows'] if r['split']=='train'],key=lambda r:r['mean_valid_object_fraction'])
    selected=[training[round((len(training)-1)*q)] for q in [.1,.5,.9]]
    result=dict(status='running',completed=0,total=3,clips=[],detail='训练集对应覆盖 P10/P50/P90：实际 RGB 时间、像素位移伪标签与可靠区域。')
    for entry in selected:
        wait_if_paused(output);row=rows[entry['sample_id']]
        with np.load(row['pack'],allow_pickle=False) as z:rgb=z['rgb'][0,entry['frame_indices']].transpose(0,2,3,1)
        sample=torch.load(ROOT/'data/prepared/m2_observed_motion'/entry['filename'],weights_only=True)
        displacement=sample['fields'][0,:,1:3].numpy()*8.;valid=sample['field_mask'][0,:,1].numpy().astype(bool)
        index=int(np.argmax((np.linalg.norm(displacement,axis=1)*valid).sum((1,2))))
        times=np.asarray(row['timestamps_seconds'])[entry['frame_indices']]
        spec=dict(sample_id=entry['sample_id'],target_sha256=entry['sha256'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        identifier='clip_'+digest(spec)[:20];folder=ROOT/'data/rendered/m2_motion'/identifier;folder.mkdir(parents=True,exist_ok=True)
        images=[Image.fromarray(x) for x in rgb]
        images[0].save(folder/'preview.gif',save_all=True,append_images=images[1:],duration=[max(20,round(float(x)*1000/10)*10) for x in np.diff(times)]+[400],loop=0)
        images[-1].save(folder/'final.png')
        flow=displacement[index].transpose(1,2,0);angle=np.arctan2(flow[...,1],flow[...,0]);magnitude=np.linalg.norm(flow,axis=-1)
        hsv=np.zeros((*magnitude.shape,3),np.uint8);hsv[...,0]=np.mod((angle+np.pi)/(2*np.pi)*180,180).astype(np.uint8)
        hsv[...,1]=255;hsv[...,2]=(np.clip(magnitude/32,0,1)*255).astype(np.uint8)
        color=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB);color[~valid[index]]=[45,45,45]
        coverage=cv2.resize(valid[index].astype(np.uint8),(224,224),interpolation=cv2.INTER_NEAREST)>0
        overlay=rgb[index].copy();overlay[~coverage]=(overlay[~coverage]*.35+np.array([120,0,0])).clip(0,255).astype(np.uint8)
        sheet=Image.new('RGB',(896,248),(15,22,32));draw=ImageDraw.Draw(sheet)
        panels=[images[0],images[index],Image.fromarray(cv2.resize(color,(224,224),interpolation=cv2.INTER_NEAREST)),Image.fromarray(overlay)]
        for j,(image,label) in enumerate(zip(panels,['First RGB',f'Current RGB t={times[index]:.2f}s','Current -> first / 0-32px','Red = no motion label'])):
            sheet.paste(image,(224*j,24));draw.text((224*j+4,6),label,fill='white')
        sheet.save(folder/'contact_sheet.png')
        result['clips'].append(dict(id=identifier,path=str(folder.relative_to(ROOT)),motion_supervision=True,
            object_id=row['object_id'],sample_id=row['sample_id'],split='train',metrics=dict(mean_valid_object_fraction=entry['mean_valid_object_fraction'],mean_displacement_px=entry['mean_reference_displacement_px']),
            specification=spec,frame_times_s=times.tolist(),deformation_amplification=1.,eligible_for_video_pilot=False,
            scope='RGB DIS with true object masks; red has no displacement supervision. Not simulator motion GT or force/stress.'))
        result['completed']=len(result['clips']);atomic(ROOT/'docs/M2_MOTION_PREVIEW.json',result)
    result['status']='complete';atomic(ROOT/'docs/M2_MOTION_PREVIEW.json',result)
    atomic(output/'status.json',dict(status='complete',reason='three_train_only_supervision_previews_ready'))


if __name__=='__main__':main()
