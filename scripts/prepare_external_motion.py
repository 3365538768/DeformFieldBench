"""Pinned real RGB motion pilot: explicit capture times, object split, no parameter GT."""
import hashlib
import json
import os
from pathlib import Path
import re
import time

import cv2
import numpy as np

from acquire_external_pilots import configure_proxy, deform360
from engine_fixture import atomic, wait_if_paused

ROOT=Path(__file__).resolve().parents[1]
OBJECTS=['001-rope','004-rubber-band','008-pink-cloth','046-sponge','105-clay-cloth',
         '043-dog','044-doll','045-cat','047-rectangle-sponge','048-butter-sponge',
         '049-ball','051-cube','052-rubber-duck','056-makeup-sponge','057-kitchen-sponge',
         '090-sloth','093-squeezable-fruit','096-octopus','102-stress-ball','106-baking-mat']
VALIDATION={'044-doll','048-butter-sponge','090-sloth','093-squeezable-fruit','106-baking-mat'}


def main():
    os.chdir(ROOT);output=ROOT/'runs/external_motion_v1';output.mkdir(parents=True,exist_ok=True)
    configure_proxy(ROOT)
    report=dict(status='running',completed=0,total=len(OBJECTS),used_for_parameter_supervision=False,
        used_for_training=False,rows=[],scope='Bounded object-isolated real-motion pretraining candidate, no E/nu/rho labels.')
    def update(**kw):
        report.update(kw,updated_unix=time.time(),detail='20 个真实物体：完整解码、真实时间戳、物体隔离与运动活跃帧采样。')
        atomic(ROOT/'docs/EXTERNAL_MOTION_V1.json',report)
    try:
        update();wait_if_paused(output)
        source=deform360(ROOT/'data/external_motion_v1',OBJECTS,before_file=lambda name:wait_if_paused(output))
        atomic(output/'download_receipt.json',source)
        report['source_revision']=source['revision'];report['license']=source.get('license')
        directory=ROOT/'data/external_motion_v1/prepared';directory.mkdir(parents=True,exist_ok=True)
        for file in source['files']:
            if not file['path'].endswith('.mp4') or file['status']!='downloaded':continue
            wait_if_paused(output);path=ROOT/'data/external_motion_v1/deform360'/file['path']
            capture=cv2.VideoCapture(str(path));frames=[]
            while True:
                ok,frame=capture.read()
                if not ok:break
                frames.append(cv2.cvtColor(cv2.resize(frame,(224,224)),cv2.COLOR_BGR2RGB))
            capture.release()
            stamps=path.with_suffix('.txt').read_text().splitlines()
            parsed=[re.fullmatch(r'frame_(\d+)_(\d+)',x) for x in stamps]
            if not all(parsed):raise ValueError('Unexpected capture timestamp schema')
            microseconds=np.asarray([int(x[1]) for x in parsed],dtype=np.int64)
            if not np.array_equal([int(x[2]) for x in parsed],np.arange(len(parsed))):raise ValueError('Unordered capture frame indices')
            times=(microseconds-microseconds[0]).astype(np.float64)*1e-6
            if len(frames)!=len(times) or len(times)<16 or not np.all(np.diff(times)>0):raise ValueError('Frame/timestamp mismatch')
            rgb=np.asarray(frames,dtype=np.uint8)
            # Pixel-change arc length is a sampling heuristic, not measured optical flow.
            gray=rgb.astype(np.float32).mean(-1)
            motion=np.abs(np.diff(gray,axis=0)).mean((1,2));arc=np.r_[0,np.cumsum(motion)]
            active=np.unique(np.searchsorted(arc,np.linspace(0,arc[-1],12))) if arc[-1]>0 else np.array([],int)
            selected=np.unique(np.r_[0,len(times)-1,active,np.linspace(0,len(times)-1,16).round().astype(int)])
            # Fixed 16 frames preserve endpoints plus active arc-length samples.
            if len(selected)>16:selected=selected[np.linspace(0,len(selected)-1,16).round().astype(int)]
            if len(selected)<16:raise ValueError('Insufficient distinct frames')
            object_id=file['path'].split('/')[1];sample_id=path.stem
            target=directory/(sample_id+'.npz')
            np.savez_compressed(target,rgb=rgb[selected],timestamps_seconds=times[selected],source_frame_indices=selected)
            row=dict(sample_id='deform360:'+sample_id,object_id=object_id,asset_group='deform360:'+object_id,
                split='val' if object_id in VALIDATION else 'train',source='https://huggingface.co/datasets/brownu/deform360',
                source_revision=source['revision'],pack=str(target.relative_to(ROOT)),source_video_sha256=file['sha256'],
                timestamps_unit='second',parameters={},role='motion_self_supervision_only',frames=len(selected),
                source_frames=len(frames),mean_pixel_change=float(motion.mean()),
                timestamp_sha256=hashlib.sha256(path.with_suffix('.txt').read_bytes()).hexdigest())
            report['rows'].append(row);update(completed=len(report['rows']))
        if len(report['rows'])!=len(OBJECTS):raise ValueError('Missing bounded real-video object')
        train={r['asset_group'] for r in report['rows'] if r['split']=='train'}
        val={r['asset_group'] for r in report['rows'] if r['split']=='val'}
        if train&val:raise ValueError('Real object leakage')
        manifest=ROOT/'data/manifests/external_motion_v1.jsonl'
        manifest.write_text(''.join(json.dumps(r)+'\n' for r in report['rows']))
        update(status='complete',train_objects=len(train),val_objects=len(val),manifest=str(manifest.relative_to(ROOT)))
        atomic(output/'status.json',dict(status='complete',reason='real_motion_inputs_prepared_not_yet_trained'))
    except Exception as error:
        update(status='failed',error_type=type(error).__name__,error=str(error));atomic(output/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
