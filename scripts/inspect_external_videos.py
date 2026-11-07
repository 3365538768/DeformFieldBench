"""Decode real pilot videos and audit capture timestamps before data adaptation."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/external_pilots')
    p.add_argument('--receipt',default='docs/DATA_SOURCE_PILOTS.json')
    p.add_argument('--output',default='docs/EXTERNAL_VIDEO_AUDIT.json');a=p.parse_args()
    receipt=json.loads(Path(a.receipt).read_text())['sources']['deform360']
    root=Path(a.root);records=[];failures=[]
    for entry in receipt['files']:
        if entry['status']!='downloaded':continue
        file=root/'deform360'/entry['path']
        if file.stat().st_size!=entry['bytes'] or hashlib.sha256(file.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Pilot file provenance mismatch: '+entry['path'])
        if file.suffix!='.mp4':continue
        lines=file.with_suffix('.txt').read_text().splitlines()
        parsed=[re.fullmatch(r'frame_(\d+)_(\d+)',line) for line in lines]
        if not all(parsed):raise ValueError('Unexpected timestamp schema')
        us=np.array([int(m[1]) for m in parsed],dtype=np.int64)
        index=np.array([int(m[2]) for m in parsed],dtype=np.int64)
        if not (np.diff(us)>0).all() or not np.array_equal(index,np.arange(len(index))):
            raise ValueError('Capture times or frame indices are not strictly ordered')
        cap=cv2.VideoCapture(str(file))
        if not cap.isOpened():raise ValueError('Cannot decode '+str(file))
        fps=cap.get(cv2.CAP_PROP_FPS);count=0;shape=None
        while True:
            ok,image=cap.read()
            if not ok:break
            if shape is not None and image.shape!=shape:raise ValueError('Resolution changed')
            shape=image.shape;count+=1
        cap.release()
        if count!=len(us):failures.append({'video':entry['path'],'decoded':count,'timestamps':len(us)})
        seconds=(us-us[0]).astype(np.float64)*1e-6
        relative=file.relative_to(Path.cwd()) if file.is_absolute() else file
        record={'sample_id':'deform360:'+file.stem,'source':receipt['source'],'source_revision':receipt['revision'],
                'object_id':entry['path'].split('/')[1],'video':str(relative),'timestamp_file':str(relative.with_suffix('.txt')),
                'decoded_frames':count,'width':shape[1] if shape else None,'height':shape[0] if shape else None,
                'container_fps':fps,'capture_duration_seconds':float(seconds[-1]),
                'capture_dt_median_seconds':float(np.median(np.diff(seconds))),
                'capture_dt_range_seconds':[float(np.diff(seconds).min()),float(np.diff(seconds).max())],
                'video_sha256':entry['sha256'],'split':'unassigned','material_parameters':{},
                'supervision_role':'real_video_motion_pretraining_candidate',
                'note':'No continuous material-parameter labels or SI stress/contact imported; not part of existing training/test manifest.'}
        records.append(record)
    manifest=root/'real_video_candidates.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
    result={'status':'passed' if records and not failures else 'failed','opencv_version':cv2.__version__,
            'videos':records,'failures':failures,'used_for_training':False,
            'scope':'Complete decode and timestamp/hash consistency for the selected deterministic videos only, not domain benchmark validation.'}
    output=Path(a.output);tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(output)
    print(json.dumps(result,indent=2))
    if result['status']!='passed':raise SystemExit(2)


if __name__=='__main__':main()
