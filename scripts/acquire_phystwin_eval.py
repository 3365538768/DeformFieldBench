"""Download two bounded official evaluation sequences, without unpickling labels."""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import zipfile

import cv2

from engine_fixture import atomic, wait_if_paused
from fetch_sample_subset import HTTPRange
from acquire_external_pilots import configure_proxy

ROOT=Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT);output=ROOT/'runs/phystwin_eval_inputs';output.mkdir(parents=True,exist_ok=True)
    configure_proxy(ROOT)
    source=json.loads((ROOT/'docs/DATA_SOURCE_PILOTS.json').read_text())['sources']['phystwin']
    url=source['source']+'/resolve/'+source['revision']+'/data.zip'
    stream=HTTPRange(url,source['archive_bytes'])
    root=ROOT/'data/external_eval/phystwin';root.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',completed=0,total=12,source=source['source'],revision=source['revision'],
        used_for_training=False,parameter_gt_available=False,files=[],evaluation_role='RGB and RGB-D tracks must be reported separately')
    def update(**kw):
        report.update(kw,updated_unix=time.time(),detail='PhysTwin 两场景的官方视频/划分/元数据；不把弹簧系数当 E。')
        atomic(ROOT/'docs/PHYSTWIN_EVAL_INPUTS.json',report)
    try:
        update()
        with zipfile.ZipFile(stream) as archive:
            for scene in ['weird_package','single_push_sloth']:
                for filename in ['metadata.json','split.json','final_data.mp4','color/0.mp4','color/1.mp4','color/2.mp4']:
                    wait_if_paused(output);name='data/different_types/'+scene+'/'+filename
                    info=archive.getinfo(name)
                    if info.file_size>32*1024**2:raise ValueError('Unexpected large evaluation member')
                    target=(root/name).resolve()
                    if not target.is_relative_to(root):raise ValueError('Unsafe archive path')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    data=archive.read(info);target.write_bytes(data)
                    row=dict(scene=scene,member=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),zip_crc_verified=True)
                    if filename.endswith('.mp4'):
                        cap=cv2.VideoCapture(str(target));frames=0
                        while True:
                            ok,_=cap.read()
                            if not ok:break
                            frames+=1
                        cap.release()
                        if frames<2:raise ValueError('Evaluation video decode failed')
                        row['decoded_frames']=frames
                    else:
                        value=json.loads(data);row['metadata']=value
                    report['files'].append(row);update(completed=len(report['files']))
        update(status='complete',range_downloaded_bytes=stream.downloaded,
            scope='Selected RGB videos/metadata only; no full archive SHA, depth/tracks or official optimizer run yet. Both scenes are evaluation-only.')
        atomic(output/'status.json',dict(status='complete',reason='bounded_official_eval_videos_ready'))
    except Exception as error:
        update(status='failed',error_type=type(error).__name__,error=str(error));atomic(output/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
