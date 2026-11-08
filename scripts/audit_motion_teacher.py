"""Pre-registered synthetic correspondence check before pseudo-label generation."""
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from engine_fixture import atomic, wait_if_paused
from visible_motion import estimate

ROOT=Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT);cv2.setNumThreads(1)
    output=ROOT/'runs/m2_teacher_audit';output.mkdir(parents=True,exist_ok=True)
    clips=json.loads((ROOT/'docs/G3_RENDER_QUALITY.json').read_text())['clips']
    report=dict(status='running',completed=0,total=len(clips),rows=[],
        thresholds=dict(minimum_valid_coverage=.5,maximum_mean_epe_px=1.,maximum_error_to_zero_ratio=.5,minimum_passing_fraction=.75),
        detail='在渲染对应真值上审计 DIS 教师；同时报告覆盖率和全有效区误差，防止用拒绝难例掩盖失败。',
        scope='Renderer diagnostic clips only; quality screen, not a real-world benchmark or solver-force admission.')
    try:
        atomic(ROOT/'docs/M2_TEACHER_AUDIT.json',report)
        for row in clips:
            wait_if_paused(output);folder=ROOT/row['path']
            with np.load(folder/'flow.npz',allow_pickle=False) as z:
                truth=z['flow_pixels'];gold=z['valid'].astype(bool)
            index=int(np.argmin(np.abs(np.asarray(row['frame_times_s'])-row['peak_time_s'])))
            a=np.asarray(Image.open(folder/'frame_000.png').convert('RGB'))
            b=np.asarray(Image.open(folder/f'frame_{index:03d}.png').convert('RGB'))
            # Reliability estimation uses no segmentation oracle; gold only scores it.
            mask_a=np.ones(gold.shape,bool);mask_b=np.ones_like(mask_a)
            flow,valid,residual=estimate(a,b,mask_a,mask_b)
            epe=np.linalg.norm(flow-truth,axis=-1);zero=np.linalg.norm(truth,axis=-1)
            support=gold&valid;coverage=float(support.sum()/max(gold.sum(),1))
            mean=float(epe[support].mean()) if support.any() else None
            baseline=float(zero[support].mean()) if support.any() else None
            ratio=mean/max(baseline,1e-8) if mean is not None else None
            passed=bool(coverage>=.5 and mean is not None and mean<=1. and ratio<=.5)
            report['rows'].append(dict(id=row['id'],paired_group_id=row.get('paired_group_id'),occlusion_variant=row['occlusion_variant'],
                valid_coverage=coverage,retained_mean_epe_px=mean,zero_motion_mean_epe_px=baseline,error_to_zero_ratio=ratio,
                all_gold_mean_epe_px=float(epe[gold].mean()),gold_pixels=int(gold.sum()),passed=passed))
            report.update(completed=len(report['rows']),updated_unix=time.time())
            atomic(ROOT/'docs/M2_TEACHER_AUDIT.json',report)
        fraction=sum(r['passed'] for r in report['rows'])/len(clips)
        report.update(status='complete',passing_fraction=fraction,teacher_accepted=fraction>=.75,
                      route='bounded_deformfield_pseudo_motion' if fraction>=.75 else 'evaluate_stronger_correspondence_teacher')
        atomic(ROOT/'docs/M2_TEACHER_AUDIT.json',report)
        atomic(output/'status.json',dict(status='complete',reason=report['route']))
    except Exception as error:
        report.update(status='failed',error=str(error));atomic(ROOT/'docs/M2_TEACHER_AUDIT.json',report)
        atomic(output/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
