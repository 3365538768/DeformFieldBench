"""Object-paired M3 response representation feedback, with explicit costs."""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from fieldbridge.data import read_manifest
from fieldbridge.runtime import sha256
from audit_single_video_controls import common_records
from audit_motion_dependence import paired_bootstrap
from setup_local_response_controls import CONTROLS
from engine_fixture import atomic,wait_if_paused

ROOT=Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT);out=ROOT/'runs/m3_local_response_audit';out.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',completed=0,total=6,models={},comparisons={},test_untouched=True,
        detail='六组紧凑单视频、相同参数量与取样：全局/相对运动双库对照；物体配对误差与效率反馈。',
        scope='Single-seed validation-selected checkpoints; no original MPM or cross-engine material claim, no final test. Residual displacement is not strain.')
    selected={};stats=json.loads((ROOT/'data/manifests/g1_diagnostic_512_128.stats.json').read_text())
    val={r['sample_id']:r for r in read_manifest('data/manifests/g1_diagnostic_512_128.jsonl') if r['split']=='val'}
    try:
        for label,_,_,_ in CONTROLS:
            wait_if_paused(out);folder=ROOT/('runs/m3_'+label+'_seed17')
            ckpt=torch.load(folder/'best_material.pt',map_location='cpu',weights_only=False)
            data=ckpt['config']['data']
            if data['view_policy']!='single' or data['single_view_layout']!='compact':raise ValueError('M3 compact single-view contract mismatch')
            if ckpt['manifest_sha256']!=sha256(ROOT/data['manifest']) or ckpt['motion_target_index_sha256']!=sha256(ROOT/data['motion_targets_dir']/'index.json'):raise ValueError('M3 data provenance mismatch')
            records=[json.loads(x) for x in (folder/f"val_epoch_{ckpt['epoch']:04d}/predictions.jsonl").read_text().splitlines()]
            common=common_records(records,ckpt['stats'],stats,val);selected[label]=common
            errors=[np.abs(np.array(r['prediction_z'])-r['target_z'])[r['target_mask']] for r in common.values()]
            timing=[json.loads(x) for x in (folder/'timing.jsonl').read_text().splitlines()]
            run=json.loads((folder/'run.json').read_text())
            report['models'][label]=dict(common_standardized_mae=float(np.concatenate(errors).mean()),best_epoch=ckpt['epoch']+1,
                epochs_completed=len(timing),trainable_parameters=run['trainable_parameters'],
                timing_medians={k:float(np.median([r[k] for r in timing if k in r])) for k in ['train_seconds','data_wait_seconds','validation_seconds'] if any(k in r for r in timing)},
                checkpoint_sha256=sha256(folder/'best_material.pt'),source_sha256=ckpt['source_sha256'])
            report.update(completed=len(report['models']),updated_unix=time.time());atomic(ROOT/'docs/M3_LOCAL_RESPONSE_AUDIT.json',report)
        if len({m['trainable_parameters'] for m in report['models'].values()})!=1:raise ValueError('M3 capacity mismatch')
        for control in ['B_rgb','C_aux','D_raw','G_raw','G_decomposed']:
            report['comparisons']['D_decomposed_minus_'+control]=paired_bootstrap(selected['D_decomposed'],selected[control])
        scores={k:v['common_standardized_mae'] for k,v in report['models'].items()};best=min(['B_rgb','C_aux','G_raw','G_decomposed'],key=scores.get)
        gain=1-scores['D_decomposed']/scores[best]
        report.update(status='complete',best_control=best,field_gain_over_best_control=gain,
            passes_2_percent_direction_gate=gain>.02,automatic_full_training=False,updated_unix=time.time(),
            next_step='Independent seeds and frozen test protocol only after reviewing matched controls; otherwise repair teacher/representation or staged optimization.')
        atomic(ROOT/'docs/M3_LOCAL_RESPONSE_AUDIT.json',report);atomic(out/'status.json',dict(status='complete',reason='matched_local_response_feedback_ready'))
    except Exception as error:
        report.update(status='failed',error=str(error));atomic(ROOT/'docs/M3_LOCAL_RESPONSE_AUDIT.json',report)
        atomic(out/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
