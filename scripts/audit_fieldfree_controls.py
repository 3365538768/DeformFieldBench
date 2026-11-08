"""Matched point results, exact mixture intervals and conservative prior checks."""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from fieldbridge.runtime import sha256
from fieldbridge.posterior import posterior_summary
from audit_motion_dependence import paired_bootstrap
from engine_fixture import atomic,wait_if_paused

ROOT=Path(__file__).resolve().parents[1]
LABELS=['N0_video_point','N1_prior_point','N2_prior_posterior']


def main():
    os.chdir(ROOT);torch.set_num_threads(1);out=ROOT/'runs/nf_posterior_audit';out.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',completed=0,total=3,models={},comparisons={},test_untouched=True,
        detail='零场三组统一尺度、物体配对及精确混合后验区间核验；单种子验证筛查，尚非最终测试。')
    selected={}
    try:
        for label in LABELS:
            wait_if_paused(out);folder=ROOT/('runs/nf_'+label+'_seed17')
            ckpt=torch.load(folder/'best_material.pt',weights_only=False,map_location='cpu');cfg=ckpt['config']
            if cfg['model']['architecture']!='direct_video' or cfg['model']['field_mode']!='none' or cfg['model']['feedback']!='none' or cfg['train']['field_weight'] or cfg['data']['motion_targets_dir']:
                raise ValueError('Zero-field contract violated')
            if ckpt['manifest_sha256']!=sha256(cfg['data']['manifest']) or ckpt['material_overlay_sha256']['material_prior_file']!=sha256(cfg['data']['material_prior_file']):
                raise ValueError('Field-free input provenance mismatch')
            records=[json.loads(s) for s in (folder/f"val_epoch_{ckpt['epoch']:04d}/predictions.jsonl").read_text().splitlines()]
            selected[label]={r['sample_id']:r for r in records}
            error=np.concatenate([np.abs(np.array(r['prediction_z'])-r['target_z'])[r['target_mask']] for r in records])
            run=json.loads((folder/'run.json').read_text());timing=[json.loads(s) for s in (folder/'timing.jsonl').read_text().splitlines()]
            report['models'][label]=dict(common_standardized_mae=float(error.mean()),best_epoch=ckpt['epoch']+1,epochs_completed=len(timing),
                trainable_parameters=run['trainable_parameters'],checkpoint_sha256=sha256(folder/'best_material.pt'),
                posterior=posterior_summary(records),uncertainty_trained=cfg['train']['nll_weight']>0,
                timing_medians={k:float(np.median([r[k] for r in timing])) for k in ['train_seconds','data_wait_seconds','validation_seconds']})
            report.update(completed=len(report['models']),updated_unix=time.time());atomic(ROOT/'docs/NF_AUDIT.json',report)
        if len({m['trainable_parameters'] for m in report['models'].values()})!=1:raise ValueError('Unequal capacity')
        if any(set(x)!=set(selected[LABELS[0]]) for x in selected.values()):raise ValueError('Validation ID mismatch')
        targets=list(selected[LABELS[0]].values())
        constant=np.concatenate([np.abs(np.asarray(r['target_z']))[r['target_mask']] for r in targets])
        prior_report=json.loads((ROOT/'docs/NF_STATIC_PREPARE.json').read_text())
        report['reference_baselines']=dict(train_codec_mean_standardized_mae=float(constant.mean()),
            static_first_frame_mae=prior_report['static_validation_mae'],
            note='Training-mean constant and OOF/static ridge baseline; same fixed validation IDs, not tuned on test.')
        for left,right in [(LABELS[1],LABELS[0]),(LABELS[2],LABELS[1]),(LABELS[2],LABELS[0])]:
            report['comparisons'][left+'_minus_'+right]=paired_bootstrap(selected[left],selected[right])
        baseline=report['models'][LABELS[0]]['common_standardized_mae'];prior=report['models'][LABELS[1]]['common_standardized_mae'];prob=report['models'][LABELS[2]]['common_standardized_mae']
        report.update(status='complete',updated_unix=time.time(),prior_gain_over_direct=1-prior/baseline,posterior_point_gain_over_direct=1-prob/baseline,
            automatic_full_training=False,original_video_teacher_admitted=False,
            next_step='Assess static information and interval coverage before independent seeds; original-video teacher distillation remains gated by geometry/loading and response validation.')
        historical=json.loads((ROOT/'docs/B1_PIXEL_AUDIT.json').read_text())
        best=min(v['common_standardized_mae'] for v in historical['models'].values())
        report['historical_sanity']=dict(best_B1_mae=best,best_current_mae=min(baseline,prior,prob),
            beats_B1=min(baseline,prior,prob)<best,
            scope='Different parameter count, objective and source; context only, not a matched bridge-removal causal estimate.')
        atomic(ROOT/'docs/NF_AUDIT.json',report);atomic(out/'status.json',dict(status='complete',reason='fieldfree_matched_and_posterior_feedback_ready'))
    except Exception as error:
        report.update(status='failed',error=str(error));atomic(ROOT/'docs/NF_AUDIT.json',report);atomic(out/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
