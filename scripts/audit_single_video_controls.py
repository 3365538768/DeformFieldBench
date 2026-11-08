"""Compare single-video controls with one frozen evaluation scale and objects.

Each cohort trains/early-stops with its own train-only codec. Cross-sampling
results combine sampling and normalization effects, not a pure sampling estimate.
"""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from fieldbridge.data import read_manifest, ParameterCodec
from fieldbridge.runtime import sha256
from audit_motion_dependence import paired_bootstrap
from engine_fixture import atomic, wait_if_paused

ROOT=Path(__file__).resolve().parents[1]
LABELS=['B_rgb','C_aux','D_fields','G_features']


def common_records(records,stats,reference,rows):
    if len(records)!=128 or len({r['sample_id'] for r in records})!=128:raise ValueError('Single-video validation coverage mismatch')
    ref=ParameterCodec(reference);result={}
    for row in records:
        sid=row['sample_id']
        if sid not in rows or rows[sid]['object_id']!=row['object_id']:raise ValueError('Validation ID/object mismatch')
        target,mask=ref.encode(rows[sid]['parameters'])
        source=(np.asarray(row['target_z'])*np.asarray(stats['std'])+stats['mean']-reference['mean'])/reference['std']
        if not np.allclose(source[np.asarray(row['target_mask'],bool)],target.numpy()[mask.numpy()],atol=1e-5):raise ValueError('Codec target conversion mismatch')
        prediction=(np.asarray(row['prediction_z'])*np.asarray(stats['std'])+stats['mean']-reference['mean'])/reference['std']
        result[sid]=dict(row,prediction_z=prediction.tolist(),target_z=target.tolist(),target_mask=mask.tolist())
    if set(result)!=set(rows):raise ValueError('Validation set differs')
    return result


def main():
    os.chdir(ROOT);out=ROOT/'runs/d3_single_video_audit';out.mkdir(parents=True,exist_ok=True)
    reference=json.loads((ROOT/'data/manifests/g1_diagnostic_512_128.stats.json').read_text())
    val={r['sample_id']:r for r in read_manifest('data/manifests/g1_diagnostic_512_128.jsonl') if r['split']=='val'}
    report=dict(status='running',completed=0,total=8,models={},comparisons={},test_untouched=True,
        common_eval_stats_sha256=sha256(ROOT/'data/manifests/g1_diagnostic_512_128.stats.json'),
        detail='固定单视角、相同128条验证；换回统一评价尺度，按物体配对区间与原生参数分项。',
        scope='Single seed, validation-selected checkpoints, no test or online-fitting claim. Sampling and train-normalization change together across cohorts.')
    selected={}
    try:
        for cohort in ['original','repeat']:
            for label in LABELS:
                wait_if_paused(out);identifier=f'd3_{cohort}_{label}_seed17';folder=ROOT/'runs'/identifier
                checkpoint=torch.load(folder/'best_material.pt',map_location='cpu',weights_only=False)
                config=checkpoint['config'];data=config['data']
                if data['view_policy']!='single' or checkpoint['manifest_sha256']!=sha256(ROOT/data['manifest']):raise ValueError('Checkpoint input/manifest contract mismatch')
                if checkpoint['motion_target_index_sha256']!=sha256(ROOT/data['motion_targets_dir']/'index.json'):raise ValueError('Checkpoint pseudo-label mismatch')
                records=[json.loads(x) for x in (folder/f"val_epoch_{checkpoint['epoch']:04d}/predictions.jsonl").read_text().splitlines()]
                shared=common_records(records,checkpoint['stats'],reference,val);selected[(cohort,label)]=shared
                errors=[abs(np.asarray(r['prediction_z'])-r['target_z'])[np.asarray(r['target_mask'],bool)] for r in shared.values()]
                per_parameter={}
                for j,name in enumerate(['E','nu','rho','sigma_y']):
                    values=[abs(r['prediction_z'][j]-r['target_z'][j]) for r in shared.values() if r['target_mask'][j]]
                    per_parameter[name]=dict(common_standardized_mae=float(np.mean(values)),count=len(values))
                summary=json.loads((folder/f"val_epoch_{checkpoint['epoch']:04d}/summary.json").read_text())
                report['models'][cohort+'_'+label]=dict(common_standardized_mae=float(np.concatenate(errors).mean()),
                    own_train_standardized_mae=summary['standardized_mae'],per_parameter=per_parameter,
                    native_parameter_metrics=summary['parameters'],checkpoint_epoch=checkpoint['epoch']+1,
                    checkpoint_sha256=sha256(folder/'best_material.pt'))
                report.update(completed=len(report['models']),updated_unix=time.time());atomic(ROOT/'docs/D3_SINGLE_VIDEO_AUDIT.json',report)
        for cohort in ['original','repeat']:
            for control in ['B_rgb','C_aux','G_features']:
                report['comparisons'][cohort+'_D_minus_'+control]=paired_bootstrap(selected[(cohort,'D_fields')],selected[(cohort,control)])
        for label in LABELS:
            report['comparisons']['repeat_minus_original_'+label]=paired_bootstrap(selected[('repeat',label)],selected[('original',label)])
        gates={}
        for cohort in ['original','repeat']:
            scores={label:report['models'][cohort+'_'+label]['common_standardized_mae'] for label in LABELS}
            gates[cohort]=dict(passes_2_percent_direction_gate=scores['D_fields']<.98*min(scores[l] for l in ['B_rgb','C_aux','G_features']),
                best_control=min(['B_rgb','C_aux','G_features'],key=scores.get),field_gain_over_best_control=1-scores['D_fields']/min(scores[l] for l in ['B_rgb','C_aux','G_features']))
        report.update(status='complete',routing_gates=gates,updated_unix=time.time(),
            next_step='review_single_video_sampling_then_motion_representation',automatic_full_training=False)
        atomic(ROOT/'docs/D3_SINGLE_VIDEO_AUDIT.json',report)
        atomic(out/'status.json',dict(status='complete',reason='common_codec_matched_single_video_audit_complete'))
    except Exception as error:
        report.update(status='failed',error=str(error));atomic(ROOT/'docs/D3_SINGLE_VIDEO_AUDIT.json',report)
        atomic(out/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
