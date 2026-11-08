"""Feedback on paired RGB branch and field bridge controls."""
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
from engine_fixture import atomic,wait_if_paused

ROOT=Path(__file__).resolve().parents[1]
LABELS=['B_rgb','C_aux','D_fields','G_features']


def main():
    os.chdir(ROOT);out=ROOT/'runs/b1_pixel_audit';out.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',completed=0,total=8,models={},comparisons={},routing_gates={},test_untouched=True,
        detail='RGB像素分支开/关各B/C/D/G：同容量、同初始化、同输入与监督，分开检验像素与场桥收益。',
        scope='Single seed, validation-selected checkpoints; existing 224px RGB, no external pretraining or backbone unfreezing. No final test claim.')
    reference=json.loads((ROOT/'data/manifests/g1_diagnostic_512_128.stats.json').read_text())
    val={r['sample_id']:r for r in read_manifest('data/manifests/g1_diagnostic_512_128.jsonl') if r['split']=='val'}
    selected={}
    try:
        for arm in ['off','on']:
            for label in LABELS:
                wait_if_paused(out);name=arm+'_'+label;folder=ROOT/('runs/b1_'+name+'_seed17')
                ckpt=torch.load(folder/'best_material.pt',map_location='cpu',weights_only=False);cfg=ckpt['config']
                if not cfg['model']['pixel_branch'] or cfg['model']['pixel_enabled']!=(arm=='on') or cfg['data']['single_view_layout']!='compact':raise ValueError('Pixel branch contract mismatch')
                if ckpt['manifest_sha256']!=sha256(ROOT/cfg['data']['manifest']) or ckpt['rgb_cache_index_sha256']!=sha256(ROOT/cfg['data']['rgb_cache_dir']/'index.json') or ckpt['motion_target_index_sha256']!=sha256(ROOT/cfg['data']['motion_targets_dir']/'index.json'):raise ValueError('Pixel run input provenance mismatch')
                records=[json.loads(x) for x in (folder/f"val_epoch_{ckpt['epoch']:04d}/predictions.jsonl").read_text().splitlines()]
                selected[name]=common_records(records,ckpt['stats'],reference,val)
                errors=[np.abs(np.asarray(r['prediction_z'])-r['target_z'])[r['target_mask']] for r in selected[name].values()]
                run=json.loads((folder/'run.json').read_text());timing=[json.loads(x) for x in (folder/'timing.jsonl').read_text().splitlines()]
                report['models'][name]=dict(common_standardized_mae=float(np.concatenate(errors).mean()),best_epoch=ckpt['epoch']+1,
                    epochs_completed=len(timing),trainable_parameters=run['trainable_parameters'],checkpoint_sha256=sha256(folder/'best_material.pt'),
                    timing_medians={k:float(np.median([r[k] for r in timing])) for k in ['train_seconds','data_wait_seconds','validation_seconds']})
                report.update(completed=len(report['models']),updated_unix=time.time());atomic(ROOT/'docs/B1_PIXEL_AUDIT.json',report)
        if len({m['trainable_parameters'] for m in report['models'].values()})!=1:raise ValueError('Pixel capacity mismatch')
        for arm in ['off','on']:
            scores={label:report['models'][arm+'_'+label]['common_standardized_mae'] for label in LABELS}
            best=min(['B_rgb','C_aux','G_features'],key=scores.get);gain=1-scores['D_fields']/scores[best]
            report['routing_gates'][arm]=dict(best_control=best,field_gain_over_best_control=gain,passes_2_percent_direction_gate=gain>.02)
            for label in ['B_rgb','C_aux','G_features']:report['comparisons'][arm+'_D_minus_'+label]=paired_bootstrap(selected[arm+'_D_fields'],selected[arm+'_'+label])
        for label in LABELS:report['comparisons']['pixel_on_minus_off_'+label]=paired_bootstrap(selected['on_'+label],selected['off_'+label])
        historical=json.loads((ROOT/'docs/M3_LOCAL_RESPONSE_AUDIT.json').read_text())
        previous_control=min(historical['models'][name]['common_standardized_mae'] for name in ['B_rgb','C_aux','G_raw','G_decomposed'])
        current=report['models']['on_D_fields']['common_standardized_mae']
        report['historical_control_sanity']=dict(previous_stage='M3',same_input_information=True,
            initialization_and_parameter_allocation_differ=True,best_previous_no_field_control=previous_control,
            pixel_field_beats_previous_control=current<previous_control,
            passes_conservative_direction_gate=report['routing_gates']['on']['passes_2_percent_direction_gate'] and current<.98*previous_control,
            note='A weak new control caused by initialization or early stopping cannot erase an already stronger same-information baseline. Cross-stage check is a sanity gate, not a matched causal estimate.')
        report.update(status='complete',automatic_full_training=False,updated_unix=time.time(),next_step='Review matched bridge gains and independent seed requirements; pixel gains alone do not justify a field bridge claim.')
        atomic(ROOT/'docs/B1_PIXEL_AUDIT.json',report);atomic(out/'status.json',dict(status='complete',reason='pixel_and_bridge_matched_feedback_ready'))
    except Exception as error:
        report.update(status='failed',error=str(error));atomic(ROOT/'docs/B1_PIXEL_AUDIT.json',report);atomic(out/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
