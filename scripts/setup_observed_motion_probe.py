"""Append matched order-aware B/C/D/G and a privileged pseudo-field diagnostic."""
import copy
import json
from pathlib import Path
from engine_fixture import atomic

ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'docs/OPTIMIZATION_PLAN.json').read_text())
    baseline=json.loads((ROOT/'configs/g1f_cache.json').read_text())
    quality=json.loads((ROOT/'docs/M2_OBSERVED_MOTION.json').read_text())
    if quality['status']!='complete' or quality['completed']!=640:raise ValueError('Motion overlay is incomplete')
    for label,feedback,weight in [('B_rgb','none',0.),('C_aux','none',.1),('D_fields','fields',.1),
                                  ('G_features','features',.1),('O_pseudo_oracle','oracle',.1)]:
        identifier='m2v2_'+label+'_seed17'
        if any(j['id']==identifier for j in plan['jobs']):continue
        cfg=copy.deepcopy(baseline)
        cfg['model'].update(architecture='motion_temporal',field_mode='observed_motion',feedback=feedback,width=256,heads=8,layers=2)
        cfg['data'].update(prepared_dir='data/prepared/g1f_dinov3l16_local',motion_targets_dir='data/prepared/m2_observed_motion',prepared_cache_gib=0.)
        cfg['train'].update(output='runs/'+identifier,purpose='diagnostic',seed=17,epochs=20,field_epochs=0,
            field_weight=weight,contact_loss='bce_dice',batch_size=4,accumulation=2,workers=4,
            val_batch_size=4,val_workers=4,early_stop_patience=4,early_stop_min_joint_epochs=6,
            resume='',initialize='',initialize_rgb_baseline='')
        filename='configs/runtime/'+identifier+'.json';atomic(ROOT/filename,cfg)
        plan['jobs'].append(dict(id=identifier,name=label,stage='M2',kind='train',queue_group='observed_motion_probe',
            config=filename,source_config=filename,devices='',output=cfg['train']['output'],
            log='runs/phase_logs/'+identifier+'.log',state='queued',desired='running',
            resources={'minimum_free_mib':24576},readiness_file='docs/NEW_A800_READINESS.json',
            split_counts={'train':512,'val':128,'test':0},depends_on=['m2_observed_motion']))
    atomic(ROOT/'docs/OPTIMIZATION_PLAN.json',plan)
    print('Five M2 diagnostics appended; pseudo-oracle uses GT object masks and is not a deployment baseline.')


if __name__=='__main__':main()
