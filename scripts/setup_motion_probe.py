"""Append one bounded, capacity-matched architecture screen; never requeue history."""
import copy
import json
from pathlib import Path

from engine_fixture import atomic


def main():
    root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/'docs/OPTIMIZATION_PLAN.json').read_text())
    baseline=json.loads((root/'configs/g1f_cache.json').read_text())
    for label,feedback,weight in [('B_motion_rgb','none',0.),('C_motion_aux','none',.1),
                                  ('D_motion_fields','fields',.1),('G_motion_features','features',.1)]:
        identifier='m1_'+label+'_seed17'
        if any(j['id']==identifier for j in plan['jobs']):continue
        cfg=copy.deepcopy(baseline)
        cfg['model'].update(architecture='motion_field',feedback=feedback,width=256,heads=8,layers=2)
        cfg['data']['prepared_dir']='data/prepared/g1f_dinov3l16_local'
        cfg['train'].update(output='runs/'+identifier,seed=17,epochs=20,field_epochs=0,
            field_weight=weight,contact_loss='bce_dice',batch_size=4,accumulation=2,workers=2,
            val_batch_size=4,val_workers=2,early_stop_patience=4,early_stop_min_joint_epochs=6,
            resume='',initialize='',initialize_rgb_baseline='')
        config='configs/runtime/'+identifier+'.json';atomic(root/config,cfg)
        job=dict(id=identifier,name=label,stage='M1',kind='train',queue_group='architecture_probe',
            config=config,source_config=config,devices='',output=cfg['train']['output'],
            log='runs/phase_logs/'+identifier+'.log',state='queued',desired='running',
            resources={'minimum_free_mib':24576},readiness_file='docs/NEW_A800_READINESS.json',
            split_counts={'train':512,'val':128,'test':0})
        if not any(j['id']==identifier for j in plan['jobs']):plan['jobs'].append(job)
    atomic(root/'docs/OPTIMIZATION_PLAN.json',plan)
    print('Appended four M1 architecture diagnostics, seed17; test untouched.')


if __name__=='__main__':main()
