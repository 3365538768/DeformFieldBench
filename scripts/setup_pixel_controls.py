"""Matched eight-arm test: frozen DINO plus an optional trainable RGB branch."""
import copy
import json
from pathlib import Path
from engine_fixture import atomic
from prepare_pixel_rgb_cache import CACHE

ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'docs/OPTIMIZATION_PLAN.json').read_text())
    base=json.loads((ROOT/'configs/single_video_base.json').read_text())
    protocol=dict(status='registered_before_training',primary='fixed 128 single-camera validation material MAE',seed=17,
        budget='512 train, 20 epochs maximum, batch4 accumulation2, workers4, patience4',
        hypothesis='Lightweight 224px RGB evidence complements frozen 14x14 DINO patch features.',
        controls='B/C/D/G each with the same pixel encoder enabled or multiplied by zero; same trainable parameter count and initialization.',
        interpretation='RGB branch gains are not field-bridge gains. Compare D against B/C/G within each arm; the off arm also controls parameter allocation and initialization.',
        scope='Existing rendered RGB only. No larger source resolution, new external data, unfreezing, SI field labels or force inputs.',
        next_gate='2% field gain over all matched controls then independent seeds; no automatic full training.')
    atomic(ROOT/'docs/B1_PIXEL_PROTOCOL.json',protocol)
    for arm,enabled in [('off',False),('on',True)]:
        for label,feedback,weight in [('B_rgb','none',0.),('C_aux','none',.1),('D_fields','fields',.1),('G_features','features',.1)]:
            identifier=f'b1_{arm}_{label}_seed17'
            if any(j['id']==identifier for j in plan['jobs']):raise ValueError('Pixel job already registered')
            cfg=copy.deepcopy(base);cfg['data'].update(view_policy='single',single_view_layout='compact',rgb_cache_dir=CACHE,view_dropout=0.,prepared_cache_gib=0.)
            cfg['model'].update(feedback=feedback,pixel_branch=True,pixel_enabled=enabled,bridge_decomposition='none',detach_feedback=False)
            cfg['train'].update(output='runs/'+identifier,field_weight=weight,seed=17,epochs=20,resume='',initialize='',
                initialize_rgb_baseline='',field_epochs=0,early_stop_patience=4,early_stop_min_joint_epochs=6)
            filename='configs/runtime/'+identifier+'.json';atomic(ROOT/filename,cfg)
            plan['jobs'].append(dict(id=identifier,name=arm+'_'+label,stage='B1',kind='train',queue_group='pixel_probe',
                cohort=arm,control_label=label,config=filename,source_config=filename,devices='',output=cfg['train']['output'],
                log='runs/phase_logs/'+identifier+'.log',state='queued',desired='running',resources={'minimum_free_mib':24576},
                readiness_file='docs/NEW_A800_READINESS.json',split_counts=dict(train=512,val=128,test=0),depends_on=['b1_pixel_prepare']))
    plan['jobs'].append(dict(id='b1_pixel_audit',name='RGB像素分支匹配反馈',stage='B1a',kind='engine_pipeline',
        args=['scripts/audit_pixel_controls.py'],devices='',output='runs/b1_pixel_audit',log='runs/phase_logs/b1_pixel_audit.log',
        state='queued',desired='running',resources={'device':'cpu'},depends_on=[f'b1_{arm}_{label}_seed17' for arm in ['off','on'] for label in ['B_rgb','C_aux','D_fields','G_features']],
        progress_report='docs/B1_PIXEL_AUDIT.json'))
    atomic(ROOT/'docs/OPTIMIZATION_PLAN.json',plan);print('Eight matched pixel branch controls registered')


if __name__=='__main__':main()
