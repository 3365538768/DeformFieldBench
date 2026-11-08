"""Register equal-budget original/repeat-sampled single-video controls."""
import copy
import json
from pathlib import Path
from engine_fixture import atomic

ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'docs/OPTIMIZATION_PLAN.json').read_text())
    base=json.loads((ROOT/'configs/single_video_base.json').read_text())
    for cohort in ['original','repeat']:
        for label,feedback,weight in [('B_rgb','none',0.),('C_aux','none',.1),('D_fields','fields',.1),('G_features','features',.1)]:
            identifier=f'd3_{cohort}_{label}_seed17'
            if any(j['id']==identifier and j['state'] not in {'queued'} for j in plan['jobs']):raise ValueError('Cannot replace started D3 job')
            plan['jobs']=[j for j in plan['jobs'] if j['id']!=identifier]
            cfg=copy.deepcopy(base)
            cfg['data'].update(view_policy='single',single_view_layout='masked',view_dropout=0.,prepared_cache_gib=0.)
            if cohort=='repeat':
                cfg['data'].update(manifest='data/manifests/d3_balanced_repeat_512_128.jsonl',
                    stats='data/manifests/d3_balanced_repeat_512_128.stats.json',
                    prepared_dir='data/prepared/d3_balanced_repeat_dinov3',motion_targets_dir='data/prepared/d3_balanced_repeat_motion')
            cfg['model'].update(feedback=feedback,detach_feedback=False)
            cfg['train'].update(output='runs/'+identifier,field_weight=weight,seed=17,epochs=20,
                resume='',initialize='',initialize_rgb_baseline='',field_epochs=0,
                early_stop_patience=4,early_stop_min_joint_epochs=6)
            filename='configs/runtime/'+identifier+'.json';atomic(ROOT/filename,cfg)
            plan['jobs'].append(dict(id=identifier,name=cohort+'_'+label,stage='D3',kind='train',queue_group='single_video_probe',
                cohort=cohort,control_label=label,config=filename,source_config=filename,devices='',output=cfg['train']['output'],
                log='runs/phase_logs/'+identifier+'.log',state='queued',desired='running',resources={'minimum_free_mib':24576},
                readiness_file='docs/NEW_A800_READINESS.json',split_counts=dict(train=512,val=128,test=0),depends_on=['d3_balanced_prepare']))
    atomic(ROOT/'docs/OPTIMIZATION_PLAN.json',plan)
    print('Eight single-video controls queued; random train camera and fixed first valid validation camera.')


if __name__=='__main__':main()
