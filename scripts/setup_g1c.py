"""Bounded local temporal residual diagnostics, with a common frozen RGB start."""
import argparse
import copy
import json
from optimization_manager import ROOT, atomic, merge_jobs, read


MATRIX = [('C_frozen_rgb','none',False), ('D_local_fields','fields',False),
          ('G_local_features','features',False), ('D_local_detached','fields',True),
          ('O_local_oracle','oracle',False), ('S_local_shuffled','oracle_shuffled',False)]


def setup(register=False):
    base = read(ROOT/'configs/g1b_D_weight01_balanced_seed17.json')
    base['model'].update(architecture='local_residual')
    base['train'].update(epochs=24, field_epochs=2, field_weight=.1, resume='', initialize='',
                         initialize_rgb_baseline='runs/g1_B_rgb_only_seed17/best_material.pt')
    jobs=[]
    rows=[('field_fit','fields',False),*MATRIX]
    for name,feedback,detach in rows:
        identifier=f'g1c_{name}_seed17';cfg=copy.deepcopy(base)
        cfg['model'].update(feedback=feedback,detach_feedback=detach)
        cfg['train']['output']='runs/'+identifier
        if name=='field_fit':
            cfg['train'].update(epochs=20,batch_size=4,accumulation=1,field_weight=1.,lr=.001)
        config='configs/'+identifier+'.json'
        existing=read(ROOT/config)
        if existing is not None and existing!=cfg:raise ValueError('Different existing config: '+config)
        atomic(ROOT/config,cfg)
        job=dict(id=identifier,name=name,stage='G1c',kind='train',config=config,devices='',
                 output=cfg['train']['output'],log='runs/phase_logs/'+identifier+'.log',
                 state='queued',desired='running',split_counts=dict(train=64 if name=='field_fit' else 512,
                                                                  val=0 if name=='field_fit' else 128,test=0))
        if name=='field_fit':job.update(role='field_fit',entrypoint='scripts/overfit_fields.py')
        jobs.append(job)
    if register:
        result=read(ROOT/'docs/G1B_RESULTS.json',{})
        if result.get('status')!='complete':raise ValueError('Complete G1b before registering G1c')
        plan=read(ROOT/'docs/OPTIMIZATION_PLAN.json');plan['jobs']=merge_jobs(plan['jobs'],jobs)
        atomic(ROOT/'docs/OPTIMIZATION_PLAN.json',plan)
    print(json.dumps({'registered':register,'jobs':[j['id'] for j in jobs]},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true')
    setup(parser.parse_args().register)
