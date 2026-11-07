"""Test the explicit RGB-to-residual shortcut found by G1c interventions."""
import argparse
import copy
import json
from optimization_manager import ROOT, atomic, merge_jobs, read


def setup(register=False):
    jobs=[]
    for name,feedback in [('D_local_fields','fields'),('G_local_features','features'),
                          ('O_local_oracle','oracle'),('S_local_shuffled','oracle_shuffled')]:
        cfg=read(ROOT/f'configs/g1c_{name}_seed17.json')
        cfg=copy.deepcopy(cfg);cfg['model']['residual_rgb_input']=False
        identifier=f'g1d_{name}_seed17';cfg['train']['output']='runs/'+identifier
        config='configs/'+identifier+'.json'
        existing=read(ROOT/config)
        if existing is not None and existing!=cfg:raise ValueError('Existing config differs: '+config)
        atomic(ROOT/config,cfg)
        jobs.append(dict(id=identifier,name=name,stage='G1d',kind='train',config=config,devices='',
                         output=cfg['train']['output'],log='runs/phase_logs/'+identifier+'.log',state='queued',
                         desired='running',split_counts=dict(train=512,val=128,test=0)))
    if register:
        if read(ROOT/'docs/G1C_RESULTS.json',{}).get('status')!='complete':
            raise ValueError('Complete G1c before G1d')
        plan=read(ROOT/'docs/OPTIMIZATION_PLAN.json');plan['jobs']=merge_jobs(plan['jobs'],jobs)
        atomic(ROOT/'docs/OPTIMIZATION_PLAN.json',plan)
    print(json.dumps({'registered':register,'jobs':[j['id'] for j in jobs]},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true')
    setup(parser.parse_args().register)
