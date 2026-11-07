"""Matched normalized/zero-anchored bridge and contact-loss diagnostics."""
import argparse
import copy
import json
from optimization_manager import ROOT, atomic, merge_jobs, read


def setup(register=False):
    jobs=[]
    cases=[('D_local_fields','D_local_fields','balanced_bce'),
           ('G_local_features','G_local_features','balanced_bce'),
           ('O_local_oracle','O_local_oracle','balanced_bce'),
           ('S_local_shuffled','S_local_shuffled','balanced_bce'),
           ('D_dice','D_local_fields','bce_dice'),('G_dice','G_local_features','bce_dice')]
    for name,source,contact_loss in cases:
        cfg=copy.deepcopy(read(ROOT/f'configs/g1d_{source}_seed17.json'))
        cfg['model']['bridge_pooling']='normalized_anchor'
        cfg['train']['contact_loss']=contact_loss
        identifier=f'g1e_{name}_seed17';cfg['train']['output']='runs/'+identifier
        config='configs/'+identifier+'.json';existing=read(ROOT/config)
        if existing is not None and existing!=cfg:raise ValueError('Existing config differs: '+config)
        atomic(ROOT/config,cfg)
        jobs.append(dict(id=identifier,name=name,stage='G1e',kind='train',config=config,devices='',
                         output=cfg['train']['output'],log='runs/phase_logs/'+identifier+'.log',state='queued',
                         desired='running',split_counts=dict(train=512,val=128,test=0)))
    if register:
        if read(ROOT/'docs/G1D_RESULTS.json',{}).get('status')!='complete':
            raise ValueError('Complete G1d before G1e')
        plan=read(ROOT/'docs/OPTIMIZATION_PLAN.json');plan['jobs']=merge_jobs(plan['jobs'],jobs)
        atomic(ROOT/'docs/OPTIMIZATION_PLAN.json',plan)
    print(json.dumps({'registered':register,'jobs':[j['id'] for j in jobs]},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');setup(p.parse_args().register)
