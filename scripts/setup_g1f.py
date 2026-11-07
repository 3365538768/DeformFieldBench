"""Register the authorized DINOv3 backbone experiment with explicit prerequisites."""
import argparse
import copy
import json
from optimization_manager import ROOT, atomic, merge_jobs, read


def setup(register=False):
    receipt=read(ROOT/'docs/DINOV3_PREPARATION.json',{})
    if receipt.get('status')!='complete' or not receipt.get('strict_load'):
        raise ValueError('Verified, authorized DINOv3 weights are required')
    jobs=[]; cache_id='g1f_cache'; baseline='g1f_B_rgb_only_seed17'
    cases=[('B_rgb_only','g1b_B_rgb_repeat_seed17'),
           ('D_local_fields','g1e_D_local_fields_seed17'),
           ('G_local_features','g1e_G_local_features_seed17'),
           ('O_local_oracle','g1e_O_local_oracle_seed17'),
           ('S_local_shuffled','g1e_S_local_shuffled_seed17')]
    for name,source in cases:
        cfg=copy.deepcopy(read(ROOT/f'configs/{source}.json'))
        cfg['model'].update(backbone='dinov3',repo='external/dinov3',hub_name='dinov3_vitl16',
                            weights=receipt['weights'],frame_chunk=16)
        cfg['data']['prepared_dir']='data/prepared/g1f_dinov3l16_local'
        identifier=f'g1f_{name}_seed17';cfg['train']['output']='runs/'+identifier
        cfg['train']['resume']='';cfg['train']['initialize']=''
        cfg['train']['initialize_rgb_baseline']='' if name=='B_rgb_only' else f'runs/{baseline}/best_material.pt'
        config='configs/'+identifier+'.json';existing=read(ROOT/config)
        if existing is not None and existing!=cfg:raise ValueError('Existing config differs: '+config)
        atomic(ROOT/config,cfg)
        jobs.append(dict(id=identifier,name=name,stage='G1f',kind='train',config=config,devices='',
            output=cfg['train']['output'],log='runs/phase_logs/'+identifier+'.log',state='queued',desired='running',
            depends_on=['g1f_cache_local'] if name=='B_rgb_only' else ['g1f_cache_local',baseline],
            split_counts=dict(train=512,val=128,test=0)))
    cache_config=copy.deepcopy(read(ROOT/f'configs/{baseline}.json'))
    cache_config['data']['prepared_dir']='data/prepared/g1f_dinov3l16'
    atomic(ROOT/'configs/g1f_cache.json',cache_config)
    jobs.insert(0,dict(id=cache_id,name='DINOv3 RGB feature cache',stage='G1f',kind='cache',
        config='configs/g1f_cache.json',devices='',output='runs/g1f_cache',log='runs/phase_logs/g1f_cache.log',
        state='queued',desired='running',status_file='docs/G1F_CACHE_STATUS.json',
        args=['scripts/prepare_feature_cache.py','--config','configs/g1f_cache.json',
              '--control-output','runs/g1f_cache','--status-output','docs/G1F_CACHE_STATUS.json',
              '--prefetch-workers','4']))
    jobs.insert(1,dict(id='g1f_cache_local',name='DINOv3 cache SHA256 local staging',stage='G1f',kind='cache_stage',
        config='configs/g1f_cache.json',devices='',output='runs/g1f_cache_local',log='runs/phase_logs/g1f_cache_local.log',
        state='queued',desired='running',depends_on=[cache_id],status_file='docs/G1F_LOCAL_CACHE.json',
        args=['scripts/stage_feature_cache.py','--source','data/prepared/g1f_dinov3l16',
              '--destination','/tmp/linrui_fieldbridge/g1f_dinov3l16','--link','data/prepared/g1f_dinov3l16_local',
              '--receipt','docs/G1F_LOCAL_CACHE.json']))
    if register:
        if read(ROOT/'docs/G1E_RESULTS.json',{}).get('status')!='complete':raise ValueError('Complete G1e before G1f')
        plan=read(ROOT/'docs/OPTIMIZATION_PLAN.json');plan['jobs']=merge_jobs(plan['jobs'],jobs)
        atomic(ROOT/'docs/OPTIMIZATION_PLAN.json',plan)
    print(json.dumps({'registered':register,'jobs':[j['id'] for j in jobs]},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');setup(p.parse_args().register)
