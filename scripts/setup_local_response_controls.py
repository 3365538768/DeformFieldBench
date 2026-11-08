"""Pre-register a compact single-video, global/local evidence ablation."""
import copy
import json
from pathlib import Path
from engine_fixture import atomic

ROOT=Path(__file__).resolve().parents[1]
CONTROLS=[('B_rgb','none',0.,'none'),('C_aux','none',.1,'none'),
    ('D_raw','fields',.1,'none'),('G_raw','features',.1,'none'),
    ('D_decomposed','fields',.1,'translation'),('G_decomposed','features',.1,'translation')]


def main():
    plan=json.loads((ROOT/'docs/OPTIMIZATION_PLAN.json').read_text())
    base=json.loads((ROOT/'configs/single_video_base.json').read_text())
    protocol=dict(status='registered_before_training',sampling='original 512/128 diagnostic; no selection by D3 winner',
        seed=17,epochs=20,patience=4,primary='common-codec material MAE on fixed single validation camera',
        controls=[x[0] for x in CONTROLS],
        hypothesis='Separate predicted global translation and confidence-weighted relative displacement; preserve both in shared-weight temporal encoders.',
        capacity='No additional parameters. D/G have raw and decomposed arms. B/C are shared: outputs and material+field gradients verified exactly equal without feedback.',
        information='No target field enters inference. DIS remains pixel correspondence, residual is not strain, confidence is not contact force.',
        efficiency='All six use compact layout; compare representation only within M3. D3 masked-layout times are a separate efficiency comparison.',
        gates='At least 2% direction gain over all B/C/G controls; paired object intervals and independent seeds required before a final claim.',
        dependencies=['d3_single_video_audit','i1_identifiability_solve64'])
    atomic(ROOT/'docs/M3_LOCAL_RESPONSE_PROTOCOL.json',protocol)
    for label,feedback,weight,decomposition in CONTROLS:
        identifier='m3_'+label+'_seed17'
        if any(j['id']==identifier for j in plan['jobs']):raise ValueError('M3 job already registered')
        cfg=copy.deepcopy(base);cfg['data'].update(view_policy='single',single_view_layout='compact',view_dropout=0.,prepared_cache_gib=0.)
        cfg['model'].update(feedback=feedback,bridge_decomposition=decomposition,detach_feedback=False)
        cfg['train'].update(output='runs/'+identifier,field_weight=weight,seed=17,epochs=20,resume='',initialize='',
            initialize_rgb_baseline='',field_epochs=0,early_stop_patience=4,early_stop_min_joint_epochs=6)
        filename='configs/runtime/'+identifier+'.json';atomic(ROOT/filename,cfg)
        plan['jobs'].append(dict(id=identifier,name=label,stage='M3',kind='train',queue_group='local_response_probe',
            config=filename,source_config=filename,devices='',output=cfg['train']['output'],log='runs/phase_logs/'+identifier+'.log',
            state='queued',desired='running',resources={'minimum_free_mib':24576},readiness_file='docs/NEW_A800_READINESS.json',
            split_counts=dict(train=512,val=128,test=0),depends_on=protocol['dependencies']))
    plan['jobs'].append(dict(id='m3_local_response_audit',name='局部响应表示的物体配对反馈',stage='M3a',kind='engine_pipeline',
        args=['scripts/audit_local_response_controls.py'],devices='',output='runs/m3_local_response_audit',
        log='runs/phase_logs/m3_local_response_audit.log',state='queued',desired='running',resources={'device':'cpu'},
        depends_on=['m3_'+x[0]+'_seed17' for x in CONTROLS],progress_report='docs/M3_LOCAL_RESPONSE_AUDIT.json'))
    atomic(ROOT/'docs/OPTIMIZATION_PLAN.json',plan)
    print('Six M3 controls registered with D3/I1 dependencies; no full-scale promotion.')


if __name__=='__main__':main()
