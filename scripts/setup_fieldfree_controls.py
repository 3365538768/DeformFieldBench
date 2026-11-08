"""Register the sequential zero-field / static prior / posterior pilot."""
import copy
import json
from pathlib import Path
from engine_fixture import atomic
from prepare_fieldfree_priors import OUTPUT

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'docs/OPTIMIZATION_PLAN.json').read_text())
    if any(j['id'].startswith('nf_') for j in plan['jobs']):raise ValueError('Field-free pilot already registered')
    base=json.loads((ROOT/'configs/single_video_base.json').read_text())
    protocol=dict(status='registered_before_training',seed=17,train_samples=512,validation_samples=128,test_samples=0,
        hypothesis='Direct temporal evidence with first-frame OOF static priors and a mixture posterior can identify materials without field heads or field feedback.',
        controls=['N0_video_point','N1_prior_point','N2_prior_posterior'],
        budget='same frozen DINOv3, width256/layers2, 20 epochs max, batch4 accumulation2, workers4, patience4/min6',
        static_prior='First-valid-frame global DINO feature ridge; five object folds for train; full training objects for val. No external semantic model yet.',
        matching='Identical modules/initialization; all read identical feature and prior inputs, N0 masks prior internally. N0/N1 NLL weight zero; N2 exact masked mixture NLL weight 0.1.',
        teacher='Solver-provenance/multi-solution interface implemented. Original video distillation is not registered until geometry/loading and response admission pass.',
        independent_teacher_pilot='Known-force small-strain linear fixture with projected tracks; software/physical-pipeline check, not the video dataset or real-world accuracy.',
        scope='No field targets, predicted fields, field auxiliary loss or feedback branch. No test, backbone unfreezing or automatic full-scale promotion.')
    atomic(ROOT/'docs/NF_PROTOCOL.json',protocol)
    def pipeline(identifier,name,script,report,depends=None):
        plan['jobs'].append(dict(id=identifier,name=name,stage='NF',kind='engine_pipeline',args=['scripts/'+script],
            devices='',output='runs/'+identifier,log='runs/phase_logs/'+identifier+'.log',state='queued',desired='running',
            resources={'device':'cpu'},progress_report='docs/'+report,depends_on=depends or []))
    pipeline('nf_static_prepare','零场路线：首帧物体OOF静态先验','prepare_fieldfree_priors.py','NF_STATIC_PREPARE.json')
    previous='nf_static_prepare'
    for label,prior,nll in [('N0_video_point',False,0.),('N1_prior_point',True,0.),('N2_prior_posterior',True,.1)]:
        identifier='nf_'+label+'_seed17';cfg=copy.deepcopy(base)
        cfg['model'].update(architecture='direct_video',field_mode='none',feedback='none',use_prior=prior,posterior_components=4)
        cfg['data'].update(view_policy='single',single_view_layout='compact',motion_targets_dir='',material_prior_file=OUTPUT,
                           rgb_cache_dir='',view_dropout=0.,prepared_cache_gib=0.)
        cfg['train'].update(output='runs/'+identifier,field_weight=0.,field_epochs=0,nll_weight=nll,distillation_weight=0.,
                            resume='',initialize='',initialize_rgb_baseline='',seed=17)
        filename='configs/runtime/'+identifier+'.json';atomic(ROOT/filename,cfg)
        plan['jobs'].append(dict(id=identifier,name=label,stage='NF',kind='train',queue_group='fieldfree_probe',
            control_label=label,config=filename,source_config=filename,devices='',output=cfg['train']['output'],
            log='runs/phase_logs/'+identifier+'.log',state='queued',desired='running',resources={'minimum_free_mib':12288},
            readiness_file='docs/NEW_A800_READINESS.json',split_counts=dict(train=512,val=128,test=0),depends_on=[previous]))
        previous=identifier
    pipeline('nf_posterior_audit','零场路线：匹配结果与后验诊断','audit_fieldfree_controls.py','NF_AUDIT.json',[previous])
    pipeline('nf_teacher_fixture','离线教师：已知夹具蒸馏核验','probe_fieldfree_teacher.py','NF_TEACHER_FIXTURE.json')
    atomic(ROOT/'docs/OPTIMIZATION_PLAN.json',plan)
    print('Sequential N0 -> N1 -> N2 plus independent controlled teacher fixture registered')


if __name__=='__main__':main()
