"""Register the calibrated engine pilot without changing existing job states."""
import argparse
from optimization_manager import ROOT, atomic, merge_jobs, read


def setup(register=False):
    job=dict(id='g2_cross_engine_pilot',name='双引擎 SI 场矩阵与共享适配控制',stage='G2',
        kind='engine_pipeline',config='configs/g2_calibration_case.json',devices='',
        output='runs/g2_cross_engine_pilot',log='runs/phase_logs/g2_cross_engine_pilot.log',
        state='queued',desired='running',args=['scripts/run_cross_engine_pilot.py','--workers','2'])
    if register:
        if read(ROOT/'docs/G1F_RESULTS.json',{}).get('status')!='complete':
            raise ValueError('Complete matched DINOv3 evidence before G2')
        replay=dict(id='g2_engine_replay',name='原生参数回放与经典逆 FEM 对照',stage='G2r',
            kind='engine_pipeline',config='configs/g2_calibration_case.json',devices='',
            output='runs/g2_engine_replay',log='runs/phase_logs/g2_engine_replay.log',state='queued',desired='running',
            depends_on=['g2_cross_engine_pilot'],args=['scripts/evaluate_engine_replay.py'])
        visibility=dict(id='g3_force_visibility',name='载荷幅值、网格与视频可观测性审计',stage='G3v',
            kind='engine_pipeline',config='configs/g2_calibration_case.json',devices='',
            output='runs/g3_force_visibility',log='runs/phase_logs/g3_force_visibility.log',state='queued',desired='running',
            depends_on=['g2_engine_replay'],args=['scripts/audit_force_visibility.py'])
        plan=read(ROOT/'docs/OPTIMIZATION_PLAN.json');plan['jobs']=merge_jobs(plan['jobs'],[job,replay,visibility])
        atomic(ROOT/'docs/OPTIMIZATION_PLAN.json',plan)
    print({'registered':register,'job':job['id']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');setup(p.parse_args().register)
