"""Append numerical/visual quality jobs without resetting historical queue states."""
import argparse

from optimization_manager import ROOT, atomic, merge_jobs, read


def setup(render=False,efficiency=False):
    if read(ROOT/'docs/G3_FORCE_VISIBILITY.json',{}).get('status')!='complete':
        raise ValueError('Finish physical-load screening first')
    specifications=[('g3_mesh_convergence','G3c','细网格收敛与数据准入',
        'scripts/audit_mesh_convergence.py','g3_force_visibility')]
    if render:
        specifications.append(('g3_physical_render','G3r','未放大视频与可见像素光流审计',
            'scripts/render_physical_pilot.py','g3_mesh_convergence'))
    if efficiency:
        specifications.append(('g3_interaction_efficiency','G3t','动作区间采样、稳态早停与时间步检查',
            'scripts/audit_interaction_efficiency.py','g3_mesh_convergence'))
    jobs=[dict(id=identifier,name=name,stage=stage,kind='engine_pipeline',
        config='configs/g3_quality.json',devices='',output='runs/'+identifier,
        log='runs/phase_logs/'+identifier+'.log',state='queued',desired='running',
        depends_on=[dependency],args=[script]) for identifier,stage,name,script,dependency in specifications]
    plan=read(ROOT/'docs/OPTIMIZATION_PLAN.json')
    plan['jobs']=merge_jobs(plan['jobs'],jobs)
    atomic(ROOT/'docs/OPTIMIZATION_PLAN.json',plan)
    print({'registered':[j['id'] for j in jobs]})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render',action='store_true');parser.add_argument('--efficiency',action='store_true')
    args=parser.parse_args();setup(args.render,args.efficiency)
