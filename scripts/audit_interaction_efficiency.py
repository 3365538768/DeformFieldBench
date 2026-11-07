"""Measure steady-stop savings and temporal convergence of load-hold-release data."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
import time

import numpy as np

from engine_fixture import atomic,digest,wait_if_paused
from interaction_protocol import select_motion_frames
from run_cross_engine_pilot import ROOT,solve


def main():
    os.chdir(ROOT);output=ROOT/'runs/g3_interaction_efficiency';base=ROOT/'data/calibration/g3_interaction'
    original=json.loads((ROOT/'configs/g2_calibration_case.json').read_text())
    stop=dict(minimum_steps=40,window=20,maximum_speed_m_s=1e-6,relative_displacement_tolerance=.001)
    tasks=[]
    for young,force in [(12000.,.05),(48000.,.08)]:
        common={**original,'cells':[32,16,16],'young_pa':young,'force_n':force}
        for mode in ['fixed_200','steady_stop','release_dt01','release_dt005','release_dt0025']:
            case={**common}
            if mode!='fixed_200':case['stop_criteria']=stop
            if mode.startswith('release'):
                case.update(force_history=[[0.,0.],[.15,1.],[.30,1.],[.50,0.],[.60,0.]],
                            dt_seconds={'release_dt01':.01,'release_dt005':.005,'release_dt0025':.0025}[mode])
                case['steps']=round(2/case['dt_seconds'])
            tasks.append((mode,case))
    rows=[];started=time.time()
    report=dict(status='running',completed=0,total=len(tasks),rows=rows,groups=[],
        scope='Known tip traction, not simulated tool contact. BDF1 numerical dissipation is not material damping. Compare trajectories at shared physical times before using transient supervision.')
    def update():
        report.update(completed=len(rows),updated_unix=time.time(),wall_seconds=time.time()-started,
            detail=f'稳态提前结束与加载—卸载时间步审计 {len(rows)}/{len(tasks)}；不跳过未来动作。')
        atomic(ROOT/'docs/G3_INTERACTION_EFFICIENCY.json',report)
        atomic(output/'runtime.json',dict(phase=report['status'],completed=len(rows),total=len(tasks),updated_unix=time.time()))
    def worker(task):
        mode,case=task;wait_if_paused(output);key,receipt=solve('uipc',case,base,timeout=900)
        with np.load(base/key/'world/trajectory.npz',allow_pickle=False) as data:
            _,sampling=select_motion_frames(data['displacement_m'],float(data['dt_seconds']))
        return dict(id=key,path=str((base/key).relative_to(ROOT)),mode=mode,fixture='交互效率',split='calibration',
            source='G3t',material_pair=digest([case['young_pa'],case['force_n'],mode]),axis=case['axis'],
            sampling=sampling,**receipt)
    update()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(worker,t) for t in tasks]):rows.append(future.result());update()
        for young in (12000.,48000.):
            group={r['mode']:r for r in rows if r['case']['young_pa']==young}
            with np.load(ROOT/group['fixed_200']['path']/'field.npz',allow_pickle=False) as a,np.load(ROOT/group['steady_stop']['path']/'field.npz',allow_pickle=False) as b:
                steady_error=float(np.linalg.norm(a['displacement_m']-b['displacement_m'])/np.linalg.norm(a['displacement_m']))
            with np.load(ROOT/group['release_dt01']['path']/'world/trajectory.npz',allow_pickle=False) as a,np.load(ROOT/group['release_dt005']['path']/'world/trajectory.npz',allow_pickle=False) as b:
                coarse=a['displacement_m'];fine=b['displacement_m'][1::2]
                count=min(len(coarse),len(fine));coarse=coarse[:count];fine=fine[:count]
                transient_error=float(np.linalg.norm(coarse-fine)/np.linalg.norm(fine))
            with np.load(ROOT/group['release_dt005']['path']/'world/trajectory.npz',allow_pickle=False) as a,np.load(ROOT/group['release_dt0025']['path']/'world/trajectory.npz',allow_pickle=False) as b:
                coarse=a['displacement_m'];fine=b['displacement_m'][1::2]
                count=min(len(coarse),len(fine));coarse=coarse[:count];fine=fine[:count]
                finer_error=float(np.linalg.norm(coarse-fine)/np.linalg.norm(fine))
            healthy=all(r['completed'] and r['min_jacobian']>.1 and r['fixed_vertex_max_error_m']<1e-9 for r in group.values())
            report['groups'].append(dict(young_pa=young,healthy=healthy,
                steady_relative_field_error=steady_error,
                steady_steps=group['steady_stop']['simulated_steps'],reference_steps=group['fixed_200']['simulated_steps'],
                steady_step_saving_fraction=1-group['steady_stop']['simulated_steps']/group['fixed_200']['simulated_steps'],
                measured_wall_speedup=group['fixed_200']['wall_seconds']/group['steady_stop']['wall_seconds'],
                transient_dt_halving_relative_l2=transient_error,transient_dt_tolerance=.05,
                transient_finer_dt_halving_relative_l2=finer_error,selected_dynamic_case_id=group['release_dt005']['id'],
                reference_dynamic_case_id=group['release_dt0025']['id'],
                eligible_for_transient_supervision=bool(healthy and transient_error<.05 and finer_error<.05),
                note='Wall speed includes startup/I/O and concurrent GPU contention; step savings are the direct workload comparison.'))
        report['status']='complete';update()
        atomic(output/'status.json',dict(status='complete',reason='interaction_efficiency_audit_complete'))
    except Exception as exc:
        report.update(status='failed',error=str(exc));update()
        atomic(output/'status.json',dict(status='failed',reason=str(exc)));raise


if __name__=='__main__':main()
