"""Bounded contact-regularization experiment; preserves failed baseline evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from engine_fixture import atomic, digest, wait_if_paused, run_controlled_solver

ROOT=Path(__file__).resolve().parents[1]


def force_statistics(row):
    with np.load(ROOT/row['path']/'contact_diagnostics.npz',allow_pickle=False) as z:
        force=z['tool_contact_force_n'];balance=z['free_node_relative_force_residual']
        time_s=(np.arange(len(force))+1)*row['case']['dt_seconds']
        active=np.linalg.norm(force,axis=1)>.05*np.linalg.norm(force,axis=1).max()
        hold=(time_s>=.9)&(time_s<=1.)
        return force,dict(active_balance_p95=float(np.quantile(balance[active],.95)),
            active_frames=int(active.sum()),hold_mean_force_y_n=float(force[hold,1].mean()),
            all_frame_balance_p95=float(np.quantile(balance,.95)))


def main():
    os.chdir(ROOT);output=ROOT/'runs/g3_contact_regularization';output.mkdir(parents=True,exist_ok=True)
    prior=json.loads((ROOT/'docs/G3_CONTACT_FORCE_AUDIT.json').read_text())
    refinement=json.loads((ROOT/'docs/G3_CONTACT_FORCE_REFINEMENT.json').read_text())
    base=next(x for x in prior['rows'] if x['case']['young_pa'][0]==12000. and x['case']['dt_seconds']==.005)
    originals=[base]+[x for x in refinement['rows'] if x['case']['friction']==0.]
    source=hashlib.sha256(b''.join((ROOT/'scripts'/n).read_bytes() for n in
        ['solve_contact_preflight.py','contact_diagnostics.py','irregular_geometry.py','engine_fixture.py'])).hexdigest()
    report=dict(status='running',completed=0,total=3,originals=[],rows=[],comparisons=[],
        admitted_for_training=False,reaction_force_gt_available=False,
        hypothesis='Fixed barrier distance may change force convergence under mesh refinement; halve d_hat with all else fixed.',
        original_force_mesh_screen_passed=refinement['force_mesh_screen_passed'])
    for row in originals:
        _,stats=force_statistics(row);report['originals'].append(dict(path=row['path'],cells=row['case']['cells'],**stats))
    def update(**kw):
        report.update(kw,updated_unix=time.time(),detail='减半接触屏障距离的连续两级反力网格诊断；未准入训练。')
        atomic(ROOT/'docs/G3_CONTACT_REGULARIZATION.json',report)
        atomic(output/'runtime.json',dict(phase=report['status'],completed=report['completed'],total=3,updated_unix=time.time()))
    try:
        update();forces=[]
        for cells in [[24,12,12],[36,18,18],[48,24,24]]:
            wait_if_paused(output);case=copy.deepcopy(base['case']);case.update(cells=cells,barrier_distance_m=.00005)
            folder=ROOT/'data/calibration/g3_contact_regularization'/digest(dict(case=case,source=source))[:20]
            folder.mkdir(parents=True,exist_ok=True);atomic(folder/'case.json',case)
            if not (folder/'result.json').exists():
                with (folder/'solver.log').open('w') as log:
                    run_controlled_solver([str(ROOT/'.engine-envs/uipc/bin/python'),'scripts/solve_contact_preflight.py',
                        '--case',str(folder/'case.json'),'--output',str(folder),'--control-output',str(output)],
                        cwd=ROOT,log=log,control_output=output,active_timeout_seconds=2400)
            row=json.loads((folder/'result.json').read_text());row['path']=str(folder.relative_to(ROOT))
            if row['case']!=case or not row['completed'] or row['min_jacobian']<=0:raise ValueError('Invalid regularization case')
            force,stats=force_statistics(row);row['phase_statistics']=stats;forces.append(force)
            if len(forces)>1:
                difference=float(np.linalg.norm(forces[-2]-force)/max(np.linalg.norm(force),1e-15))
                report['comparisons'].append(dict(cells_coarse=report['rows'][-1]['case']['cells'],cells_fine=cells,
                    relative_l2=difference,threshold=.03,passed=difference<.03))
            report['rows'].append(row);update(completed=len(report['rows']))
        passed=all(c['passed'] for c in report['comparisons'])
        update(status='complete',force_mesh_screen_passed=passed,
            route='time_step_and_full_work_audit' if passed else 'contact_discretization_or_alternative_solver_required',
            note='Active-only balance avoids division by almost-zero contact in inactive frames; failed mesh thresholds remain unchanged. No force GT admission.')
        atomic(output/'status.json',dict(status='complete',reason='bounded_contact_regularization_diagnostic_complete'))
    except Exception as error:
        update(status='failed',error_type=type(error).__name__);atomic(output/'status.json',dict(status='failed',reason=str(error)));raise


if __name__=='__main__':main()
