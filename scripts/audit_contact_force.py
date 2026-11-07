"""Four bounded SI force checks; exports remain diagnostic, not training GT."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from engine_fixture import atomic, digest, wait_if_paused, run_controlled_solver

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    output = ROOT/'runs/g3_contact_force_audit'; output.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', completed=0, total=4, rows=[], comparisons=[],
                  admitted_for_training=False, reaction_force_gt_available=False)
    source_files = ['engine_fixture.py','irregular_geometry.py','solve_contact_preflight.py','contact_diagnostics.py']
    source = hashlib.sha256(b''.join((ROOT/'scripts'/s).read_bytes() for s in source_files)).hexdigest()
    def update(**kw):
        report.update(kw, updated_unix=time.time())
        atomic(ROOT/'docs/G3_CONTACT_FORCE_AUDIT.json', report)
        atomic(output/'runtime.json', dict(phase=report['status'], completed=report['completed'], total=4,
               detail='接触反力单位、时间步和刚度比例诊断；非训练 GT', updated_unix=time.time()))
    update()
    samples = {}
    try:
        for young in [12000., 48000.]:
            for dt in [.005, .0025]:
                wait_if_paused(output)
                case = dict(shape='waisted',cells=[24,12,12],layout='homogeneous',young_pa=[young,young],
                    poisson=.3,dt_seconds=dt,duration_s=2.4,motion_time_scale=4.,indentation_m=.004,
                    initial_gap_m=.0007,barrier_distance_m=.0001,friction=0.,export_contact_diagnostics=True)
                folder = ROOT/'data/calibration/g3_contact_force_audit'/digest(dict(case=case,source=source))[:20]
                folder.mkdir(parents=True,exist_ok=True); atomic(folder/'case.json',case)
                if not (folder/'result.json').exists():
                    with (folder/'solver.log').open('w') as log:
                        run_controlled_solver([str(ROOT/'.engine-envs/uipc/bin/python'),'scripts/solve_contact_preflight.py',
                            '--case',str(folder/'case.json'),'--output',str(folder),'--control-output',str(output)],
                            cwd=ROOT,log=log,control_output=output,active_timeout_seconds=1800)
                row = json.loads((folder/'result.json').read_text())
                if row['case']!=case or not row['completed'] or row['min_jacobian']<=0:
                    raise ValueError('Force case numerical screening failed')
                with np.load(folder/'trajectory.npz',allow_pickle=False) as z: displacement=z['displacement_m']
                with np.load(folder/'contact_diagnostics.npz',allow_pickle=False) as z:
                    diagnostics = {k:z[k] for k in z.files}
                force = diagnostics['tool_contact_force_n']
                if not np.isfinite(force).all():raise ValueError('Nonfinite contact force')
                times = (np.arange(len(force))+1)*dt
                hold = (times>=.9)&(times<=1.)
                active = np.linalg.norm(force,axis=1)>.05*np.linalg.norm(force,axis=1).max()
                row.update(path=str(folder.relative_to(ROOT)),hold_mean_tool_force_y_n=float(force[hold,1].mean()),
                    active_force_balance_p95=float(np.quantile(diagnostics['free_node_relative_force_residual'][active],.95)),
                    final_input_endpoint_work_j=float(diagnostics['input_work_endpoint_j'][-1]),
                    final_stored_energy_j=float(diagnostics['energy_j'][-1].sum()))
                report['rows'].append(row); samples[(young,dt)]=dict(force=force,displacement=displacement,row=row)
                update(completed=len(report['rows']))
        for young in [12000.,48000.]:
            a,b=samples[(young,.005)],samples[(young,.0025)]
            delta=float(np.linalg.norm(a['force']-b['force'][1::2])/max(np.linalg.norm(b['force'][1::2]),1e-15))
            report['comparisons'].append(dict(kind='dt_halving_tool_force',young_pa=young,relative_l2=delta,
                                             threshold=.05,passed=delta<.05))
        for dt in [.005,.0025]:
            a,b=samples[(12000.,dt)],samples[(48000.,dt)]
            ratio=b['row']['hold_mean_tool_force_y_n']/a['row']['hold_mean_tool_force_y_n']
            times=(np.arange(len(a['force']))+1)*dt;hold=(times>=.9)&(times<=1.)
            shape_delta=float(np.linalg.norm(a['displacement'][hold]-b['displacement'][hold])/max(np.linalg.norm(a['displacement'][hold]),1e-15))
            report['comparisons'].append(dict(kind='same_motion_fourfold_stiffness',dt_seconds=dt,
                hold_tool_force_ratio=ratio,expected_ratio=4.,ratio_relative_error=abs(ratio/4-1),
                hold_shape_relative_l2=shape_delta,threshold=.05,passed=abs(ratio/4-1)<.05))
        diagnostic_pass = all(r['passed'] for r in report['comparisons']) and all(
            r['contact_diagnostics']['max_action_reaction_relative_error']<1e-8 and r['active_force_balance_p95']<.05
            for r in report['rows'])
        update(status='complete',diagnostic_screen_passed=diagnostic_pass,
            next_gate='Complete surface clearance, full actuator/energy accounting, friction and force mesh convergence; no training/scale-up yet',
            scope='Endpoint contact work and solid kinetic/elastic plus barrier energy are diagnostics only. Prescribed motion carries no measured force input in RGB-only evaluation.')
        atomic(output/'status.json',dict(status='complete',reason='force_diagnostics_passed_admission_pending' if diagnostic_pass else 'force_diagnostics_failed_scale_up_blocked'))
    except Exception as exc:
        update(status='failed',error_type=type(exc).__name__)
        atomic(output/'status.json',dict(status='failed',reason=type(exc).__name__));raise


if __name__=='__main__':main()
