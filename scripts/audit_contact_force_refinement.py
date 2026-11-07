"""Finite reaction-force mesh and friction probes after the SI force audit.

Keeps data admission disabled, including when these local diagnostics pass.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from engine_fixture import atomic, digest, wait_if_paused, run_controlled_solver

ROOT=Path(__file__).resolve().parents[1]


def read_case(row):
    with np.load(ROOT/row['path']/'contact_diagnostics.npz',allow_pickle=False) as z:
        return z['tool_contact_force_n']


def main():
    os.chdir(ROOT);output=ROOT/'runs/g3_contact_force_refinement';output.mkdir(parents=True,exist_ok=True)
    prior=json.loads((ROOT/'docs/G3_CONTACT_FORCE_AUDIT.json').read_text())
    mesh=json.loads((ROOT/'docs/G3_CONTACT_NUMERICAL_AUDIT.json').read_text())
    if not prior.get('diagnostic_screen_passed') or not mesh.get('repair_screen_passed'):
        raise ValueError('Prior force and repaired mesh diagnostics must pass')
    baseline=next(x for x in prior['rows'] if x['case']['young_pa'][0]==12000. and x['case']['dt_seconds']==.005)
    source=hashlib.sha256(b''.join((ROOT/'scripts'/name).read_bytes() for name in
        ['engine_fixture.py','irregular_geometry.py','solve_contact_preflight.py','contact_diagnostics.py'])).hexdigest()
    report=dict(status='running',completed=0,total=4,rows=[],comparisons=[],admitted_for_training=False,
                reaction_force_gt_available=False,baseline_path=baseline['path'])
    def update(**kw):
        report.update(kw,updated_unix=time.time());atomic(ROOT/'docs/G3_CONTACT_FORCE_REFINEMENT.json',report)
        atomic(output/'runtime.json',dict(phase=report['status'],completed=report['completed'],total=4,
            detail='反力连续两级网格与摩擦敏感性；仍未准入训练',updated_unix=time.time()))
    update();meshes=[baseline]
    try:
        variations=[dict(cells=[36,18,18]),dict(cells=[48,24,24]),dict(friction=.15),dict(friction=.4)]
        for variation in variations:
            wait_if_paused(output);case=copy.deepcopy(baseline['case']);case.update(variation)
            folder=ROOT/'data/calibration/g3_contact_force_refinement'/digest(dict(case=case,source=source))[:20]
            folder.mkdir(parents=True,exist_ok=True);atomic(folder/'case.json',case)
            if not (folder/'result.json').exists():
                with (folder/'solver.log').open('w') as log:
                    run_controlled_solver([str(ROOT/'.engine-envs/uipc/bin/python'),'scripts/solve_contact_preflight.py',
                        '--case',str(folder/'case.json'),'--output',str(folder),'--control-output',str(output)],
                        cwd=ROOT,log=log,control_output=output,active_timeout_seconds=1800)
            row=json.loads((folder/'result.json').read_text());row['path']=str(folder.relative_to(ROOT))
            if row['case']!=case or not row['completed'] or row['min_jacobian']<=0:raise ValueError('Force refinement numerical failure')
            report['rows'].append(row)
            if 'cells' in variation:
                previous=meshes[-1];a,b=read_case(previous),read_case(row)
                relative=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-15))
                report['comparisons'].append(dict(kind='tool_force_mesh',cells_coarse=previous['case']['cells'],
                    cells_fine=case['cells'],relative_l2=relative,threshold=.03,passed=relative<.03))
                meshes.append(row)
            else:
                a,b=read_case(baseline),read_case(row)
                report['comparisons'].append(dict(kind='friction_sensitivity',friction=case['friction'],
                    tool_force_change_relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-15)),
                    scope='Sensitivity only; no expected ground-truth coefficient or direction inferred'))
            update(completed=len(report['rows']))
        passed=all(x['passed'] for x in report['comparisons'] if x['kind']=='tool_force_mesh')
        update(status='complete',force_mesh_screen_passed=passed,
            next_gate='Full work/actuator accounting and minimum/continuous clearance; per-geometry/material/action admission before scale-up')
        atomic(output/'status.json',dict(status='complete',reason='force_mesh_probe_passed_admission_pending' if passed else 'force_mesh_probe_failed_scale_up_blocked'))
    except Exception as exc:
        update(status='failed',error_type=type(exc).__name__)
        atomic(output/'status.json',dict(status='failed',reason=type(exc).__name__));raise


if __name__=='__main__':main()
