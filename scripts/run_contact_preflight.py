"""Bounded geometry/material/contact probes; fail closed before dataset scale-up."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from engine_fixture import atomic, digest, wait_if_paused, run_controlled_solver
from irregular_geometry import SHAPES, make_solid, mesh_health
from render_contact_preflight import render

ROOT=Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT);output=ROOT/'runs/g3_contact_preflight';output.mkdir(parents=True,exist_ok=True)
    base=ROOT/'data/calibration/g3_contact_preflight'
    report=dict(status='running',completed=0,total=8,rows=[],clips=[],admitted_for_training=False,
        scope='Procedural metric solids, bonded material coefficients and physical 3D punch. Not GS reconstruction; prescribed motion, no calibrated force GT; convergence/energy/clearance not yet admitted.')
    sources=['engine_fixture.py','irregular_geometry.py','solve_contact_preflight.py']
    fingerprint=hashlib.sha256(b''.join((ROOT/'scripts'/s).read_bytes() for s in sources)).hexdigest()
    def update(**kw):
        report.update(kw,updated_unix=time.time());atomic(ROOT/'docs/G3_CONTACT_PREFLIGHT.json',report)
        atomic(output/'runtime.json',dict(phase=report['status'],completed=report['completed'],total=report['total'],
            detail=f"不规则实体三维接触预检 {report['completed']}/8；未准入训练。",updated_unix=time.time()))
    update()
    try:
        for shape in SHAPES:
            for layout in ('homogeneous','core_shell'):
                wait_if_paused(output)
                case=dict(shape=shape,cells=[12,6,6],layout=layout,young_pa=[12000.,48000.],poisson=.3,
                    dt_seconds=.005,duration_s=.6,indentation_m=.004,initial_gap_m=.0007,
                    barrier_distance_m=.0001,friction=0.)
                identifier='uipc_'+digest({'case':case,'source':fingerprint})[:20]
                folder=base/identifier;folder.mkdir(parents=True,exist_ok=True)
                cp=folder/'case.json';atomic(cp,case);rp=folder/'result.json'
                if not rp.exists():
                    with (folder/'solver.log').open('w') as f:
                        run_controlled_solver([str(ROOT/'.engine-envs/uipc/bin/python'),'scripts/solve_contact_preflight.py',
                            '--case',str(cp),'--output',str(folder),'--control-output',str(output)],
                            cwd=ROOT,log=f,control_output=output,active_timeout_seconds=900)
                row=json.loads(rp.read_text())
                if row['case']!=case or row['source_sha256']!=hashlib.sha256((ROOT/'scripts/solve_contact_preflight.py').read_bytes()).hexdigest():
                    raise ValueError('Contact probe provenance mismatch')
                if not row['completed'] or row['min_jacobian']<=0 or row['fixed_vertex_max_error_m']>=1e-9 or row['max_tool_tracking_error_m']>=1e-4:
                    raise ValueError('Contact probe failed numerical screening')
                report['rows'].append(dict(id=identifier,path=str(folder.relative_to(ROOT)),**row))
                report['clips'].append(render(ROOT,folder,row));update(completed=len(report['rows']))
        update(status='complete',next_gate='mesh_time_step_contact_force_energy_and_identifiability_audit',
            detail='8/8 接触管线预检完成；仍未准入训练，不自动扩大到 64 求解或启动模型训练。')
        atomic(output/'status.json',dict(status='complete',reason='contact_pipeline_probed_quality_admission_pending'))
    except Exception as exc:
        update(status='failed',error_type=type(exc).__name__)
        atomic(output/'status.json',dict(status='failed',reason=type(exc).__name__));raise


if __name__=='__main__':main()
