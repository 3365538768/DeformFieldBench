"""Determine whether actual SI deformation would be visible before RGB rendering.

Ideal orthographic pixel-motion bound, NOT a rendered-video accuracy claim.
UIPC only: linear Warp equivalence is not imposed at finite strain.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import numpy as np

from engine_fixture import atomic, wait_if_paused
from run_cross_engine_pilot import ROOT, solve


def main():
    os.chdir(ROOT);output=ROOT/'runs/g3_force_visibility';base=ROOT/'data/calibration/g3_visibility'
    original=json.loads((ROOT/'configs/g2_calibration_case.json').read_text())
    cases=[{**original,'young_pa':young,'force_n':force,'cells':cells}
        for young in (12000.,48000.) for force in (.0002,.002,.01,.02,.05)
        for cells in ([8,4,4],[16,8,8])]
    results=[];started=time.time()
    def progress(status):
        report=dict(status=status,completed=len(results),total=len(cases),results=results,
            updated_unix=time.time(),detail=f'实际 SNK1 载荷/网格与视频可观测性审计 {len(results)}/{len(cases)}；不强制大形变与线性引擎等价。',
            scope='UIPC known-traction block only; ideal orthographic 224px / 0.08m FoV, XY view, no occlusion/appearance/noise. Screening bound, not measured rendered optical flow.')
        if status=='complete':
            pairs={}
            for row in results:pairs.setdefault((row['young_pa'],row['force_n']),{})[row['cells'][0]]=row
            report['candidates']=[]
            for pair in pairs.values():
                a,b=pair[8],pair[16]
                delta=abs(a['max_final_displacement_m']/b['max_final_displacement_m']-1)
                report['candidates'].append(dict(young_pa=b['young_pa'],force_n=b['force_n'],
                    fine_ideal_max_motion_px=b['ideal_max_motion_px'],mesh_relative_displacement_change=delta,
                    usable_for_next_refinement=bool(b['healthy'] and 2<=b['ideal_max_motion_px']<=30 and delta<.15),
                    note='A positive candidate still needs finer mesh convergence, realistic rendering, contact and boundary/scale variation.'))
            report['wall_seconds']=time.time()-started
        atomic(ROOT/'docs/G3_FORCE_VISIBILITY.json',report)
        atomic(output/'runtime.json',dict(phase=status,completed=len(results),total=len(cases),updated_unix=time.time()))
    def worker(case):
        wait_if_paused(output);key,r=solve('uipc',case,base)
        displacement=np.load(base/key/'field.npz',allow_pickle=False)['displacement_m']
        magnitude=float(np.linalg.norm(displacement,axis=1).max())
        pixels=float(np.linalg.norm(displacement[:,:2],axis=1).max()*224/.08)
        healthy=bool(r['completed'] and r['min_jacobian']>.1 and r['fixed_vertex_max_error_m']<1e-9
            and r['final_max_speed_m_s']<1e-6 and r['last20_displacement_change_m']/max(magnitude,1e-12)<.001)
        return dict(id=key,young_pa=case['young_pa'],poisson=case['poisson'],force_n=case['force_n'],cells=case['cells'],
            max_final_displacement_m=magnitude,ideal_max_motion_px=pixels,healthy=healthy,
            min_jacobian=r['min_jacobian'],max_displacement_gradient=r['max_displacement_gradient'],
            final_max_speed_m_s=r['final_max_speed_m_s'],wall_seconds=r['wall_seconds'])
    progress('running')
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(worker,c) for c in cases]):results.append(future.result());progress('running')
        progress('complete');atomic(output/'status.json',dict(status='complete',reason='visibility_audit_complete'))
    except Exception as exc:
        atomic(output/'status.json',dict(status='failed',reason=str(exc)));raise


if __name__=='__main__':main()
