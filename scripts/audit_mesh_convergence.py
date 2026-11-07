"""Refine the five visible UIPC candidates before authorizing any RGB data scale-up."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time

import numpy as np

from engine_fixture import atomic, digest, wait_if_paused
from mesh_quality import accepted_refinements, compare_fields
from run_cross_engine_pilot import ROOT, SOLVER_SOURCE_SHA, solve


def numerical_health(receipt, magnitude):
    case = receipt['case']
    expected = np.eye(3)[case['axis']]*case['force_n']
    return bool(receipt['completed'] and receipt['fixed_vertex_max_error_m'] < 1e-9
        and receipt['min_jacobian'] > .1 and receipt['final_max_speed_m_s'] < 1e-6
        and receipt['last20_displacement_change_m']/max(magnitude, 1e-12) < .001
        and np.linalg.norm(np.asarray(receipt['assembled_resultant_n'])-expected)/case['force_n'] < 1e-5)


def main():
    os.chdir(ROOT)
    output = ROOT/'runs/g3_mesh_convergence'
    base = ROOT/'data/calibration/g3_convergence'
    cfg = json.loads((ROOT/'configs/g3_quality.json').read_text())
    source = json.loads((ROOT/'docs/G3_FORCE_VISIBILITY.json').read_text())
    candidates = [c for c in source['candidates'] if c['usable_for_next_refinement']]
    original = json.loads((ROOT/'configs/g2_calibration_case.json').read_text())
    rows = []; started = time.time()
    report = dict(status='running', completed=0, total=len(candidates)*len(cfg['mesh_levels']),
        protocol=cfg, protocol_sha256=digest(cfg), solver_source_sha256=SOLVER_SOURCE_SHA,
        rows=rows, groups=[], scope='Single-engine UIPC known-load blocks; convergence is a local discretization check, not continuum ground truth or RGB success.')
    def update():
        report.update(completed=len(rows), updated_unix=time.time(), wall_seconds=time.time()-started,
            detail=f'细网格实际求解 {len(rows)}/{report["total"]}；连续两级全场及端面响应变化均 <3% 才允许渲染。')
        atomic(ROOT/'docs/G3_MESH_CONVERGENCE.json', report)
        atomic(output/'runtime.json', dict(phase=report['status'], completed=len(rows), total=report['total'], updated_unix=time.time()))
    def worker(task):
        candidate, level = task
        wait_if_paused(output)
        case = {**original, 'young_pa':candidate['young_pa'], 'force_n':candidate['force_n'],
                'cells':[level, level//2, level//2]}
        key, receipt = solve('uipc', case, base, timeout=cfg['solver_timeout_seconds'])
        with np.load(base/key/'field.npz', allow_pickle=False) as field:
            magnitude = float(np.linalg.norm(field['displacement_m'], axis=1).max())
            pixels = float(np.linalg.norm(field['displacement_m'][:, :2], axis=1).max()*cfg['image_size']/cfg['field_of_view_m'])
        return dict(id=key, path=str((base/key).relative_to(ROOT)), fixture='收敛审计', split='calibration',
            source='G3c', material_pair=digest([case['young_pa'],case['force_n']]), axis=case['axis'],
            healthy=numerical_health(receipt,magnitude), ideal_max_motion_px=pixels, **receipt)
    update()
    try:
        with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
            futures = [pool.submit(worker,(c,n)) for c in candidates for n in cfg['mesh_levels']]
            for future in as_completed(futures):
                rows.append(future.result()); update()
        for candidate in candidates:
            group = sorted([r for r in rows if r['case']['young_pa']==candidate['young_pa']
                and r['case']['force_n']==candidate['force_n']], key=lambda r:r['case']['cells'][0])
            comparisons = []
            for a,b in zip(group,group[1:]):
                with np.load(ROOT/a['path']/'field.npz',allow_pickle=False) as fa, np.load(ROOT/b['path']/'field.npz',allow_pickle=False) as fb:
                    field_error = compare_fields(fa,fb,a['case'])
                tip_a,tip_b = np.array(a['tip_displacement_m']),np.array(b['tip_displacement_m'])
                comparisons.append(dict(coarse=a['case']['cells'], fine=b['case']['cells'],
                    field_relative_l2=field_error, tip_relative_l2=float(np.linalg.norm(tip_a-tip_b)/np.linalg.norm(tip_b))))
            passed = all(r['healthy'] for r in group) and accepted_refinements(comparisons,
                cfg['field_relative_l2_tolerance'],cfg['tip_relative_l2_tolerance'],cfg['required_successive_refinements'])
            selected = group[-2]
            report['groups'].append(dict(young_pa=candidate['young_pa'],force_n=candidate['force_n'],
                comparisons=comparisons, numerically_converged=passed, selected_case_id=selected['id'],
                selected_path=selected['path'],reference_case_id=group[-1]['id'],
                eligible_for_render=bool(passed and cfg['minimum_motion_px']<=selected['ideal_max_motion_px']<=cfg['maximum_motion_px'])))
        report.update(status='complete', accepted_groups=sum(g['eligible_for_render'] for g in report['groups']))
        update()
        atomic(output/'status.json', dict(status='complete',reason='mesh_audit_complete'))
    except Exception as exc:
        report.update(status='failed',error=str(exc));update()
        atomic(output/'status.json', dict(status='failed',reason=str(exc)));raise


if __name__=='__main__': main()
