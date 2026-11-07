"""Bounded shared-node response diagnostics; never admits contact data to training.

Curved geometry/material discretization changes are included in the mesh delta.
Common reference nodes are a diagnostic, not a volume-integrated convergence proof.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from engine_fixture import atomic, digest, wait_if_paused

ROOT = Path(__file__).resolve().parents[1]


def common_node_indices(cells, reference_cells):
    cells = np.asarray(cells, dtype=int)
    reference_cells = np.asarray(reference_cells, dtype=int)
    if (cells % reference_cells).any():
        raise ValueError('Reference grid is not a shared subgrid')
    xyz = np.stack(np.meshgrid(*(np.arange(n + 1) for n in reference_cells), indexing='ij'), -1)
    indices = xyz.reshape(-1, 3) * (cells // reference_cells)
    return np.ravel_multi_index(indices.T, tuple(cells + 1))


def compare(a, b, reference_cells=(12, 6, 6)):
    """a is coarse dt/mesh; b is its finer reference, matched at real timestamps."""
    ca, cb = a['case'], b['case']
    ratio = ca['dt_seconds'] / cb['dt_seconds']
    if ratio < 1 or not np.isclose(ratio, round(ratio)):
        raise ValueError('Time grids are not nested')
    ia = common_node_indices(ca['cells'], reference_cells)
    ib = common_node_indices(cb['cells'], reference_cells)
    ua, ub = a['displacement'][:, ia], b['displacement'][round(ratio)-1::round(ratio), ib]
    if ua.shape != ub.shape or not np.isfinite(ua).all() or not np.isfinite(ub).all():
        raise ValueError('Invalid/misaligned trajectories')
    delta = np.linalg.norm(ua-ub)
    relative = float(delta / max(np.linalg.norm(ub), 1e-15))
    return dict(shared_reference_nodes=len(ia), full_time_common_node_relative_l2=relative,
                max_common_node_difference_m=float(np.linalg.norm(ua-ub, axis=-1).max()),
                comparison='Unweighted shared reference nodes at identical physical timestamps; not full volume field error')


def main():
    os.chdir(ROOT)
    repair = '--mesh-repair' in sys.argv
    output = ROOT/('runs/g3_contact_mesh_repair' if repair else 'runs/g3_contact_numerical_audit')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', completed=0, total=11, cases=[], comparisons=[],
                  admitted_for_training=False, scope=__doc__)
    if repair:
        report = json.loads((ROOT/'docs/G3_CONTACT_NUMERICAL_AUDIT.json').read_text())
        if len(report['cases']) < 11:
            raise ValueError('Original eleven diagnostics must finish before repair')
        report.update(status='running', total=13, repair_comparisons=[])
    def update(**kw):
        report.update(kw, updated_unix=time.time())
        atomic(ROOT/'docs/G3_CONTACT_NUMERICAL_AUDIT.json', report)
        atomic(output/'runtime.json', dict(phase=report['status'], completed=report['completed'],
               total=report['total'], detail='接触数值诊断；未准入训练', updated_unix=time.time()))
    source_files = ['engine_fixture.py', 'irregular_geometry.py', 'solve_contact_preflight.py']
    source = hashlib.sha256(b''.join((ROOT/'scripts'/p).read_bytes() for p in source_files)).hexdigest()
    def solve(case):
        wait_if_paused(output)
        folder = ROOT/'data/calibration/g3_contact_numerical_audit'/digest(dict(case=case, source=source))[:20]
        folder.mkdir(parents=True, exist_ok=True)
        atomic(folder/'case.json', case)
        if not (folder/'result.json').exists():
            with (folder/'solver.log').open('w') as log:
                subprocess.run([str(ROOT/'.engine-envs/uipc/bin/python'), 'scripts/solve_contact_preflight.py',
                    '--case', str(folder/'case.json'), '--output', str(folder), '--control-output', str(output)],
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        row = json.loads((folder/'result.json').read_text())
        if row['case'] != case or not row['completed'] or row['min_jacobian'] <= 0:
            raise ValueError('Numerical case failed')
        with np.load(folder/'trajectory.npz', allow_pickle=False) as z:
            displacement = z['displacement_m']
        if not np.isfinite(displacement).all():
            raise ValueError('Nonfinite trajectory')
        entry = dict(path=str(folder.relative_to(ROOT)), **row)
        if not any(x['path'] == entry['path'] for x in report['cases']):
            report['cases'].append(entry)
        update(completed=len(report['cases']))
        return dict(case=case, displacement=displacement)
    update()
    try:
        if repair:
            for layout in ['homogeneous', 'core_shell']:
                saved = []
                for cells in ([24,12,12], [36,18,18]):
                    row = next(x for x in report['cases'] if x['case']['layout'] == layout
                               and x['case']['cells'] == cells and x['case']['dt_seconds'] == .005
                               and x['case']['young_pa'] == [12000.,48000.])
                    with np.load(ROOT/row['path']/'trajectory.npz', allow_pickle=False) as z:
                        saved.append(dict(case=row['case'], displacement=z['displacement_m']))
                fine = copy.deepcopy(saved[-1]['case']); fine['cells'] = [48,24,24]
                saved.append(solve(fine))
                for i in (0, 1):
                    metrics = compare(saved[i], saved[i+1])
                    report['repair_comparisons'].append(dict(layout=layout, kind='mesh_repair',
                        cells_coarse=saved[i]['case']['cells'], cells_fine=saved[i+1]['case']['cells'],
                        threshold=.03, passed=metrics['full_time_common_node_relative_l2'] < .03, **metrics))
                update()
            passed = all(x['passed'] for x in report['repair_comparisons'])
            update(status='complete', repair_screen_passed=passed, rejected_coarse_cells=[12,6,6],
                   candidate_cells=[24,12,12] if passed else None,
                   detail='保留原粗网格失败证据；24→36→48 公共节点诊断已完成，仍未准入训练。')
            atomic(output/'status.json', dict(status='complete', reason='repair_passed_admission_pending' if passed else 'repair_failed_scale_up_blocked'))
            return
        homogeneous = None
        for layout in ['homogeneous', 'core_shell']:
            case = dict(shape='waisted', cells=[12,6,6], layout=layout, young_pa=[12000.,48000.],
                        poisson=.3, dt_seconds=.005, duration_s=.6, indentation_m=.004,
                        initial_gap_m=.0007, barrier_distance_m=.0001, friction=0.)
            baseline = solve(case)
            if layout == 'homogeneous':
                homogeneous = baseline
            meshes = [baseline]
            for cells in ([24,12,12], [36,18,18]):
                fine = copy.deepcopy(case); fine['cells'] = cells
                meshes.append(solve(fine))
            times = [baseline]
            for dt in (.0025, .00125):
                fine = copy.deepcopy(case); fine['dt_seconds'] = dt
                times.append(solve(fine))
            for kind, values, threshold in [('mesh', meshes, .03), ('time_step', times, .05)]:
                for i in (0, 1):
                    metrics = compare(values[i], values[i+1])
                    report['comparisons'].append(dict(layout=layout, kind=kind, refinement=i+1,
                        threshold=threshold, passed=metrics['full_time_common_node_relative_l2'] < threshold, **metrics))
            update()
        limit_case = copy.deepcopy(homogeneous['case'])
        limit_case.update(layout='core_shell', young_pa=[12000.,12000.])
        metrics = compare(homogeneous, solve(limit_case))
        report['comparisons'].append(dict(kind='homogeneous_limit', threshold=1e-4,
            passed=metrics['full_time_common_node_relative_l2'] < 1e-4, **metrics))
        passed = all(row['passed'] for row in report['comparisons'])
        update(status='complete', diagnostic_screen_passed=passed,
               next_gate='Calibrated reaction force, full surface clearance, energy/friction, identifiability; repair failed diagnostics before scale-up')
        atomic(output/'status.json', dict(status='complete', reason='diagnostics_passed_admission_pending' if passed else 'diagnostics_failed_scale_up_blocked'))
    except Exception as exc:
        update(status='failed', error_type=type(exc).__name__)
        atomic(output/'status.json', dict(status='failed', reason=type(exc).__name__))
        raise


if __name__ == '__main__':
    main()
