"""Gated two-engine SI response pilot. Resumable cases; cooperative pause.

This calibrates observed deformation fields, NOT a video inference benchmark.
All thresholds and the geometry split are specified before generation/training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
SOLVER_SOURCE_SHA = hashlib.sha256(b''.join((ROOT/'scripts'/name).read_bytes() for name in
    ('engine_fixture.py','solve_engine_fixture.py','warp_fixture_forms.py','interaction_protocol.py'))).hexdigest()


def solve(engine, case, base, timeout=180):
    identifier = engine + '_' + digest({'case':case,'solver_source_sha256':SOLVER_SOURCE_SHA})[:20]
    output = base / identifier; receipt = output / 'result.json'
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if result['case'] != case or result['engine'] != engine or not (output / 'field.npz').exists():
            raise ValueError('Existing case provenance mismatch')
        return identifier, result
    output.mkdir(parents=True, exist_ok=True); atomic(output / 'case.json', case)
    with (output / 'solver.log').open('w') as log:
        subprocess.run([str(ROOT / '.engine-envs' / engine / 'bin/python'), 'scripts/solve_engine_fixture.py',
                        '--engine', engine, '--case', str(output / 'case.json'), '--output', str(output)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    result = json.loads(receipt.read_text()); result['solver_source_sha256'] = SOLVER_SOURCE_SHA
    atomic(receipt, result)
    return identifier, result


def healthy(result):
    expected = np.eye(3)[result['case']['axis']] * result['case']['force_n']
    displacement = np.linalg.norm(result['tip_displacement_m'])
    return bool(result['completed'] and result['fixed_vertex_max_error_m'] < 1e-9
        and result['min_jacobian'] > .95 and result['max_displacement_gradient'] < .02
        and np.isfinite(displacement) and displacement > 1e-10
        and np.linalg.norm(np.array(result['assembled_resultant_n']) - expected) / np.linalg.norm(expected) < 1e-5
        and (result['final_max_speed_m_s'] is None or result['final_max_speed_m_s'] < 1e-6)
        and (result['last20_displacement_change_m'] is None or result['last20_displacement_change_m'] / displacement < .001)
        and result.get('actual_relative_residual', 0) < 1e-3)


def run(output, workers):
    base = ROOT / 'data/calibration/g2_cases'; base.mkdir(parents=True, exist_ok=True)
    start = time.time(); completed = 0
    progress = {'status': 'running', 'completed': 0, 'total': 10, 'phase': 'calibration',
                'detail': '两引擎同夹具：载荷分配、边界、残差、网格/时间步及小形变响应检查。'}
    def update(**values):
        progress.update(values, updated_unix=time.time(), started_unix=start)
        atomic(ROOT / 'docs/G2_ENGINE_PROGRESS.json', progress)
        atomic(output / 'runtime.json', {'phase': progress['status'], **progress})
    update()
    case = json.loads((ROOT / 'configs/g2_calibration_case.json').read_text())
    fine = {**case, 'cells': [16, 8, 8]}
    calibration = [case, fine, {**fine, 'young_pa': 40000.}, {**fine, 'axis': 0},
                   {**fine, 'dt_seconds': .01, 'steps': 400}]
    results = {}
    for i, item in enumerate(calibration):
        for engine in ('warp', 'uipc'):
            wait_if_paused(output); key, receipt = solve(engine, item, base)
            results[(engine, i)] = (key, receipt); completed += 1
            update(completed=completed, detail=f'标定 {completed}/10：{engine} / 配置 {i}；通过后才生成数据。')
    comparisons = []
    for i in range(len(calibration)):
        a, b = (results[(engine, i)][1] for engine in ('warp', 'uipc'))
        if a['response_provenance'] != b['response_provenance']:
            raise ValueError('Cross-engine response provenance mismatch')
        fields = [np.load(base / results[(engine, i)][0] / 'field.npz')['displacement_m'] for engine in ('warp', 'uipc')]
        comparisons.append(float(np.linalg.norm(fields[0]-fields[1])/np.linalg.norm(fields[0])))
    mesh_change = {}; inverse_e = {}; dt_change = {}
    for engine in ('warp', 'uipc'):
        tip = [np.array(results[(engine, i)][1]['tip_displacement_m']) for i in range(5)]
        mesh_change[engine] = float(np.linalg.norm(tip[1]-tip[0])/np.linalg.norm(tip[1]))
        inverse_e[engine] = float(abs(np.linalg.norm(tip[1])/np.linalg.norm(tip[2])/2-1))
        dt_change[engine] = float(np.linalg.norm(tip[4]-tip[1])/np.linalg.norm(tip[1]))
    gates = {'all_cases_healthy': all(healthy(v[1]) for v in results.values()),
             'cross_engine_field_difference_below_2_percent': max(comparisons) < .02,
             'mesh_change_below_15_percent': max(mesh_change.values()) < .15,
             'inverse_E_error_below_1_percent': max(inverse_e.values()) < .01,
             'time_step_change_below_1_percent': max(dt_change.values()) < .01}
    calibration_receipt = {'status': 'passed' if all(gates.values()) else 'failed', 'gates': gates,
        'cross_engine_relative_field_errors': comparisons, 'mesh_relative_changes': mesh_change,
        'inverse_E_relative_errors': inverse_e, 'time_step_relative_changes': dt_change,
        'cases': [{'id': key, **receipt} for key, receipt in results.values()],
        'scope': 'Known SI block, clamp and small known tip traction. Warp linear elasticity vs corrected UIPC SNK1. No contact, finite-strain equivalence, density inference, mixed material or RGB claims.'}
    atomic(ROOT / 'docs/G2_CROSS_ENGINE_CALIBRATION.json', calibration_receipt)
    if not all(gates.values()):
        raise ValueError('Calibration gate failed: ' + ','.join(k for k, v in gates.items() if not v))
    # Entire material/engine/action group stays within its fixture geometry split.
    geometries = [([.032,.020,.020], 'train'), ([.040,.016,.020], 'train'),
                  ([.048,.024,.020], 'train'), ([.044,.020,.024], 'train'),
                  ([.036,.018,.024], 'val'), ([.046,.022,.018], 'val')]
    tasks = []
    for fixture, (dimensions, split) in enumerate(geometries):
        for e in (12000., 24000., 48000.):
            for nu in (.20, .30, .40):
                for axis in (0, 1, 2):
                    item = {**case, 'dimensions_m': dimensions, 'young_pa': e, 'poisson': nu, 'axis': axis}
                    for engine in ('warp', 'uipc'):
                        tasks.append((engine, item, {'fixture': fixture, 'split': split,
                            'material_pair': digest({'E':e,'nu':nu}), 'axis': axis}))
    update(phase='response_matrix', total=10+len(tasks), detail='标定通过：生成 SI 位移场矩阵；4 个训练夹具 / 2 个验证夹具，整组隔离。')
    manifest = []
    atomic(ROOT/'docs/G2_SIMULATION_CASES.json', {'status':'generating', 'rows':[], 'planned':len(tasks),
        'pair_validation':'pending', 'updated_unix':time.time()})
    def worker(task):
        engine, item, meta = task; wait_if_paused(output)
        key, receipt = solve(engine, item, base)
        if not healthy(receipt):
            raise ValueError('Unhealthy generated case: '+key)
        return {**meta, 'engine': engine, 'id': key, 'path': str((base/key).relative_to(ROOT)), **receipt}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            manifest.append(future.result()); completed += 1
            atomic(ROOT/'docs/G2_SIMULATION_CASES.json', {'status':'generating', 'rows':manifest,
                'planned':len(tasks), 'pair_validation':'pending', 'updated_unix':time.time()})
            update(completed=completed, detail=f'SI 响应矩阵 {len(manifest)}/{len(tasks)}；仅使用通过数值健康检查的案例。')
    manifest.sort(key=lambda r: (r['fixture'], r['material_pair'], r['axis'], r['engine']))
    # Gate EVERY matched pair, not only the original fixture.
    grouped = {}
    for row in manifest:
        grouped.setdefault((row['fixture'], row['material_pair'], row['axis']), {})[row['engine']] = row
    errors = []
    for pair in grouped.values():
        a, b = pair['warp'], pair['uipc']
        if a['response_provenance'] != b['response_provenance']:
            raise ValueError('Generated pair provenance mismatch')
        x, y = [np.load(ROOT/row['path']/'field.npz')['displacement_m'] for row in (a, b)]
        errors.append(float(np.linalg.norm(x-y)/np.linalg.norm(x)))
    if max(errors) >= .02:
        raise ValueError('Generated pair field error exceeds preregistered 2 percent gate')
    manifest_path = ROOT / 'data/manifests/g2_engine_response.json'
    atomic(manifest_path, {'rows': manifest, 'case_count': len(manifest), 'pair_errors': errors,
                          'geometry_split': geometries, 'test_used': False, 'schema_version': 1})
    atomic(ROOT/'docs/G2_SIMULATION_CASES.json', {'status':'validated', 'rows':manifest,
        'planned':len(tasks), 'pair_validation':'passed', 'updated_unix':time.time()})
    update(phase='adapter_training', total=completed+9, detail='矩阵全部通过：训练独立原生头、共享响应辅助及配对对齐三个控制，3 个固定种子。')
    subprocess.run([sys.executable, 'scripts/train_engine_adapter_pilot.py', '--manifest', str(manifest_path),
                    '--output', str(output/'adapter')], check=True)
    update(status='complete', phase='finished', completed=completed+9, detail='受控跨引擎场实验完成；请查看适配误差和独立种子结果。尚无视频推断或在线拟合胜出结论。')
    atomic(output/'status.json', {'status':'complete', 'reason':'controlled_engine_pilot_complete',
                                 'updated_unix':time.time()})


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', default='runs/g2_cross_engine_pilot')
    p.add_argument('--workers', type=int, choices=range(1,5), default=2); args = p.parse_args()
    os.chdir(ROOT); output = ROOT / args.output
    try:
        run(output, args.workers)
    except Exception as exc:
        # No credentials/URLs are embedded in these engine commands.
        atomic(ROOT/'docs/G2_ENGINE_PROGRESS.json', {'status':'failed', 'detail':str(exc), 'updated_unix':time.time()})
        atomic(output/'status.json', {'status':'failed', 'reason':str(exc), 'updated_unix':time.time()})
        raise


if __name__ == '__main__':
    main()
