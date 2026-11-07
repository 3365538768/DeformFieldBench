"""Common SI fixture, mesh and consistent surface loads for engine calibration."""
import hashlib
import itertools
import json
from pathlib import Path
import time
import subprocess
import uuid

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def block_mesh(cells, dimensions):
    shape = tuple(n + 1 for n in cells)
    points = np.stack(np.meshgrid(*(np.linspace(0, d, n + 1) for n, d in zip(cells, dimensions)),
                                 indexing='ij'), -1).reshape(-1, 3)
    tets = []
    for cell in itertools.product(*(range(n) for n in cells)):
        for permutation in itertools.permutations(range(3)):
            vertices = [np.array(cell)]
            for axis in permutation:
                vertex = vertices[-1].copy(); vertex[axis] += 1; vertices.append(vertex)
            tet = [int(np.ravel_multi_index(tuple(v), shape)) for v in vertices]
            if np.linalg.det((points[tet[1:]] - points[tet[0]]).T) < 0:
                tet[1], tet[2] = tet[2], tet[1]
            tets.append(tet)
    return points, np.array(tets, dtype=np.int32)


def tip_weights(points, tets, length):
    """Linear FEM traction: each exterior tip triangle contributes area / 3."""
    faces = {}
    for tet in tets:
        for face in itertools.combinations(tet, 3):
            key = tuple(sorted(int(i) for i in face)); faces[key] = faces.get(key, 0) + 1
    weights = np.zeros(len(points))
    for face, count in faces.items():
        xyz = points[list(face)]
        if count == 1 and np.all(np.isclose(xyz[:, 0], length, rtol=0, atol=1e-12)):
            area = np.linalg.norm(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])) / 2
            weights[list(face)] += area / 3
    if weights.sum() <= 0:
        raise ValueError('Missing exterior traction face')
    return weights / weights.sum()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8'); tmp.replace(path)


def is_paused(output):
    for path in (Path('docs/OPTIMIZATION_CONTROL.json'),Path(output)/'control.json'):
        if path.exists():
            value=json.loads(path.read_text())
            if value.get('desired',value.get('action')) in ('paused','pause'):return True
    return False


def run_controlled_solver(command, log, control_output, active_timeout_seconds, cwd):
    """Cooperative children pause themselves; wall time spent paused isn't a timeout."""
    process=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    previous=time.monotonic();active=0.
    try:
        while process.poll() is None:
            time.sleep(.25)
            current=time.monotonic()
            if not is_paused(control_output):active+=current-previous
            previous=current
            if active>active_timeout_seconds:
                raise subprocess.TimeoutExpired(command,active_timeout_seconds)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode,command)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        raise


def wait_if_paused(output):
    """Cooperative pause at case boundaries; no Torch dependency in engine envs."""
    was_paused = False
    while True:
        paused = is_paused(output)
        if not paused:
            if was_paused:
                atomic(Path(output) / 'runtime.json', {'phase': 'running', 'updated_unix': time.time()})
            return
        was_paused = True
        atomic(Path(output) / 'runtime.json', {'phase': 'paused', 'updated_unix': time.time()})
        time.sleep(2)


def provenance(case):
    return dict(protocol_sha256=digest({'load': 'constant tip traction', 'observable': 'settled displacement',
                                       'strain_regime': 'measured per result; not imposed', 'temperature': 'unspecified isothermal'}),
                fixture_geometry_sha256=digest({'dimensions_m': case['dimensions_m'], 'cells': case['cells']}),
                length_scale_m=case['dimensions_m'][0],
                load_history_sha256=digest({'force_n': case['force_n'], 'axis': case['axis'],
                    'history':case.get('force_history','constant'),'time_unit':'s'}),
                boundary_sha256=digest({'clamped': 'x=0 all displacement components'}),
                observable='settled full nodal displacement', units='m')
