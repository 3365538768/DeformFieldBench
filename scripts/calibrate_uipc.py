"""Small measured CUDA FEM calibration; not a production dataset acceptance test.

Run with the isolated pyuipc environment, not the training environment.
The free tetrahedron has known SI mass and a finite force pulse. Check total
momentum, clearing the external force, positive Jacobians and dt convergence.
API reference: libuipc python/tests/sim_case/test_finite_element_external_force.py.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from uipc import Engine, Scene, Timer, World, builtin, view
from uipc.constitution import ElasticModuli, FiniteElementExternalForce, StableNeoHookean
from uipc.core import FiniteElementStateAccessorFeature
from uipc.geometry import label_surface, label_triangle_orient, tetmesh


def simulate(workspace, young, density, dt):
    Timer.disable_all()
    start = time.perf_counter()
    workspace.mkdir(parents=True, exist_ok=True)
    engine = Engine('cuda', str(workspace))
    world = World(engine)
    cfg = Scene.default_config()
    cfg['gravity'] = [[0.0], [0.0], [0.0]]
    cfg['contact']['enable'] = False
    cfg['dt'] = dt
    cfg['newton']['velocity_tol'] = 1e-7
    cfg['newton']['semi_implicit']['enable'] = False
    cfg['linear_system']['tol_rate'] = 1e-8
    scene = Scene(cfg)
    rest = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]], dtype=np.float64)
    rest_basis = (rest[1:] - rest[0]).T
    rest_volume = float(np.linalg.det(rest_basis) / 6)
    mass = density * rest_volume
    mesh = tetmesh(rest, np.array([[0, 1, 2, 3]], dtype=np.int32))
    label_surface(mesh)
    label_triangle_orient(mesh)
    StableNeoHookean().apply_to(mesh, ElasticModuli.youngs_poisson(young, .35), mass_density=density)
    FiniteElementExternalForce().apply_to(mesh, np.zeros(3))
    obj = scene.objects().create('calibration_tet')
    obj.geometries().create(mesh)
    pulse_seconds, duration_seconds, force_n = .004, .04, 1.0
    pulse_steps = round(pulse_seconds / dt)

    def animate(info):
        geometry = info.geo_slots()[0].geometry()
        f = view(geometry.vertices().find('external_force'))
        enabled = view(geometry.vertices().find('external_force/is_constrained'))
        f[:] = 0
        enabled[:] = 0
        if 1 <= info.frame() <= pulse_steps:
            enabled[0] = 1
            f[0] = np.array([[force_n], [0.0], [0.0]])

    scene.animator().insert(obj, animate)
    world.init(scene)
    feature = world.features().find(FiniteElementStateAccessorFeature)
    if feature is None:
        raise RuntimeError('Engine has no FEM state accessor')
    positions, velocities, jacobians = [], [], []
    completed = True
    for step in range(round(duration_seconds / dt)):
        world.advance()
        world.retrieve()
        state = feature.create_geometry()
        state.vertices().create(builtin.position, np.zeros(3))
        state.vertices().create(builtin.velocity, np.zeros(3))
        feature.copy_to(state)
        x = np.array(view(state.vertices().find(builtin.position))).reshape(-1, 3)
        v = np.array(view(state.vertices().find(builtin.velocity))).reshape(-1, 3)
        if not np.isfinite(x).all() or not np.isfinite(v).all():
            raise ValueError('Nonfinite FEM state')
        positions.append(x)
        velocities.append(v)
        jacobians.append(float(np.linalg.det((x[1:] - x[0]).T) / np.linalg.det(rest_basis)))
        completed &= bool(engine.frame_stats()['completed'])
    x, v = np.array(positions), np.array(velocities)
    # Equal lumped mass at the four vertices of the single tet.
    momentum = mass * v.mean(axis=1)
    expected = np.zeros_like(momentum)
    expected[:, 0] = np.minimum(np.arange(1, len(v) + 1) * dt, pulse_seconds) * force_n
    impulse = force_n * pulse_seconds
    error = np.linalg.norm(momentum - expected, axis=1).max() / impulse
    drift = np.linalg.norm(momentum[pulse_steps:] - momentum[pulse_steps-1], axis=1).max() / impulse
    # Displacement relative to COM measures actual shape change, not translation.
    deformation = x - x.mean(axis=1, keepdims=True) - (rest - rest.mean(axis=0))[None]
    workspace.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(workspace / 'trajectory.npz', positions_m=x, velocities_m_s=v,
                        momentum_kg_m_s=momentum, jacobian=np.array(jacobians), dt_seconds=dt)
    return {'young_pa': young, 'poisson': .35, 'density_kg_m3': density, 'dt_seconds': dt,
            'frames': len(x), 'pulse_seconds': pulse_seconds, 'force_n': force_n,
            'rest_volume_m3': rest_volume, 'mass_kg': mass,
            'relative_momentum_error': float(error), 'post_pulse_momentum_drift': float(drift),
            'min_jacobian': min(jacobians), 'peak_shape_displacement_m': float(np.linalg.norm(deformation, axis=2).max()),
            'final_positions_m': x[-1].tolist(), 'completed': completed,
            'wall_seconds': time.perf_counter() - start, 'scene_config': cfg,
            'passed': bool(completed and error < .01 and drift < .01 and min(jacobians) > 0)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='docs/UIPC_CALIBRATION.json')
    p.add_argument('--workspace', default='data/calibration/uipc_v0029')
    args = p.parse_args()
    output = Path(args.output)
    cases = []
    for young, density in [(1e4, 1000.), (1e5, 1000.), (1e4, 500.)]:
        for dt in [.002, .001]:
            cases.append(simulate(Path(args.workspace) / f'E{young:g}_rho{density:g}_dt{dt:g}', young, density, dt))
            print(json.dumps({k:v for k,v in cases[-1].items() if k not in ('scene_config','final_positions_m')}), flush=True)
    convergence = []
    for coarse, fine in zip(cases[::2], cases[1::2]):
        delta = np.linalg.norm(np.array(coarse['final_positions_m']) - fine['final_positions_m'], axis=1).max()
        convergence.append({'young_pa': coarse['young_pa'], 'density_kg_m3': coarse['density_kg_m3'],
                            'max_final_position_difference_m': float(delta), 'relative_to_size': float(delta / .1)})
    result = {'status': 'passed' if all(c['passed'] for c in cases) and all(c['relative_to_size'] < .01 for c in convergence) else 'failed',
              'engine': 'libuipc', 'package': 'pyuipc', 'version': importlib.metadata.version('pyuipc'),
              'cases': cases, 'dt_convergence': convergence,
              'scope': 'Single free tetrahedron: SI impulse/mass, force clearing, finite positive volume, dt sensitivity. Does NOT certify contact, stress labels, complex mesh, mixed materials or real-video fitting.',
              'reference': 'https://github.com/spiriMirror/libuipc/blob/main/python/tests/sim_case/test_finite_element_external_force.py'}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(output)
    print('CALIBRATION_RESULT', result['status'], flush=True)
    if result['status'] != 'passed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
