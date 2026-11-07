"""Single common fixture using actual Warp FEM or libuipc (isolated environments).

Warp is a static small-strain Lamé model; UIPC is dynamic SNK1 settled to static.
Their native lambda coefficients differ. This is not finite-strain equivalence.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from engine_fixture import atomic, block_mesh, provenance, tip_weights


def warp_solve(case, rest, tets):
    import warp as wp
    import warp.fem as fem
    from warp.examples.fem.utils import bsr_cg

    # Definitions need a source file for Warp's AST compiler.
    from warp_fixture_forms import stiffness, clamp_projector, tip_traction
    wp.init(); wp.set_module_options({'enable_backward': False})
    with wp.ScopedDevice('cuda:0'):
        geo = fem.Tetmesh(tet_vertex_indices=wp.array(tets, dtype=wp.int32),
                          positions=wp.array(rest, dtype=wp.vec3))
        space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec3)
        test = fem.make_test(space=space, domain=fem.Cells(geo))
        trial = fem.make_trial(space=space, domain=fem.Cells(geo))
        young, nu = case['young_pa'], case['poisson']
        mu = young / (2 * (1 + nu)); lam = young * nu / ((1 + nu) * (1 - 2 * nu))
        matrix = fem.integrate(stiffness, fields={'u': trial, 'v': test},
                               values={'mu': wp.float32(mu), 'lam': wp.float32(lam)}, output_dtype=wp.float32)
        boundary = fem.BoundarySides(geo)
        bd_test = fem.make_test(space=space, domain=boundary)
        bd_trial = fem.make_trial(space=space, domain=boundary)
        projector = fem.integrate(clamp_projector, fields={'u': bd_trial, 'v': bd_test}, assembly='nodal', output_dtype=wp.float32)
        force = np.zeros(3); force[case['axis']] = case['force_n'] / np.prod(case['dimensions_m'][1:])
        rhs = fem.integrate(tip_traction, fields={'v': bd_test},
                            values={'traction': wp.vec3(*force), 'length': wp.float32(case['dimensions_m'][0])}, output_dtype=wp.vec3)
        # Record assembled resultant before the fixed-boundary projection.
        resultant = rhs.numpy().sum(axis=0)
        fem.project_linear_system(matrix, rhs, projector)
        x = wp.zeros_like(rhs)
        residual, iterations = bsr_cg(matrix, b=rhs, x=x, tol=1e-11, max_iters=10000, quiet=True)
        from warp.sparse import bsr_mv
        ax = wp.zeros_like(rhs); bsr_mv(matrix, x, ax)
        wp.synchronize()
        actual_relative_residual = float(np.linalg.norm(ax.numpy()-rhs.numpy()) / np.linalg.norm(rhs.numpy()))
        displacement = x.numpy()
        # P1 Tetmesh topology uses vertex indices. Assert rather than assume order.
        nodes = space.node_positions().numpy()
        if not np.allclose(nodes, rest, rtol=0, atol=1e-8):
            raise ValueError('Warp P1 node ordering differs from fixture vertices')
        return rest + displacement, {'native_coefficients_pa': {'mu': mu, 'lambda_lame': lam},
            'assembled_resultant_n': resultant.tolist(), 'cg_residual': float(residual),
            'actual_relative_residual': actual_relative_residual, 'field_precision': 'float32',
            'cg_iterations': int(iterations), 'completed': True, 'final_max_speed_m_s': None,
            'last20_displacement_change_m': None, 'engine_version': importlib.metadata.version('warp-lang')}


def uipc_solve(case, rest, tets, workspace):
    from uipc import Engine, Scene, Timer, World, builtin, view
    from uipc.constitution import ElasticModuli, FiniteElementExternalForce, StableNeoHookean
    from uipc.core import FiniteElementStateAccessorFeature
    from uipc.geometry import label_surface, label_triangle_orient, tetmesh
    Timer.disable_all(); workspace.mkdir(parents=True, exist_ok=True)
    young, nu = case['young_pa'], case['poisson']
    mu = young / (2 * (1 + nu)); lam = young * nu / ((1 + nu) * (1 - 2 * nu))
    mesh = tetmesh(rest, tets); label_surface(mesh); label_triangle_orient(mesh)
    StableNeoHookean().apply_to(mesh, ElasticModuli.youngs_poisson(young, nu), mass_density=1000.)
    view(mesh.tetrahedra().find('mu'))[:] = mu
    view(mesh.tetrahedra().find('lambda'))[:] = lam + mu
    fixed = np.isclose(rest[:, 0], 0, rtol=0, atol=1e-12)
    view(mesh.vertices().find(builtin.is_fixed))[fixed] = 1
    FiniteElementExternalForce().apply_to(mesh, np.zeros(3))
    weights = tip_weights(rest, tets, case['dimensions_m'][0])
    nodal_force = np.zeros_like(rest); nodal_force[:, case['axis']] = weights * case['force_n']
    cfg = Scene.default_config(); cfg['gravity'] = [[0.], [0.], [0.]]; cfg['contact']['enable'] = False
    cfg['dt'] = case.get('dt_seconds', .02); cfg['newton']['velocity_tol'] = 1e-9
    cfg['newton']['semi_implicit']['enable'] = False; cfg['linear_system']['tol_rate'] = 1e-9
    engine = Engine('cuda', str(workspace)); world = World(engine); scene = Scene(cfg)
    obj = scene.objects().create('fixture'); obj.geometries().create(mesh)
    def animate(info):
        geom = info.geo_slots()[0].geometry()
        view(geom.vertices().find('external_force'))[:] = nodal_force.reshape(-1, 3, 1)
        view(geom.vertices().find('external_force/is_constrained'))[:] = (weights > 0).reshape(-1, 1)
    scene.animator().insert(obj, animate); world.init(scene)
    accessor = world.features().find(FiniteElementStateAccessorFeature)
    trace = []; completed = True
    for _ in range(case.get('steps', 200)):
        world.advance(); world.retrieve()
        state = accessor.create_geometry(); state.vertices().create(builtin.position, np.zeros(3))
        state.vertices().create(builtin.velocity, np.zeros(3)); accessor.copy_to(state)
        x = np.array(view(state.vertices().find(builtin.position))).reshape(-1, 3)
        speed = float(np.linalg.norm(np.array(view(state.vertices().find(builtin.velocity))).reshape(-1, 3), axis=1).max())
        trace.append((x - rest).copy()); completed &= bool(engine.frame_stats()['completed'])
    # Keep actual dynamics for later replay checks; Warp has no video trajectory.
    np.savez_compressed(workspace / 'trajectory.npz', displacement_m=np.array(trace), dt_seconds=cfg['dt'])
    return x, {'native_coefficients_pa': {'mu': mu, 'lambda_snk1': lam + mu},
        'assembled_resultant_n': nodal_force.sum(axis=0).tolist(), 'completed': completed,
        'final_max_speed_m_s': speed,
        'last20_displacement_change_m': float(np.ptp(np.stack(trace[-20:]), axis=0).max()),
        'engine_version': importlib.metadata.version('pyuipc')}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--case', required=True)
    p.add_argument('--engine', choices=['warp', 'uipc'], required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); case = json.loads(Path(args.case).read_text()); output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True); start = time.perf_counter()
    rest, tets = block_mesh(case['cells'], case['dimensions_m'])
    x, result = warp_solve(case, rest, tets) if args.engine == 'warp' else uipc_solve(case, rest, tets, output / 'world')
    basis = np.stack([rest[tets[:, i]] - rest[tets[:, 0]] for i in range(1, 4)], -1)
    current = np.stack([x[tets[:, i]] - x[tets[:, 0]] for i in range(1, 4)], -1)
    deformation = current @ np.linalg.inv(basis)
    tip = tip_weights(rest, tets, case['dimensions_m'][0])
    result.update(engine=args.engine, case=case, response_provenance=provenance(case),
        fixed_vertex_max_error_m=float(np.abs((x-rest)[rest[:, 0] == 0]).max()),
        min_jacobian=float(np.linalg.det(deformation).min()),
        max_displacement_gradient=float(np.linalg.norm(deformation - np.eye(3), axis=(1, 2)).max()),
        tip_displacement_m=(tip @ (x-rest)).tolist(), wall_seconds=time.perf_counter()-start)
    np.savez_compressed(output / 'field.npz', rest_positions_m=rest, displacement_m=x-rest, tetrahedra=tets)
    atomic(output / 'result.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
