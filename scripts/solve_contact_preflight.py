"""One real 3D UIPC tool/solid contact probe, NOT admitted training ground truth.

Reference: libuipc-samples examples 12 (soft transform constraint) and 90
(ABD/FEM contact). Metric geometry and bonded per-tet coefficients are ours.
No GUI, no image-space force substitution, no inferred reaction-force labels.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from engine_fixture import atomic, block_mesh, wait_if_paused
from irregular_geometry import make_solid, material_regions, mesh_health


def tool_motion(t):
    if t <= .2:
        return .5-.5*np.cos(np.pi*t/.2)
    if t <= .25:
        return 1.
    if t <= .45:
        return .5+.5*np.cos(np.pi*(t-.25)/.2)
    return 0.


def run(case, output, control_output):
    from uipc import Engine, Logger, Scene, Timer, World, builtin, view
    from uipc.constitution import AffineBodyConstitution, ElasticModuli, SoftTransformConstraint, StableNeoHookean
    from uipc.core import FiniteElementStateAccessorFeature, ContactSystemFeature
    from uipc.geometry import label_surface, label_triangle_orient, tetmesh, Geometry

    started = time.time(); output.mkdir(parents=True, exist_ok=True); Timer.disable_all(); Logger.set_level(Logger.Level.Warn)
    rest, tets, reference = make_solid(case['shape'], case['cells'])
    health = mesh_health(rest, tets)
    if not health['passed']:
        raise ValueError('Rest mesh did not pass signed-volume/manifold screening')
    ids = material_regions(reference, tets, case['layout'])
    young = np.asarray(case['young_pa'])[ids]; nu = case['poisson']
    mu = young/(2*(1+nu)); lam_snk1 = young*nu/((1+nu)*(1-2*nu))+mu
    solid = tetmesh(rest, tets); label_surface(solid); label_triangle_orient(solid)
    StableNeoHookean().apply_to(solid, ElasticModuli.youngs_poisson(float(young[0]), nu), 1000.)
    view(solid.tetrahedra().find('mu')).reshape(-1)[:] = mu
    view(solid.tetrahedra().find('lambda')).reshape(-1)[:] = lam_snk1
    fixed = np.isclose(rest[:, 0], 0, rtol=0, atol=1e-12)
    view(solid.vertices().find(builtin.is_fixed))[fixed] = 1

    # The moving body is a real finite rigid cuboid contacting the deformable mesh.
    tool_rest, tool_tets = block_mesh([1,1,1], [.010,.006,.030])
    tool_rest += [.025, float(rest[:,1].max())+case['initial_gap_m'], -.015]
    tool = tetmesh(tool_rest, tool_tets); label_surface(tool); label_triangle_orient(tool)
    AffineBodyConstitution().apply_to(tool, 1e9, 1000.)
    SoftTransformConstraint().apply_to(tool, np.array([1e6,1e6]))

    cfg = Scene.default_config(); cfg['gravity'] = [[0.],[0.],[0.]]
    cfg['dt'] = case['dt_seconds']; cfg['contact']['d_hat'] = case['barrier_distance_m']
    cfg['contact']['d_hat_relative'] = 0.; cfg['contact']['friction']['enable'] = case['friction'] > 0
    cfg['contact']['eps_velocity'] = 1e-5; cfg['newton']['velocity_tol'] = 1e-7
    cfg['newton']['semi_implicit']['enable'] = False; cfg['linear_system']['tol_rate'] = 1e-7
    scene = Scene(cfg); scene.contact_tabular().default_model(case['friction'], 1e8)
    for mesh in (solid, tool):
        scene.contact_tabular().default_element().apply_to(mesh)
    soft_obj = scene.objects().create('bonded_solid'); solid_slot, _ = soft_obj.geometries().create(solid)
    tool_obj = scene.objects().create('physical_punch'); tool_slot, _ = tool_obj.geometries().create(tool)
    target = [0.]
    def animate(info):
        mesh = info.geo_slots()[0].geometry()
        view(mesh.instances().find(builtin.is_constrained))[:] = 1
        transform = np.eye(4); transform[1,3] = -target[0]
        view(mesh.instances().find(builtin.aim_transform))[:] = transform
    scene.animator().insert(tool_obj, animate)
    (output/'world').mkdir(parents=True, exist_ok=True)
    engine = Engine('cuda', str(output/'world')); world = World(engine); world.init(scene)
    accessor = world.features().find(FiniteElementStateAccessorFeature)
    diagnostics = case.get('export_contact_diagnostics', False)
    if diagnostics:
        from contact_diagnostics import export_contact, snk1_state
        contact_feature = world.features().find(ContactSystemFeature)
        tool_offset = int(np.asarray(view(tool_slot.geometry().meta().find(builtin.global_vertex_offset))).reshape(-1)[0])
        solid_offset = int(np.asarray(view(solid_slot.geometry().meta().find(builtin.global_vertex_offset))).reshape(-1)[0])
        if set(range(tool_offset, tool_offset+len(tool_rest))) & set(range(solid_offset, solid_offset+len(rest))):
            raise ValueError('Global vertex ranges overlap')
        vertex_count = max(tool_offset+len(tool_rest), solid_offset+len(rest))
        force_trace = []; energy_trace = []; residual_trace = []; balance_trace = []; work_trace = []
        previous = rest.copy(); previous_velocity = np.zeros_like(rest); previous_tool = tool_rest.copy(); work = 0.
    inverse_basis = np.linalg.inv(np.stack([rest[tets[:,i]]-rest[tets[:,0]] for i in (1,2,3)], -1))
    trace = []; tools = []; targets = []; min_j = 1.; max_fixed = 0.; tracking = []; inside_count = 0
    completed = True; steps = round(case['duration_s']/case['dt_seconds'])
    for i in range(steps):
        wait_if_paused(control_output)
        target[0] = case['indentation_m']*tool_motion((i+1)*case['dt_seconds']/case.get('motion_time_scale', 1.))
        world.advance(); world.retrieve()
        state = accessor.create_geometry(); state.vertices().create(builtin.position, np.zeros(3))
        accessor.copy_to(state); positions = np.asarray(view(state.vertices().find(builtin.position))).reshape(-1,3).copy()
        if positions.shape != rest.shape:
            raise ValueError('Unexpected FEM vertex ordering/count')
        transform = np.asarray(tool_slot.geometry().transforms().view()).reshape(-1,4,4)[0].copy()
        actual_tool = tool_rest@transform[:3,:3].T+transform[:3,3]
        if diagnostics:
            dt = case['dt_seconds']
            forces, barrier_energy = export_contact(contact_feature, view, Geometry, vertex_count, dt)
            tool_force = forces[tool_offset:tool_offset+len(tool_rest)]
            solid_force = forces[solid_offset:solid_offset+len(rest)]
            mechanical = snk1_state(rest, positions, tets, mu, lam_snk1)
            mass = mechanical['lumped_mass_kg']
            velocity = (positions-previous)/dt
            inertia = mass[:, None]*(velocity-previous_velocity)/dt
            residual = inertia+mechanical['elastic_gradient_n']-solid_force
            force_scale = max(np.linalg.norm(solid_force, axis=1).sum(), 1e-12)
            residual_trace.append(np.linalg.norm(residual[~fixed], axis=1).sum()/force_scale)
            balance_trace.append(np.linalg.norm(forces.sum(0))/force_scale)
            work -= float((tool_force*(actual_tool-previous_tool)).sum())
            work_trace.append(work)
            force_trace.append(tool_force.sum(0))
            energy_trace.append([mechanical['elastic_energy_j'], .5*float((mass[:, None]*velocity**2).sum()), barrier_energy])
            previous = positions.copy(); previous_velocity = velocity; previous_tool = actual_tool.copy()
        current = np.stack([positions[tets[:,j]]-positions[tets[:,0]] for j in (1,2,3)], -1)
        min_j = min(min_j, float(np.linalg.det(current@inverse_basis).min()))
        max_fixed = max(max_fixed, float(np.abs((positions-rest)[fixed]).max()))
        local = (positions-transform[:3,3])@np.linalg.inv(transform[:3,:3]).T
        # A diagnostic only: vertex-inside-box misses edge/face intersections.
        inside = ((local > tool_rest.min(0)+1e-8)&(local < tool_rest.max(0)-1e-8)).all(1)
        inside_count = max(inside_count, int(inside.sum()))
        tracking.append(abs(float(transform[1,3])+target[0]))
        trace.append(positions-rest); tools.append(actual_tool); targets.append(target[0])
        completed &= bool(engine.frame_stats()['completed'])
        if not completed:
            raise ValueError('Engine reported incomplete contact step')
        if not np.isfinite(positions).all() or min_j <= 0:
            raise ValueError('Nonfinite/inverted contact trajectory')
        atomic(control_output/'runtime.json', dict(phase='running', completed=i+1,total=steps,
            detail=f"三维压头接触预检：{case['shape']} / {case['layout']} / {i+1}/{steps}", updated_unix=time.time()))
    displacement = np.asarray(trace)
    np.savez_compressed(output/'trajectory.npz',rest_positions_m=rest,tetrahedra=tets,material_ids=ids,
        displacement_m=displacement,tool_rest_positions_m=tool_rest,tool_tetrahedra=tool_tets,
        tool_positions_m=np.asarray(tools),tool_target_indentation_m=np.asarray(targets),dt_seconds=case['dt_seconds'])
    np.savez_compressed(output/'field.npz',rest_positions_m=rest,tetrahedra=tets,
        displacement_m=displacement[-1],material_ids=ids,young_pa=young,mu_pa=mu,lambda_snk1_pa=lam_snk1)
    if diagnostics:
        np.savez_compressed(output/'contact_diagnostics.npz',tool_contact_force_n=np.asarray(force_trace),
            energy_j=np.asarray(energy_trace),energy_columns=np.asarray(['elastic_relative_rest','solid_kinetic_backward_difference','contact_barrier']),
            input_work_endpoint_j=np.asarray(work_trace),free_node_relative_force_residual=np.asarray(residual_trace),
            contact_action_reaction_relative_error=np.asarray(balance_trace),dt_seconds=case['dt_seconds'])
    result = dict(status='probe_complete',case=case,engine='uipc',engine_version=importlib.metadata.version('pyuipc'),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),mesh=health,completed=completed,
        min_jacobian=min_j,fixed_vertex_max_error_m=max_fixed,max_tool_tracking_error_m=max(tracking),
        vertex_inside_tool_count=inside_count,max_displacement_m=float(np.linalg.norm(displacement,axis=-1).max()),
        final_displacement_m=float(np.linalg.norm(displacement[-1],axis=-1).max()),
        native_parameter_mapping='mu=E/(2(1+nu)); lambda_snk1=lambda_lame+mu',
        material_interface='shared-node bonded P1 tetrahedral mesh; no slip/debonding',
        reaction_force_gt_available=False,admitted_for_training=False,wall_seconds=time.time()-started,
        pending=['two successive mesh refinements','time-step halving','full surface clearance and contact force units',
                 'energy balance and friction sensitivity','material identifiability and independent geometry split'],
        scope='Physical ABD/FEM contact prototype. Prescribed tool motion is not calibrated force. Uniform absolute E cannot be inferred from a static displacement-controlled shape without force evidence.')
    if diagnostics:
        result['contact_diagnostics'] = dict(status='exported_not_calibrated',tool_global_vertex_offset=tool_offset,
            solid_global_vertex_offset=solid_offset,conversion='force=-raw_incremental_gradient/dt^2; barrier_energy=raw_incremental_energy/dt^2',
            source_reference_commit='cd1acc989b29f18d044cf4e449bee1576ac7d49f',
            max_action_reaction_relative_error=float(max(balance_trace)),
            free_node_relative_residual_p95=float(np.quantile(residual_trace,.95)),
            peak_tool_contact_force_n=float(np.linalg.norm(force_trace,axis=1).max()),
            energy_scope='Solid elastic/kinetic and barrier energy only; endpoint contact work excludes actuator/tool inertia/ABD elastic energy; not a complete energy balance certification')
    atomic(output/'result.json',result); print(json.dumps(result),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--case',required=True); p.add_argument('--output',required=True)
    p.add_argument('--control-output',required=True); a = p.parse_args()
    run(json.loads(Path(a.case).read_text()),Path(a.output),Path(a.control_output))
