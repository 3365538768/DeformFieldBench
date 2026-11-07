"""Clamped elastic cantilever: measured SI deflection and mesh sensitivity.

Fixed geometry/load, E and mesh refinement are controlled. Linear tetrahedra
may lock in bending: a failed analytic gate must block bulk dataset generation.
"""
import argparse
import itertools
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from uipc import Engine, Scene, Timer, World, builtin, view
from uipc.constitution import ElasticModuli, FiniteElementExternalForce, StableNeoHookean
from uipc.core import FiniteElementStateAccessorFeature
from uipc.geometry import label_surface, label_triangle_orient, tetmesh


def mesh_beam(nx, ny, nz):
    # Six consistently oriented tets per rectangular cell, sharing face diagonals.
    shape = (nx+1, ny+1, nz+1)
    points = np.stack(np.meshgrid(np.linspace(0,.2,nx+1), np.linspace(0,.02,ny+1),
                                 np.linspace(0,.02,nz+1), indexing='ij'), -1).reshape(-1,3)
    tets = []
    for cell in itertools.product(range(nx),range(ny),range(nz)):
        for permutation in itertools.permutations(range(3)):
            indices = [np.array(cell)]
            for axis in permutation:
                vertex = indices[-1].copy();vertex[axis] += 1;indices.append(vertex)
            tet = [int(np.ravel_multi_index(tuple(i),shape)) for i in indices]
            if np.linalg.det((points[tet[1:]]-points[tet[0]]).T)<0:
                tet[1],tet[2]=tet[2],tet[1]
            tets.append(tet)
    return points, np.array(tets,dtype=np.int32)


def run_case(workspace, nx, young, mapping):
    Timer.disable_all();workspace.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    rest,tets=mesh_beam(nx,nx//5,nx//5)
    fixed=np.isclose(rest[:,0],0);tip=np.isclose(rest[:,0],.2)
    mesh=tetmesh(rest,tets);label_surface(mesh);label_triangle_orient(mesh)
    StableNeoHookean().apply_to(mesh,ElasticModuli.youngs_poisson(young,.3),mass_density=1000.)
    native_coefficients={name:float(np.array(view(mesh.tetrahedra().find(name))).flat[0]) for name in ('mu','lambda')}
    if mapping=='snk1_physical':
        # Explicit experimental correction from upstream's current SNK1 formula.
        # Keep native baseline results: installed 0.0.29 still uses the old map.
        mu=young/(2*(1+.3));lam=young*.3/((1+.3)*(1-2*.3))
        view(mesh.tetrahedra().find('mu'))[:]=mu
        view(mesh.tetrahedra().find('lambda'))[:]=lam+mu
    used_coefficients={name:float(np.array(view(mesh.tetrahedra().find(name))).flat[0]) for name in ('mu','lambda')}
    view(mesh.vertices().find(builtin.is_fixed))[fixed]=1
    FiniteElementExternalForce().apply_to(mesh,np.zeros(3))
    engine=Engine('cuda',str(workspace));world=World(engine)
    cfg=Scene.default_config();cfg['gravity']=[[0.],[0.],[0.]];cfg['contact']['enable']=False
    cfg['dt']=.02;cfg['newton']['velocity_tol']=1e-7
    cfg['newton']['semi_implicit']['enable']=False;cfg['linear_system']['tol_rate']=1e-8
    scene=Scene(cfg);obj=scene.objects().create('beam');obj.geometries().create(mesh)
    force_n=.001
    def animate(info):
        geom=info.geo_slots()[0].geometry()
        forces=view(geom.vertices().find('external_force'));active=view(geom.vertices().find('external_force/is_constrained'))
        forces[:]=0;active[:]=0;active[tip]=1
        forces[tip,1,0]=-force_n/tip.sum()
    scene.animator().insert(obj,animate);world.init(scene)
    feature=world.features().find(FiniteElementStateAccessorFeature)
    trace=[];completed=True
    for step in range(200):
        world.advance();world.retrieve()
        state=feature.create_geometry();state.vertices().create(builtin.position,np.zeros(3));state.vertices().create(builtin.velocity,np.zeros(3))
        feature.copy_to(state)
        x=np.array(view(state.vertices().find(builtin.position))).reshape(-1,3)
        v=np.array(view(state.vertices().find(builtin.velocity))).reshape(-1,3)
        trace.append([float(-(x[tip,1]-rest[tip,1]).mean()),float(np.linalg.norm(v,axis=1).max())])
        completed &= bool(engine.frame_stats()['completed'])
    rest_basis=np.stack([rest[tets[:,i]]-rest[tets[:,0]] for i in range(1,4)],-1)
    basis=np.stack([x[tets[:,i]]-x[tets[:,0]] for i in range(1,4)],-1)
    jacobian=np.linalg.det(basis)/np.linalg.det(rest_basis)
    inertia=.02*.02**3/12
    analytic=force_n*.2**3/(3*young*inertia)
    displacement=trace[-1][0]
    steady_range=(max(t[0] for t in trace[-20:])-min(t[0] for t in trace[-20:]))/analytic
    fixed_error=float(np.abs(x[fixed]-rest[fixed]).max())
    np.savez_compressed(workspace/'trajectory.npz',tip_trace=np.array(trace),rest_positions_m=rest,
                        final_positions_m=x,tetrahedra=tets,dt_seconds=.02)
    result={'young_pa':young,'poisson':.3,'density_kg_m3':1000.,'force_n':force_n,'dt_seconds':.02,
            'frames':200,'mesh_cells':[nx,nx//5,nx//5],'vertices':len(rest),'tetrahedra':len(tets),
            'length_m':.2,'cross_section_m':[.02,.02],'fixed_boundary':'all x=0 vertices, exact is_fixed',
            'tip_deflection_m':displacement,'euler_bernoulli_deflection_m':analytic,
            'relative_analytic_error':abs(displacement/analytic-1),'last20_deflection_range_over_analytic':steady_range,
            'final_max_speed_m_s':trace[-1][1],'fixed_vertex_max_error_m':fixed_error,
            'min_final_jacobian':float(jacobian.min()),'completed':completed,'wall_seconds':time.perf_counter()-started}
    result['native_energy_coefficients_pa']=native_coefficients
    result['used_energy_coefficients_pa']=used_coefficients
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',default='data/calibration/uipc_beam')
    p.add_argument('--output',default='docs/UIPC_BEAM_CALIBRATION.json')
    p.add_argument('--mapping',choices=['native','snk1_physical'],default='native')
    p.add_argument('--resolutions',default='10,20');a=p.parse_args()
    resolutions=[int(n) for n in a.resolutions.split(',')]
    if len(resolutions)!=2 or any(n<5 or n%5 for n in resolutions) or resolutions[0]>=resolutions[1]:
        p.error('Two increasing multiples of 5 required')
    cases=[]
    for nx,young in [(resolutions[0],1e6),(resolutions[1],1e6),(resolutions[1],2e6)]:
        case=run_case(Path(a.workspace)/f'{a.mapping}_n{nx}_E{young:g}',nx,young,a.mapping);cases.append(case)
        print(json.dumps(case),flush=True)
    coarse,fine,stiff=cases
    mesh_change=abs(fine['tip_deflection_m']-coarse['tip_deflection_m'])/fine['euler_bernoulli_deflection_m']
    e_ratio=fine['tip_deflection_m']/stiff['tip_deflection_m']
    gates={'finite_positive_volume':all(np.isfinite(c['tip_deflection_m']) and c['min_final_jacobian']>0 and c['completed'] for c in cases),
           'fixed_boundary':all(c['fixed_vertex_max_error_m']<1e-10 for c in cases),
           'settled':all(c['last20_deflection_range_over_analytic']<.01 for c in cases),
           'fine_analytic_within_20_percent':fine['relative_analytic_error']<.2,
           'mesh_change_within_10_percent':mesh_change<.1,
           'inverse_E_within_5_percent':abs(e_ratio/2-1)<.05}
    result={'status':'passed' if all(gates.values()) else 'failed','engine':'libuipc','version':importlib.metadata.version('pyuipc'),
            'mapping':a.mapping,
            'cases':cases,'gates':gates,'mesh_change_over_analytic':mesh_change,'inverse_E_response_ratio':e_ratio,
            'scope':'Small-strain clamped beam under known end load. Euler-Bernoulli reference excludes shear; aspect ratio 10. Gate thresholds set before execution. Does not validate contact/mixed materials.',
            'material_reference':'https://github.com/spiriMirror/libuipc/blob/main/docs/specification/constitutions/stable_neo_hookean.md'}
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(output)
    print('BEAM_RESULT',result['status'],flush=True)
    if result['status']!='passed':raise SystemExit(2)


if __name__=='__main__':main()
