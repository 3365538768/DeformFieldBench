"""All saved-frame intersections for our affine cuboid tool and tetrahedral solid.

Triangle/box SAT catches edge/face crossings missed by vertex-inside tests.
Containment uses the tool center in the complete deformed tet volume. This is
specific to this convex cuboid tool; it is not a generic tool collision checker,
minimum-clearance certificate, or a continuous-time CCD proof.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from engine_fixture import atomic, wait_if_paused
from physical_renderer import exterior_triangles

ROOT=Path(__file__).resolve().parents[1]


def triangles_inside_box(triangles, lower, upper, tolerance=1e-8):
    center=(lower+upper)/2
    half=(upper-lower)/2-tolerance
    if (half<=0).any():raise ValueError('Box tolerance exceeds dimensions')
    v=np.asarray(triangles)-center
    edges=np.roll(v,-1,axis=1)-v
    axes=[np.broadcast_to(axis,(len(v),3)) for axis in np.eye(3)]
    axes.append(np.cross(edges[:,0],edges[:,1]))
    axes.extend(np.cross(edges[:,e],axis) for e in range(3) for axis in np.eye(3))
    overlap=np.ones(len(v),dtype=bool)
    for axis in axes:
        norm=np.linalg.norm(axis,axis=1)
        valid=norm>1e-15
        unit=axis/np.maximum(norm[:,None],1e-30)
        projected=np.einsum('nvc,nc->nv',v,unit)
        radius=np.abs(unit)@half
        overlap &= ~valid | ((projected.min(1)<=radius)&(projected.max(1)>=-radius))
    return overlap


def point_in_tet_volume(point, positions, tets):
    if ((point<positions.min(0))|(point>positions.max(0))).any():return False
    tet=positions[tets]
    basis=(tet[:,1:]-tet[:,:1]).transpose(0,2,1)
    bary=np.linalg.solve(basis,(point-tet[:,0])[...,None])[...,0]
    weights=np.column_stack((1-bary.sum(1),bary))
    return bool(((weights>=-1e-10)&(weights<=1+1e-10)).all(1).any())


def frame_overlap(rest_tool, actual_tool, positions, tets, faces):
    design=np.column_stack((rest_tool,np.ones(len(rest_tool))))
    affine=np.linalg.lstsq(design,actual_tool,rcond=None)[0]
    error=float(np.abs(design@affine-actual_tool).max())
    if error>1e-10:raise ValueError('Tool is not an affine image of the registered cuboid')
    local=(positions-affine[3])@np.linalg.inv(affine[:3])
    intersections=triangles_inside_box(local[faces],rest_tool.min(0),rest_tool.max(0))
    # In the no-intersection case, a center inside detects fully contained tool.
    center=rest_tool.mean(0)@affine[:3]+affine[3]
    contained=point_in_tet_volume(center,positions,tets) if not intersections.any() else False
    return int(intersections.sum()),contained


def main():
    os.chdir(ROOT);output=ROOT/'runs/g3_contact_surface_audit';output.mkdir(parents=True,exist_ok=True)
    sources=[]
    for filename in ['G3_CONTACT_PREFLIGHT.json','G3_CONTACT_FORCE_AUDIT.json']:
        report=json.loads((ROOT/'docs'/filename).read_text())
        if report['status']!='complete':raise ValueError('Predecessor is incomplete')
        sources.extend(report['rows'])
    state=dict(status='running',completed=0,total=len(sources),rows=[],admitted_for_training=False,scope=__doc__)
    def update(**kwargs):
        state.update(kwargs,updated_unix=time.time());atomic(ROOT/'docs/G3_CONTACT_SURFACE_AUDIT.json',state)
        atomic(output/'runtime.json',dict(phase=state['status'],completed=state['completed'],total=state['total'],
            detail='全保存帧：物体三角形/压头盒体交叠与工具中心包含检查',updated_unix=time.time()))
    update()
    try:
        for row in sources:
            wait_if_paused(output)
            path=ROOT/row['path']/'trajectory.npz'
            with np.load(path,allow_pickle=False) as z:
                rest=z['rest_positions_m'];tets=z['tetrahedra'];trace=z['displacement_m']
                rest_tool=z['tool_rest_positions_m'];tools=z['tool_positions_m']
            faces=exterior_triangles(tets);bad=[];max_triangles=0
            for i in range(len(trace)+1):
                wait_if_paused(output)
                positions=rest if i==0 else rest+trace[i-1]
                tool=rest_tool if i==0 else tools[i-1]
                count,contained=frame_overlap(rest_tool,tool,positions,tets,faces)
                max_triangles=max(max_triangles,count)
                if count or contained:bad.append(dict(frame=i,overlapping_surface_triangles=count,tool_center_inside_solid=contained))
                if i%100==0:update(current_case=str(path.parent.relative_to(ROOT)),current_frame=i,current_frames=len(trace)+1)
            state['rows'].append(dict(path=str(path.parent.relative_to(ROOT)),trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                saved_frames=len(trace)+1,penetration_tolerance_m=1e-8,max_overlapping_triangles=max_triangles,
                offending_frames=bad,passed=not bad))
            update(completed=len(state['rows']))
        passed=all(x['passed'] for x in state['rows'])
        update(status='complete',saved_frame_intersection_screen_passed=passed,
               next_gate='Minimum surface separation/continuous-time proof, complete energy/actuator and friction audits remain; no automatic training')
        atomic(output/'status.json',dict(status='complete',reason='surface_intersection_screen_passed_admission_pending' if passed else 'surface_intersection_detected_scale_up_blocked'))
    except Exception as exc:
        update(status='failed',error_type=type(exc).__name__)
        atomic(output/'status.json',dict(status='failed',reason=type(exc).__name__));raise


if __name__=='__main__':main()
