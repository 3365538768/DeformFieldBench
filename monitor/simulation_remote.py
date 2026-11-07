"""Read-only, manifest-allowlisted simulation preview from A800.

No arbitrary paths, pickle loading, or solver execution. Frames are selected
from the saved UIPC trajectory; static Warp solutions have one real frame.
"""
import itertools
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT=Path('/mnt/afs_engine/linrui/rsi_real_test')


def load_preview(request, root=ROOT):
    if set(request)!={'case'} or not isinstance(request['case'],str):
        raise ValueError('Expected one manifest case ID')
    manifest=root/'data/manifests/g2_engine_response.json'
    partial=root/'docs/G2_SIMULATION_CASES.json'
    if not manifest.exists() and not partial.exists():
        return {'status':'preparing','cases':[]}
    index=json.loads(manifest.read_text()) if manifest.exists() else json.loads(partial.read_text())
    rows=[{**r,'source':'G2'} for r in index['rows']]
    pair_validation='passed' if manifest.exists() else index.get('pair_validation','pending')
    visibility=root/'docs/G3_FORCE_VISIBILITY.json'
    if visibility.exists():
        for item in json.loads(visibility.read_text()).get('results',[]):
            identifier=item['id']
            if not re.fullmatch(r'uipc_[0-9a-f]{20}',identifier):
                continue
            relative='data/calibration/g3_visibility/'+identifier
            receipt=root/relative/'result.json'
            if not receipt.exists():
                continue
            result=json.loads(receipt.read_text())
            rows.append({**result,'id':identifier,'path':relative,'source':'G3v',
                'fixture':'载荷审计','split':'calibration','material_pair':identifier,
                'axis':result['case']['axis'],'ideal_max_motion_px':item['ideal_max_motion_px'],
                'healthy':item['healthy']})
    convergence=root/'docs/G3_MESH_CONVERGENCE.json'
    if convergence.exists():
        rows.extend({**r,'source':'G3c'} for r in json.loads(convergence.read_text()).get('rows',[]))
    efficiency=root/'docs/G3_INTERACTION_EFFICIENCY.json'
    if efficiency.exists():
        rows.extend({**r,'source':'G3t'} for r in json.loads(efficiency.read_text()).get('rows',[]))
    catalog=[dict(id=r['id'],engine=r['engine'],fixture=r['fixture'],split=r['split'],
        material_pair=r['material_pair'],axis=r['axis'],case=r['case'],version=r['engine_version'],source=r['source']) for r in rows]
    identifier=request['case']
    if not identifier:
        return {'status':'ready','cases':catalog,'pair_validation':pair_validation,
                'note':'Controlled SI blocks; no contact or mixed materials yet.'}
    if not re.fullmatch(r'(warp|uipc)_[0-9a-f]{20}',identifier):
        raise ValueError('Invalid case ID')
    row=next((r for r in rows if r['id']==identifier),None)
    if row is None:
        raise ValueError('Case not registered in response manifest')
    path=(root/row['path']).resolve()
    allowed=(root/{'G2':'data/calibration/g2_cases','G3v':'data/calibration/g3_visibility',
                   'G3c':'data/calibration/g3_convergence','G3t':'data/calibration/g3_interaction'}[row['source']]).resolve()
    if not path.is_relative_to(allowed):
        raise ValueError('Invalid registered case path')
    with np.load(path/'field.npz',allow_pickle=False) as data:
        rest=data['rest_positions_m'];displacement=data['displacement_m'];tets=data['tetrahedra']
    if rest.shape!=displacement.shape or rest.ndim!=2 or rest.shape[1]!=3 or len(rest)>100000 or len(tets)>500000:
        raise ValueError('Preview geometry exceeds supported shape/size')
    all_faces=np.concatenate([tets[:,indices] for indices in ([0,1,2],[0,1,3],[0,2,3],[1,2,3])])
    unique,counts=np.unique(np.sort(all_faces,axis=1),axis=0,return_counts=True)
    boundary=unique[counts==1]
    surface=np.unique(boundary);volume_vertices=len(rest)
    if len(surface)>10000:raise ValueError('Surface preview exceeds supported size')
    inverse=np.full(len(rest),-1,dtype=int);inverse[surface]=np.arange(len(surface))
    boundary=inverse[boundary].tolist()
    frames=displacement[None];times=[None];kind='static_solution'
    trajectory=path/'world/trajectory.npz'
    if row['engine']=='uipc' and trajectory.exists():
        with np.load(trajectory,allow_pickle=False) as data:
            trace=data['displacement_m'];dt=float(data['dt_seconds'])
        if trace.ndim!=3 or trace.shape[1:]!=rest.shape or len(trace)>2000:
            raise ValueError('Invalid saved trajectory')
        indices=np.unique(np.linspace(0,len(trace)-1,min(25,len(trace))).round().astype(int))
        frames=np.concatenate([np.zeros_like(displacement)[None],trace[indices]],axis=0)
        times=[0.]+((indices+1)*dt).tolist();kind='actual_solver_trajectory'
        np.testing.assert_allclose(trace[-1],displacement,rtol=1e-7,atol=1e-12)
    if not np.isfinite(rest).all() or not np.isfinite(frames).all():
        raise ValueError('Nonfinite simulation preview')
    # Interior FEM vertices never contribute to the surface display. Remove them
    # exactly, preserving all surface nodes and actual saved displacements.
    rest=rest[surface];frames=frames[:,surface]
    return dict(status='ready',id=identifier,engine=row['engine'],version=row['engine_version'],
        case=row['case'],fixture=row['fixture'],split=row['split'],kind=kind,
        pair_validation=pair_validation if row['source']=='G2' else 'not_applicable',source=row['source'],
        ideal_max_motion_px=row.get('ideal_max_motion_px'),healthy=row.get('healthy'),
        volume_vertices=volume_vertices,preview_vertices=len(surface),preview_geometry='exact exterior surface',
        rest_positions_m=rest.tolist(),surface_triangles=boundary,displacement_frames_m=frames.tolist(),
        frame_times_s=times,scalar='displacement_magnitude_m',
        max_displacement_m=float(np.linalg.norm(frames,axis=2).max()),
        health={key:row.get(key) for key in ('min_jacobian','max_displacement_gradient',
            'fixed_vertex_max_error_m','actual_relative_residual','final_max_speed_m_s',
            'last20_displacement_change_m','tip_displacement_m','wall_seconds')},
        response_provenance=row['response_provenance'])


if __name__=='__main__':
    try:
        print(json.dumps(load_preview(json.loads(sys.stdin.read())),allow_nan=False,separators=(',',':')))
    except Exception as exc:
        print(json.dumps({'status':'error','error':type(exc).__name__}))
        raise SystemExit(1)
