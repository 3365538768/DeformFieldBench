"""Render only admitted converged cases, measuring visible flow without amplification."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from engine_fixture import atomic, digest, wait_if_paused
from interaction_protocol import select_motion_frames
from physical_renderer import correspondence_flow, exterior_triangles, overlay_occluder, rasterize
from run_cross_engine_pilot import ROOT, solve


def render_clip(row, camera, texture_seed, output, quality, occlusion='clean',raster_cache=None):
    source=ROOT/row['path']
    with np.load(source/'field.npz',allow_pickle=False) as field:
        rest=field['rest_positions_m'];triangles=exterior_triangles(field['tetrahedra'])
    surface=np.unique(triangles)
    inverse=np.full(len(rest),-1,dtype=int);inverse[surface]=np.arange(len(surface))
    triangles=inverse[triangles];rest=rest[surface]
    trajectory_path=source/'world/trajectory.npz'
    with trajectory_path.open('rb') as stream:
        trajectory_sha256=hashlib.file_digest(stream,'sha256').hexdigest()
    with np.load(trajectory_path,allow_pickle=False) as trajectory:
        trace=trajectory['displacement_m'][:,surface];dt=float(trajectory['dt_seconds'])
    selected,sampling=select_motion_frames(trace,dt,camera['image_size'],camera['field_of_view_m'])
    frames=np.concatenate((np.zeros_like(rest)[None],trace),axis=0)[selected]
    times=selected*dt
    peak_index=int(np.argmax(np.linalg.norm(frames,axis=2).max(axis=1)))
    spec=dict(case_id=row['id'],camera=camera,texture_seed=texture_seed,source_sha256=row['solver_source_sha256'],
        renderer_sha256=hashlib.sha256((ROOT/'scripts/physical_renderer.py').read_bytes()).hexdigest(),
        pipeline_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        trajectory_sha256=trajectory_sha256,frame_indices_with_initial_rest=selected.tolist(),dt_seconds=dt,
        occlusion_variant=occlusion,occlusion_kind='none' if occlusion=='clean' else 'paired image-space augmentation; not physical contact',
        screening={key:quality[key] for key in ('minimum_motion_px','maximum_motion_px')})
    identifier='clip_'+digest(spec)[:20];folder=output/identifier;folder.mkdir(parents=True,exist_ok=True)
    images=[];first=target=None;reference_mask=None
    if raster_cache is None:raster_cache={}
    for i,displacement in enumerate(frames):
        if i not in raster_cache:raster_cache[i]=rasterize(rest,rest+displacement,triangles,camera,texture_seed)
        result=dict(raster_cache[i])
        if reference_mask is None:reference_mask=result['mask']
        result['rgb'],result['occlusion_mask']=overlay_occluder(result['rgb'],reference_mask,occlusion)
        image=Image.fromarray(result['rgb']);image.save(folder/f'frame_{i:03d}.png');images.append(image)
        if i==0:first=result
        if i==peak_index:target=result
    flow,valid=correspondence_flow(first,target,triangles,rest+frames[peak_index],camera)
    yy,xx=np.mgrid[:valid.shape[0],:valid.shape[1]]
    u=np.clip(np.floor(xx+.5+flow[...,0]).astype(int),0,valid.shape[1]-1)
    v=np.clip(np.floor(yy+.5+flow[...,1]).astype(int),0,valid.shape[0]-1)
    valid&=~first['occlusion_mask']&~target['occlusion_mask'][v,u]
    images[-1].save(folder/'final.png')
    if valid.sum()==0:raise ValueError('No visible correspondence pixels')
    magnitude=np.linalg.norm(flow[valid],axis=1)
    mask_union=np.count_nonzero(first['mask']|target['mask'])
    visible_object=first['mask']&~first['occlusion_mask']
    metrics=dict(visible_correspondence_pixels=int(valid.sum()),foreground_pixels=int(visible_object.sum()),
        amodal_foreground_pixels=int(first['mask'].sum()),observed_object_fraction=float(visible_object.sum()/max(first['mask'].sum(),1)),
        visible_fraction=float(valid.sum()/max(visible_object.sum(),1)),motion_reference='initial_to_peak_visible_correspondence',
        final_motion_p50_px=float(np.quantile(magnitude,.5)),final_motion_p95_px=float(np.quantile(magnitude,.95)),
        final_motion_max_px=float(magnitude.max()),silhouette_iou=float(np.count_nonzero(first['mask']&target['mask'])/max(mask_union,1)),
        foreground_fraction=float(first['mask'].mean()))
    # GIF preserves solver timestamps; PNG frames and float flow remain authoritative.
    durations=[max(20,round(float(t)*1000/10)*10) for t in np.diff(times)]+[400]
    images[0].save(folder/'preview.gif',save_all=True,append_images=images[1:],duration=durations,loop=0)
    rgb_flow=np.zeros((*valid.shape,3),dtype=np.uint8)
    normalized=np.clip(np.linalg.norm(flow,axis=-1)/30,0,1)
    rgb_flow[...,0]=(255*normalized).astype(np.uint8)
    rgb_flow[...,1]=(255*(1-normalized)).astype(np.uint8);rgb_flow[~valid]=0
    contact=Image.new('RGB',(camera['image_size']*3,camera['image_size']+24),(15,22,32))
    for i,image in enumerate([images[0],images[peak_index],Image.fromarray(rgb_flow)]):contact.paste(image,(i*camera['image_size'],24))
    draw=ImageDraw.Draw(contact)
    for i,label in enumerate(['Initial RGB / 1x','Peak RGB / 1x','Visible displacement / 0-30 px']):draw.text((i*camera['image_size']+5,5),label,fill='white')
    contact.save(folder/'contact_sheet.png')
    np.savez_compressed(folder/'flow.npz',flow_pixels=flow,valid=valid,mask_initial_amodal=first['mask'],
        mask_target_amodal=target['mask'],occlusion_mask_initial=first['occlusion_mask'],occlusion_mask_target=target['occlusion_mask'])
    receipt=dict(id=identifier,path=str(folder.relative_to(ROOT)),case_id=row['id'],source_path=row['path'],
        young_pa=row['case']['young_pa'],force_n=row['case']['force_n'],poisson=row['case']['poisson'],
        case=row['case'],camera=camera,texture_seed=texture_seed,frame_times_s=times.tolist(),
        deformation_amplification=1.,metrics=metrics,specification=spec,sampling=sampling,occlusion_variant=occlusion,
        paired_group_id=digest({'case_id':row['id'],'camera':camera,'texture_seed':texture_seed}),
        peak_time_s=float(times[peak_index]),
        eligible_for_video_pilot=bool(metrics['final_motion_p95_px']>=quality['minimum_motion_px'] and metrics['final_motion_max_px']<=quality['maximum_motion_px']
            and metrics['visible_fraction']>=.5 and metrics['foreground_fraction']>=.05 and metrics['observed_object_fraction']>=.35))
    atomic(folder/'result.json',receipt)
    return receipt


def main():
    os.chdir(ROOT);output=ROOT/'runs/g3_physical_render';base=ROOT/'data/rendered/g3_pilot'
    cfg=json.loads((ROOT/'configs/g3_quality.json').read_text())
    convergence=json.loads((ROOT/'docs/G3_MESH_CONVERGENCE.json').read_text())
    groups=[g for g in convergence['groups'] if g['eligible_for_render']]
    if convergence['status']!='complete' or not groups:raise ValueError('No converged candidates; render scale-up held')
    by_id={r['id']:r for r in convergence['rows']}
    efficiency_path=ROOT/'docs/G3_INTERACTION_EFFICIENCY.json'
    efficiency=json.loads(efficiency_path.read_text()) if efficiency_path.exists() else {}
    dynamic_by_id={r['id']:r for r in efficiency.get('rows',[])}
    dynamic=[dynamic_by_id[g['selected_dynamic_case_id']] for g in efficiency.get('groups',[])
             if g.get('eligible_for_transient_supervision') and g.get('selected_dynamic_case_id') in dynamic_by_id]
    cameras=[dict(name=name,yaw=yaw,pitch=pitch,image_size=cfg['image_size'],field_of_view_m=cfg['field_of_view_m'],center_m=[.02,.01,.01])
             for name,yaw,pitch in [('XY',0.,0.),('oblique',.45,-.3)]]
    rows=[];started=time.time()
    report=dict(status='running',completed=0,total=len(groups)*len(cameras)*3+len(dynamic)*2,clips=rows,
        scope='Diagnostic textured blocks with actual SI motion, fixed orthographic cameras, no realistic scene/contact/noise. Flow is renderer correspondence with occlusion screening, not an estimated optical-flow score.',
        input_contract=dict(known_at_training=['metric geometry','camera','force history','boundary conditions'],
            unavailable_parameters_from_single_static_image=['absolute stiffness without load/scale calibration','density','damping'],
            deployment='Uncontrolled or static real video must expose uncertainty; absolute modulus is not guaranteed identifiable.'))
    def update():
        report.update(completed=len(rows),updated_unix=time.time(),wall_seconds=time.time()-started,
            detail=f'真实 SI 轨迹渲染 {len(rows)}/{report["total"]}；未放大 RGB、遮挡筛选对应光流、同材质坐标纹理。')
        atomic(ROOT/'docs/G3_RENDER_QUALITY.json',report)
        atomic(output/'runtime.json',dict(phase=report['status'],completed=len(rows),total=report['total'],updated_unix=time.time()))
    update()
    try:
        for group in groups:
            for camera in cameras:
                wait_if_paused(output)
                raster_cache={}
                for occlusion in ('clean','moderate','heavy'):
                    wait_if_paused(output)
                    rows.append(render_clip(by_id[group['selected_case_id']],camera,17,base,cfg,occlusion,raster_cache));update()
        for row in dynamic:
            raster_cache={}
            for occlusion in ('clean','moderate'):
                wait_if_paused(output)
                clip=render_clip(row,cameras[1],17,base,cfg,occlusion,raster_cache)
                clip['interaction']='load_hold_release';rows.append(clip);update()
        # Real solver check of the quasi-static force/stiffness scale ambiguity.
        template=next((r for r in by_id.values() if r['case']['young_pa']==12000 and r['case']['force_n']==.02 and r['case']['cells'][0]==32),None)
        if template:
            wait_if_paused(output)
            case={**template['case'],'young_pa':48000.,'force_n':.08}
            key,receipt=solve('uipc',case,ROOT/'data/calibration/g3_identifiability',timeout=cfg['solver_timeout_seconds'])
            with np.load(ROOT/template['path']/'field.npz',allow_pickle=False) as a, np.load(ROOT/'data/calibration/g3_identifiability'/key/'field.npz',allow_pickle=False) as b:
                error=float(np.linalg.norm(a['displacement_m']-b['displacement_m'])/np.linalg.norm(a['displacement_m']))
            report['quasistatic_identifiability']=dict(reference_case_id=template['id'],scaled_case_id=key,
                modulus_ratio=4.,force_ratio=4.,settled_field_relative_l2=error,
                numerical_health=dict(completed=receipt['completed'],min_jacobian=receipt['min_jacobian'],final_max_speed_m_s=receipt['final_max_speed_m_s']),
                scope='Settled displacement only. Transient dynamics may differ at fixed density. This does not prove impossibility for calibrated dynamic videos.')
        report.update(status='complete',accepted_clips=sum(r['eligible_for_video_pilot'] for r in rows),
            raster_reuse_factor=3,occlusion_scope='Paired image-space occluders with visibility masks. No 3D hand, gripper contact or friction simulation yet.')
        update();atomic(output/'status.json',dict(status='complete',reason='physical_render_audit_complete'))
    except Exception as exc:
        report.update(status='failed',error=str(exc));update()
        atomic(output/'status.json',dict(status='failed',reason=str(exc)));raise


if __name__=='__main__':main()
