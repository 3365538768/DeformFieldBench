"""Render the saved deformable body and its actual depth-tested 3D tool."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from engine_fixture import atomic, digest
from interaction_protocol import select_motion_frames
from physical_renderer import correspondence_flow, exterior_triangles, rasterize


def render(root, source, result):
    with np.load(source/'trajectory.npz', allow_pickle=False) as z:
        rest=z['rest_positions_m'];tets=z['tetrahedra'];trace=z['displacement_m']
        tool_rest=z['tool_rest_positions_m'];tool_trace=z['tool_positions_m'];tool_tets=z['tool_tetrahedra'];dt=float(z['dt_seconds'])
    camera=dict(name='3D tool / oblique',yaw=.45,pitch=.65,image_size=256,
                field_of_view_m=.07,center_m=[.02,.0,.0])
    size=camera['image_size']
    indices,sampling=select_motion_frames(trace,dt,size,camera['field_of_view_m'])
    frames=np.concatenate((np.zeros_like(rest)[None],trace))[indices]
    tools=np.concatenate((tool_rest[None],tool_trace))[indices];times=indices*dt
    spec=dict(case=result['case'],solver_sha256=result['source_sha256'],
        trajectory_sha256=hashlib.sha256((source/'trajectory.npz').read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),camera=camera,
        rasterizer_sha256=hashlib.sha256((root/'scripts/physical_renderer.py').read_bytes()).hexdigest())
    identifier='clip_'+digest(spec)[:20];folder=root/'data/rendered/g3_contact_preflight'/identifier;folder.mkdir(parents=True,exist_ok=True)
    faces=exterior_triangles(tets);tool_faces=exterior_triangles(tool_tets)
    peak=int(np.argmax(np.linalg.norm(frames,axis=-1).max(axis=1)))
    images=[];visibility=[];body_rasters=[]
    for i,(displacement,tool_positions) in enumerate(zip(frames,tools)):
        body=rasterize(rest,rest+displacement,faces,camera,17)
        tool=rasterize(tool_rest,tool_positions,tool_faces,camera,59)
        front=tool['mask']&(tool['depth']>body['depth'])
        rgb=body['rgb'].copy();rgb[front]=[174,154,133]
        image=Image.fromarray(rgb);image.save(folder/f'frame_{i:03d}.png');images.append(image)
        visibility.append(body['mask']&~front);body_rasters.append(body)
    flow,valid=correspondence_flow(body_rasters[0],body_rasters[peak],faces,rest+frames[peak],camera)
    yy,xx=np.mgrid[:size,:size]
    u=np.clip(np.floor(xx+.5+flow[...,0]).astype(int),0,size-1);v=np.clip(np.floor(yy+.5+flow[...,1]).astype(int),0,size-1)
    valid &= visibility[0]&visibility[peak][v,u]
    if not valid.any():raise ValueError('No visible body correspondence in contact probe')
    magnitude=np.linalg.norm(flow[valid],axis=1)
    images[0].save(folder/'preview.gif',save_all=True,append_images=images[1:],
        duration=[max(20,round(float(t)*1000/10)*10) for t in np.diff(times)]+[400],loop=0)
    images[-1].save(folder/'final.png')
    contact=Image.new('RGB',(size*3,size+24),(15,22,32));draw=ImageDraw.Draw(contact)
    for i,index in enumerate((0,peak,len(images)-1)):
        contact.paste(images[index],(i*size,24));draw.text((i*size+5,5),['Initial / actual 3D tool','Peak / 1x deformation','Released / true timestamps'][i],fill='white')
    contact.save(folder/'contact_sheet.png')
    np.savez_compressed(folder/'flow.npz',flow_pixels=flow,valid=valid,
        visible_body_masks=np.asarray(visibility),frame_times_s=times)
    metrics=dict(visible_correspondence_pixels=int(valid.sum()),visible_fraction=float(valid.sum()/max(visibility[0].sum(),1)),
        observed_object_fraction=float(visibility[peak].sum()/max(body_rasters[peak]['mask'].sum(),1)),
        final_motion_p50_px=float(np.quantile(magnitude,.5)),final_motion_p95_px=float(np.quantile(magnitude,.95)),
        final_motion_max_px=float(magnitude.max()),motion_reference='initial_to_peak_visible_correspondence')
    row=dict(id=identifier,path=str(folder.relative_to(root)),case_id=source.name,case=result['case'],
        young_pa=None,poisson=result['case']['poisson'],force_n=None,camera=camera,frame_times_s=times.tolist(),
        sampling=sampling,metrics=metrics,specification=spec,physical_tool_contact=True,
        occlusion_variant='physical_tool',interaction='physical_punch_release',deformation_amplification=1.,
        eligible_for_video_pilot=False,pending_quality_checks=result['pending'])
    atomic(folder/'result.json',row)
    return row
