"""Resume migrated rendering without changing the original renderer's provenance.

Only complete, registered clips with matching inputs and source hashes can be
reused. Invalid or missing artifacts fall back to the original renderer.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import render_physical_pilot as renderer
from engine_fixture import atomic, digest


def reuse_clip(receipt, row, camera, seed, output, quality, occlusion):
    if not receipt:
        return None
    spec=receipt.get('specification',{})
    expected={'case_id':row['id'],'camera':camera,'texture_seed':seed,
              'source_sha256':row['solver_source_sha256'],'occlusion_variant':occlusion,
              'screening':{k:quality[k] for k in ('minimum_motion_px','maximum_motion_px')}}
    if any(spec.get(k)!=v for k,v in expected.items()):
        return None
    hashes={'pipeline_sha256':Path(renderer.__file__),
            'renderer_sha256':renderer.ROOT/'scripts/physical_renderer.py',
            'trajectory_sha256':renderer.ROOT/row['path']/'world/trajectory.npz'}
    try:
        for key,path in hashes.items():
            with path.open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=spec.get(key):
                    return None
        identifier='clip_'+digest(spec)[:20]
        if receipt.get('id')!=identifier:
            return None
        folder=Path(output)/identifier
        if receipt.get('path')!=str(folder.relative_to(renderer.ROOT)):
            return None
        count=len(receipt['frame_times_s'])
        frames=sorted(folder.glob('frame_*.png'))
        if len(frames)!=count:
            return None
        for path in frames+[folder/'final.png',folder/'contact_sheet.png',folder/'preview.gif']:
            with Image.open(path) as image:
                image.verify()
        with np.load(folder/'flow.npz',allow_pickle=False) as flow:
            if flow['valid'].dtype!=np.bool_ or not flow['valid'].any() or not np.isfinite(flow['flow_pixels']).all():
                return None
    except (OSError,ValueError,KeyError,TypeError):
        return None
    return copy.deepcopy(receipt)


def main():
    root=renderer.ROOT
    path=root/'docs/G3_RENDER_QUALITY.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    registered={(x['case_id'],json.dumps(x['camera'],sort_keys=True),x['texture_seed'],x.get('occlusion_variant','clean')):x
                for x in previous.get('clips',[])}
    original=renderer.render_clip
    resumed={'status':'running','registered_before_resume':len(registered),'reused':0,'rendered':0}
    def render(row,camera,texture_seed,output,quality,occlusion='clean',raster_cache=None):
        key=(row['id'],json.dumps(camera,sort_keys=True),texture_seed,occlusion)
        cached=reuse_clip(registered.get(key),row,camera,texture_seed,output,quality,occlusion)
        resumed['reused' if cached else 'rendered']+=1
        result=cached or original(row,camera,texture_seed,output,quality,occlusion,raster_cache)
        atomic(root/'runs/g3_physical_render/migration_resume.json',resumed)
        return result
    renderer.render_clip=render
    try:
        renderer.main()
        resumed['status']='complete'
    except Exception as exc:
        resumed.update(status='failed',error_type=type(exc).__name__)
        raise
    finally:
        atomic(root/'runs/g3_physical_render/migration_resume.json',resumed)


if __name__=='__main__':
    main()
