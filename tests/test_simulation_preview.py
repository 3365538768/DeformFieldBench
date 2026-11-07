"""Preview must show actual saved frames and reject unregistered paths."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('simulation_preview',Path(__file__).parents[1]/'monitor/simulation_remote.py')
preview=importlib.util.module_from_spec(spec);spec.loader.exec_module(preview)


def test_preview_uses_real_trajectory_and_manifest_allowlist(tmp_path):
    identifier='uipc_'+'a'*20;relative='data/calibration/g2_cases/'+identifier
    folder=tmp_path/relative;(folder/'world').mkdir(parents=True)
    rest=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    displacement=rest*.001
    np.savez(folder/'field.npz',rest_positions_m=rest,displacement_m=displacement,tetrahedra=np.array([[0,1,2,3]]))
    trace=np.stack([displacement*.3,displacement*.8,displacement])
    np.savez(folder/'world/trajectory.npz',displacement_m=trace,dt_seconds=.02)
    manifest=tmp_path/'data/manifests/g2_engine_response.json';manifest.parent.mkdir(parents=True)
    row=dict(id=identifier,path=relative,engine='uipc',fixture=0,split='train',material_pair='p',axis=1,
             case={},engine_version='0.0.29',response_provenance={})
    manifest.write_text(json.dumps({'rows':[row]}))
    result=preview.load_preview({'case':identifier},tmp_path)
    assert result['kind']=='actual_solver_trajectory'
    assert result['frame_times_s']==[0.,.02,.04,.06]
    np.testing.assert_allclose(result['displacement_frames_m'][1:],trace)
    assert len(result['surface_triangles'])==4
    partial=tmp_path/'docs/G2_SIMULATION_CASES.json';partial.parent.mkdir()
    partial.write_text(json.dumps({'rows':[row],'pair_validation':'pending'}));manifest.unlink()
    assert preview.load_preview({'case':identifier},tmp_path)['pair_validation']=='pending'
    manifest.write_text(json.dumps({'rows':[row]}))
    for identifier in ('../../etc/passwd','uipc_'+'b'*20):
        with pytest.raises(ValueError):preview.load_preview({'case':identifier},tmp_path)
    row['path']='outside';manifest.write_text(json.dumps({'rows':[row]}))
    with pytest.raises(ValueError,match='path'):preview.load_preview({'case':row['id']},tmp_path)


def test_warp_preview_does_not_fabricate_a_dynamic_trajectory(tmp_path):
    identifier='warp_'+'a'*20;relative='data/calibration/g2_cases/'+identifier
    folder=tmp_path/relative;folder.mkdir(parents=True)
    rest=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    np.savez(folder/'field.npz',rest_positions_m=rest,displacement_m=rest*.001,tetrahedra=np.array([[0,1,2,3]]))
    manifest=tmp_path/'data/manifests/g2_engine_response.json';manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'rows':[dict(id=identifier,path=relative,engine='warp',fixture=0,split='val',
        material_pair='p',axis=2,case={},engine_version='1.11.1',response_provenance={})]}))
    result=preview.load_preview({'case':identifier},tmp_path)
    assert result['kind']=='static_solution' and result['frame_times_s']==[None]
    assert len(result['displacement_frames_m'])==1
