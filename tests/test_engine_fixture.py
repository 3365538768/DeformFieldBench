"""Physical/provenance guards for the controlled cross-engine data pilot."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

SCRIPTS=Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from engine_fixture import block_mesh, tip_weights, provenance
from run_cross_engine_pilot import healthy


def test_common_mesh_volume_and_consistent_surface_resultant():
    dims=[.04,.02,.03];x,tets=block_mesh([2,1,1],dims)
    volumes=np.linalg.det(np.stack([x[tets[:,i]]-x[tets[:,0]] for i in (1,2,3)],-1))/6
    assert np.all(volumes>0)
    np.testing.assert_allclose(volumes.sum(),np.prod(dims),rtol=1e-12)
    w=tip_weights(x,tets,dims[0]);tip=np.isclose(x[:,0],dims[0])
    np.testing.assert_allclose(w.sum(),1.,rtol=1e-12)
    assert np.all(w[~tip]==0)
    # Uniform per-vertex forces are not the consistent P1 traction load.
    np.testing.assert_allclose(np.sort(w[tip]),[1/6,1/6,1/3,1/3])
    np.testing.assert_allclose(w@x,[dims[0],dims[1]/2,dims[2]/2])


def test_health_gate_rejects_unconverged_or_inverted_engine_results():
    case=dict(dimensions_m=[.04,.02,.02],cells=[8,4,4],axis=1,force_n=.0002)
    good=dict(case=case,completed=True,fixed_vertex_max_error_m=0.,min_jacobian=.999,
        max_displacement_gradient=.001,tip_displacement_m=[0.,1e-5,0.],
        assembled_resultant_n=[0.,.0002,0.],final_max_speed_m_s=0.,
        last20_displacement_change_m=1e-10,actual_relative_residual=1e-4)
    assert healthy(good)
    for key,value in [('min_jacobian',-.1),('fixed_vertex_max_error_m',1e-4),
                      ('actual_relative_residual',.01),('final_max_speed_m_s',.01),
                      ('max_displacement_gradient',.1),('completed',False)]:
        assert not healthy({**good,key:value})
    assert not healthy({**good,'assembled_resultant_n':[0.,.0003,0.]})
    assert not healthy({**good,'last20_displacement_change_m':1e-6})


def test_response_pair_provenance_tracks_geometry_force_and_boundary_protocol():
    case=dict(dimensions_m=[.04,.02,.02],cells=[8,4,4],axis=1,force_n=.0002)
    first=provenance(case)
    assert first==provenance(dict(case))
    assert first['fixture_geometry_sha256']!=provenance({**case,'cells':[16,8,8]})['fixture_geometry_sha256']
    assert first['load_history_sha256']!=provenance({**case,'axis':0})['load_history_sha256']
    assert first['load_history_sha256']!=provenance({**case,'force_n':.0004})['load_history_sha256']
