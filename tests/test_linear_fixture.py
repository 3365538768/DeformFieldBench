import sys
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('scipy')
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from engine_fixture import block_mesh,tip_weights
from fieldbridge.linear_fixture import assemble,equilibrium,frequencies


def fixture():
    rest,tets=block_mesh([2,1,1],[.12,.04,.04]);k,m=assemble(rest,tets,10000.,.3,1000.)
    fixed=np.flatnonzero(np.repeat(rest[:,0]==0,3));force=np.zeros_like(rest);force[:,2]=tip_weights(rest,tets,.12)*.002
    return rest,k,m,fixed,force.flatten()


def test_fem_rigid_modes_consistent_mass_and_reaction_balance():
    rest,k,m,fixed,force=fixture()
    np.testing.assert_allclose(k,k.T,atol=1e-11)
    rigid=np.tile([.01,.02,.03],len(rest));rotation=np.cross(rest,[.1,.2,.3]).flatten()
    np.testing.assert_allclose(k@rigid,0,atol=1e-11);np.testing.assert_allclose(k@rotation,0,atol=1e-11)
    assert np.linalg.eigvalsh(m).min()>0
    np.testing.assert_allclose(m.sum(),3*1000*.12*.04*.04,rtol=1e-12)
    u,reaction,residual=equilibrium(k,force,dict.fromkeys(fixed,0.))
    np.testing.assert_allclose((reaction+force).reshape(-1,3).sum(0),0,atol=1e-11)
    assert residual<1e-10 and np.isfinite(u).all()


def test_loading_information_controls_identifiability():
    rest,k,m,fixed,force=fixture();prescribed=dict.fromkeys(fixed,0.)
    a,_,_=equilibrium(k,force,prescribed);b,_,_=equilibrium(2*k,force,prescribed)
    np.testing.assert_allclose(a,2*b,rtol=1e-10,atol=1e-12)
    for node in np.flatnonzero(rest[:,0]==.12):
        for axis in range(3):prescribed[3*node+axis]=.0001 if axis==2 else 0.
    a,ra,_=equilibrium(k,np.zeros_like(force),prescribed);b,rb,_=equilibrium(2*k,np.zeros_like(force),prescribed)
    np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12);np.testing.assert_allclose(2*ra,rb,atol=1e-11)
    np.testing.assert_allclose(frequencies(k,m,fixed),frequencies(2*k,2*m,fixed),rtol=1e-10)
