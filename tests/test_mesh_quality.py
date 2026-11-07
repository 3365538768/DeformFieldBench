"""Nonnested comparisons must use the actual P1 element, not voxel interpolation."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from engine_fixture import block_mesh
from mesh_quality import accepted_refinements, compare_fields, interpolate_p1


def test_p1_interpolation_preserves_affine_fields_and_boundaries():
    points,tets=block_mesh([2,2,2],[2.,3.,4.])
    queries=np.array([[0.,0.,0.],[2.,3.,4.],[.23,1.7,2.9]])
    matrix=np.array([[1.,2.,3.],[.1,-.3,.8],[0.,2.,.1]])
    values=points@matrix+np.array([.2,.3,.4])
    np.testing.assert_allclose(interpolate_p1(queries,values,[2,2,2],[2.,3.,4.]),queries@matrix+[.2,.3,.4])
    with pytest.raises(ValueError):interpolate_p1([[2.01,0,0]],values,[2,2,2],[2.,3.,4.])


def test_freudenthal_interpolation_differs_from_trilinear_and_comparison_detects_error():
    points,tets=block_mesh([1,1,1],[1.,1.,1.])
    values=(points[:,0]*points[:,1])[:,None]
    actual=interpolate_p1([[.8,.3,.2]],values,[1,1,1],[1.,1.,1.])
    assert actual[0,0]==pytest.approx(.3)  # Trilinear would give .24.
    coarse=dict(rest_positions_m=points,tetrahedra=tets,displacement_m=points*.01)
    fine_points,fine_tets=block_mesh([3,3,3],[1.,1.,1.])
    fine=dict(rest_positions_m=fine_points,tetrahedra=fine_tets,displacement_m=fine_points*.01)
    assert compare_fields(coarse,fine,dict(cells=[1,1,1],dimensions_m=[1.,1.,1.]))<1e-14
    coarse['displacement_m']*=2
    assert compare_fields(coarse,fine,dict(cells=[1,1,1],dimensions_m=[1.,1.,1.]))==pytest.approx(1.)


def test_one_good_refinement_cannot_hide_a_bad_previous_refinement():
    good=dict(field_relative_l2=.01,tip_relative_l2=.02)
    bad=dict(field_relative_l2=.04,tip_relative_l2=.01)
    assert not accepted_refinements([bad,good],.03,.03)
    assert not accepted_refinements([good],.03,.03)
    assert accepted_refinements([bad,good,good],.03,.03)
