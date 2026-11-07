"""Depth, SI projection and correspondences must be physical, not amplified visuals."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from physical_renderer import correspondence_flow, overlay_occluder, rasterize


def test_depth_test_and_material_motion_in_pixels():
    camera=dict(center_m=[0,0,0],yaw=0.,pitch=0.,image_size=64,field_of_view_m=.08)
    # Overlapping sheets: front z=+.01 must occlude back z=-.01.
    sheet=np.array([[-.02,-.02,0],[.02,-.02,0],[.02,.02,0],[-.02,.02,0]])
    rest=np.r_[sheet+[0,0,-.01],sheet+[0,0,.01]]
    triangles=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]])
    first=rasterize(rest,rest,triangles,camera)
    assert np.all(first['face_index'][first['mask']]>=2)
    moved=rest+[.005,0,0]
    final=rasterize(rest,moved,triangles,camera)
    flow,valid=correspondence_flow(first,final,triangles,moved,camera)
    assert valid.sum()>500
    np.testing.assert_allclose(flow[valid,0],4.,atol=1e-5)
    np.testing.assert_allclose(flow[valid,1],0.,atol=1e-5)
    # Uniform material translation preserves the independently seeded texture.
    np.testing.assert_array_equal(first['rgb'][20:40,20:40],final['rgb'][20:40,24:44])


def test_zero_motion_has_zero_correspondence_and_deterministic_appearance():
    camera=dict(center_m=[0,0,0],yaw=0.,pitch=0.,image_size=32,field_of_view_m=.08)
    rest=np.array([[-.03,-.03,0],[.03,-.03,0],[0,.03,0]])
    triangles=np.array([[0,1,2]])
    first=rasterize(rest,rest,triangles,camera,17)
    second=rasterize(rest,rest,triangles,camera,17)
    flow,valid=correspondence_flow(first,second,triangles,rest,camera)
    assert valid.sum()>100
    assert np.abs(flow[valid]).max()<1e-5
    np.testing.assert_array_equal(first['rgb'],second['rgb'])


def test_paired_occlusion_changes_visibility_without_changing_clean_image():
    rgb=np.full((64,64,3),100,dtype=np.uint8);mask=np.zeros((64,64),bool);mask[16:48,12:52]=True
    moderate,occlusion=overlay_occluder(rgb,mask,'moderate')
    heavy,heavy_mask=overlay_occluder(rgb,mask,'heavy')
    assert .2<np.count_nonzero(mask&occlusion)/mask.sum()<.4
    assert np.count_nonzero(mask&heavy_mask)>np.count_nonzero(mask&occlusion)
    assert np.all(rgb==100)
    np.testing.assert_array_equal(moderate[~occlusion],rgb[~occlusion])
