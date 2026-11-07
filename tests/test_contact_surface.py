import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from audit_contact_surface import frame_overlap, triangles_inside_box
from engine_fixture import block_mesh
from physical_renderer import exterior_triangles


def test_triangle_crossing_box_is_detected_when_all_vertices_are_outside():
    triangle=np.array([[[-2.,0,0],[2.,2,0],[2.,-2,0]]])
    assert not (np.abs(triangle)<1).all(2).any()
    assert triangles_inside_box(triangle,-np.ones(3),np.ones(3))[0]
    assert not triangles_inside_box(triangle+np.array([0,0,2]),-np.ones(3),np.ones(3))[0]


def test_containment_and_affine_tool_mapping():
    body,tets=block_mesh([1,1,1],[2,2,2]);body-=1
    tool,_=block_mesh([1,1,1],[.2,.2,.2]);tool-=.1
    count,contained=frame_overlap(tool,tool,body,tets,exterior_triangles(tets))
    assert count==0 and contained
    translated=tool+np.array([3.,0,0])
    count,contained=frame_overlap(tool,translated,body,tets,exterior_triangles(tets))
    assert count==0 and not contained
