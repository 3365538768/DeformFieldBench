import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from irregular_geometry import SHAPES, make_solid, material_regions, mesh_health


@pytest.mark.parametrize('shape', SHAPES)
def test_procedural_solids_have_positive_volume_and_closed_boundary(shape):
    points, tets, _ = make_solid(shape, (8, 4, 4))
    health = mesh_health(points, tets)
    assert health['passed'] and 5e-6 < health['volume_m3'] < 2e-5
    # Every boundary edge occurs exactly twice: closed manifold surface.
    faces = np.concatenate([tets[:, f] for f in ([0,1,2], [0,1,3], [0,2,3], [1,2,3])])
    faces, count = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    faces = faces[count == 1]
    edges = np.concatenate([faces[:, e] for e in ([0,1], [0,2], [1,2])])
    assert (np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)[1] == 2).all()


def test_bonded_regions_share_interface_vertices_and_homogeneous_limit():
    _, tets, reference = make_solid('waisted', (8,4,4))
    assert not material_regions(reference, tets, 'homogeneous').any()
    for layout in ('left_right', 'core_shell'):
        ids = material_regions(reference, tets, layout)
        assert set(ids) == {0, 1}
        assert len(np.intersect1d(tets[ids == 0], tets[ids == 1])) > 0


def test_inverted_tet_cannot_enter_contact_pipeline():
    points, tets, _ = make_solid('tapered', (2,2,2))
    tets[0, [1,2]] = tets[0, [2,1]]
    assert not mesh_health(points, tets)['passed']
