"""Metric procedural solids for contact preflight; these are not reconstructed GS."""
import numpy as np

from engine_fixture import block_mesh


SHAPES = ('tapered', 'waisted', 'curved', 'twisted')


def make_solid(shape, cells=(12, 6, 6), dimensions=(.04, .02, .02)):
    if shape not in SHAPES:
        raise ValueError('Unknown procedural solid')
    reference, tets = block_mesh(cells, dimensions)
    points = reference.copy()
    s = reference[:, 0] / dimensions[0]
    yz = reference[:, 1:] - np.asarray(dimensions[1:]) / 2
    scale = np.ones(len(points))
    if shape == 'tapered':
        scale = 1 - .35*s
    elif shape == 'waisted':
        scale = .65 + .35*np.cos(np.pi*s)**2
    elif shape == 'curved':
        scale = .75 + .25*np.sin(np.pi*s)
        yz[:, 0] += .003*np.sin(np.pi*s)
    elif shape == 'twisted':
        angle = .65*s
        y, z = yz.T.copy()
        yz = np.column_stack((np.cos(angle)*y-np.sin(angle)*z,
                              np.sin(angle)*y+np.cos(angle)*z))
    points[:, 1:] = yz*scale[:, None]
    return points, tets, reference


def material_regions(reference, tets, layout, dimensions=(.04, .02, .02)):
    centers = reference[tets].mean(axis=1)
    if layout == 'homogeneous':
        return np.zeros(len(tets), dtype=np.int32)
    if layout == 'left_right':
        return (centers[:, 0] >= dimensions[0]*.55).astype(np.int32)
    if layout == 'core_shell':
        yz = (centers[:, 1:]-np.asarray(dimensions[1:])/2)/(np.asarray(dimensions[1:])/2)
        return (np.linalg.norm(yz, axis=1) >= .65).astype(np.int32)
    raise ValueError('Unknown bonded material layout')


def mesh_health(points, tets):
    basis = points[tets[:, 1:]]-points[tets[:, 0, None]]
    determinants = np.linalg.det(basis)
    faces = np.concatenate([tets[:, f] for f in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3])])
    _, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    singular = np.linalg.svd(basis, compute_uv=False)
    condition = singular[:, 0]/singular[:, -1]
    return dict(vertices=len(points), tetrahedra=len(tets), volume_m3=float(determinants.sum()/6),
                min_signed_tet_volume_m3=float(determinants.min()/6),
                max_basis_condition=float(condition.max()), boundary_triangles=int((counts == 1).sum()),
                face_incidence_valid=bool(counts.max() <= 2),
                passed=bool(np.isfinite(points).all() and (determinants > 0).all()
                            and counts.max() <= 2 and condition.max() < 20))
