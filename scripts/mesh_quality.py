"""Compare nested or nonnested structured P1 tetrahedral solutions in SI units."""
import numpy as np


def interpolate_p1(points, values, cells, dimensions):
    """Exact interpolation in block_mesh's six Freudenthal tetrahedra per cell.

    Trilinear interpolation would silently compare a different finite element.
    """
    cells = np.asarray(cells, dtype=int)
    q = np.asarray(points, dtype=float) / np.asarray(dimensions) * cells
    if np.any(q < -1e-8) or np.any(q > cells + 1e-8):
        raise ValueError('Interpolation query outside fixture')
    q = np.clip(q, 0, cells)
    lower = np.minimum(np.floor(q).astype(int), cells - 1)
    fraction = q - lower
    order = np.argsort(-fraction, axis=1, kind='stable')
    sorted_fraction = np.take_along_axis(fraction, order, axis=1)
    weights = np.column_stack((1-sorted_fraction[:, 0],
        sorted_fraction[:, 0]-sorted_fraction[:, 1],
        sorted_fraction[:, 1]-sorted_fraction[:, 2], sorted_fraction[:, 2]))
    vertices = [lower.copy()]
    for j in range(3):
        vertex = vertices[-1].copy()
        vertex[np.arange(len(q)), order[:, j]] += 1
        vertices.append(vertex)
    indices = np.stack([np.ravel_multi_index(v.T, cells+1) for v in vertices], axis=1)
    return np.einsum('ni,ni...->n...', weights, np.asarray(values)[indices])


def volume_weights(points, tets):
    basis = points[tets[:, 1:]] - points[tets[:, 0, None]]
    volume = np.abs(np.linalg.det(basis)) / 6
    weights = np.zeros(len(points))
    np.add.at(weights, tets.ravel(), np.repeat(volume/4, 4))
    return weights


def compare_fields(coarse, fine, coarse_case):
    interpolated = interpolate_p1(fine['rest_positions_m'], coarse['displacement_m'],
                                 coarse_case['cells'], coarse_case['dimensions_m'])
    weights = volume_weights(fine['rest_positions_m'], fine['tetrahedra'])
    reference = np.sum(weights[:, None]*fine['displacement_m']**2)
    if reference <= 0:
        raise ValueError('Zero reference response')
    error = np.sum(weights[:, None]*(interpolated-fine['displacement_m'])**2)
    return float(np.sqrt(error/reference))


def accepted_refinements(comparisons, field_tolerance, tip_tolerance, required=2):
    return len(comparisons) >= required and all(
        c['field_relative_l2'] < field_tolerance and c['tip_relative_l2'] < tip_tolerance
        for c in comparisons[-required:])
