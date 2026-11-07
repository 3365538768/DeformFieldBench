import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))

from scripts.audit_contact_numerics import common_node_indices, compare


def test_nested_nodes_and_real_timestamp_alignment():
    cells = [2, 2, 2]
    fine = [4, 4, 4]
    a = dict(case=dict(cells=cells, dt_seconds=.02), displacement=np.ones((2, 27, 3)))
    data = np.zeros((4, 125, 3)); data[1::2] = 1
    b = dict(case=dict(cells=fine, dt_seconds=.01), displacement=data)
    assert len(np.unique(common_node_indices(fine, cells))) == 27
    assert compare(a, b, cells)['full_time_common_node_relative_l2'] == 0
    # Incorrectly taking frames 0,2 would miss these aligned samples.
    data[1::2] = 2
    assert compare(a, b, cells)['full_time_common_node_relative_l2'] == .5


def test_non_nested_mesh_is_rejected():
    with pytest.raises(ValueError):
        common_node_indices([5, 4, 4], [2, 2, 2])
