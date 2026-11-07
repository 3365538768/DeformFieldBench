import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from contact_diagnostics import assemble_contact, snk1_state
from engine_fixture import block_mesh


def test_incremental_contact_units_duplicate_scatter_and_action_reaction():
    forces, energy = assemble_contact([(np.array([0,1,0]), np.array([[0.,-.0002,0],[0,.0003,0],[0,-.0001,0]]), [1e-7])], 2, .01)
    np.testing.assert_allclose(forces, [[0,3,0],[0,-3,0]], atol=1e-14)
    assert energy == pytest.approx(.001)
    with pytest.raises(ValueError):
        assemble_contact([(np.array([2]), [[0,0,0]], [])], 2, .01)


def test_snk1_gradient_matches_energy_finite_difference_and_rest_mass():
    rest, tets = block_mesh([1,1,1], [.04,.02,.02])
    mu, lam = np.full(len(tets), 4000.), np.full(len(tets), 11000.)
    at_rest = snk1_state(rest, rest, tets, mu, lam)
    np.testing.assert_allclose(at_rest['elastic_gradient_n'], 0, atol=1e-14)
    assert at_rest['lumped_mass_kg'].sum() == pytest.approx(.016)
    assert at_rest['elastic_energy_j'] == pytest.approx(0, abs=1e-15)
    positions = rest.copy(); positions[:,1] += .001*rest[:,0]/.04
    state = snk1_state(rest, positions, tets, mu, lam)
    eps = 1e-7
    for vertex, axis in [(0,0),(3,1),(7,2)]:
        a,b = positions.copy(), positions.copy();a[vertex,axis] += eps;b[vertex,axis] -= eps
        fd = (snk1_state(rest,a,tets,mu,lam)['elastic_energy_j']-snk1_state(rest,b,tets,mu,lam)['elastic_energy_j'])/(2*eps)
        assert fd == pytest.approx(state['elastic_gradient_n'][vertex,axis], rel=1e-5, abs=1e-9)
