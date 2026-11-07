"""SI diagnostics for SNK1 and dt-squared incremental contact exports.

Reference source: spiriMirror/libuipc cd1acc989b29f18d044cf4e449bee1576ac7d49f,
ipc_simplex_normal_contact.cu (kappa*dt*dt), simplex_normal_contact.cu
(raw exporter), stable_neo_hookean_3d_snk1.inl (density and stress).
Reference source is not a claim of installed wheel binary identity.
"""
import numpy as np


def assemble_contact(parts, vertex_count, dt):
    if dt <= 0:
        raise ValueError('Nonpositive time step')
    forces = np.zeros((vertex_count, 3))
    energy = 0.
    for ids, gradients, energies in parts:
        ids = np.asarray(ids).reshape(-1)
        gradients = np.asarray(gradients).reshape(-1, 3)
        energies = np.asarray(energies).reshape(-1)
        if len(ids) != len(gradients) or (ids < 0).any() or (ids >= vertex_count).any():
            raise ValueError('Invalid global vertex indices')
        if not np.isfinite(gradients).all() or not np.isfinite(energies).all():
            raise ValueError('Nonfinite incremental contact export')
        np.add.at(forces, ids, -gradients / dt**2)
        energy += float(energies.sum() / dt**2)
    return forces, energy


def snk1_state(rest, positions, tets, mu, lam, density=1000.):
    dm = np.stack([rest[tets[:, i]]-rest[tets[:, 0]] for i in (1, 2, 3)], -1)
    inv = np.linalg.inv(dm)
    volume = np.linalg.det(dm)/6
    if (volume <= 0).any():
        raise ValueError('Nonpositive rest volume')
    ds = np.stack([positions[tets[:, i]]-positions[tets[:, 0]] for i in (1, 2, 3)], -1)
    f = ds@inv
    j = np.linalg.det(f)
    if (j <= 0).any() or not np.isfinite(f).all():
        raise ValueError('Invalid deformation gradient')
    mu, lam = np.asarray(mu), np.asarray(lam)
    psi = .5*mu*((f*f).sum((1,2))-3)+.5*lam*(j-1-mu/lam)**2-.5*mu**2/lam
    p = mu[:, None, None]*f+(lam*(j-1)-mu)[:, None, None]*j[:, None, None]*np.linalg.inv(f).transpose(0,2,1)
    corner_gradient = volume[:, None, None]*(p@inv.transpose(0,2,1)).transpose(0,2,1)
    gradients = np.concatenate([-corner_gradient.sum(1, keepdims=True), corner_gradient], axis=1)
    assembled = np.zeros_like(rest)
    np.add.at(assembled, tets.reshape(-1), gradients.reshape(-1,3))
    masses = np.zeros(len(rest))
    np.add.at(masses, tets.reshape(-1), np.repeat(density*volume/4, 4))
    return dict(elastic_energy_j=float((psi*volume).sum()), elastic_gradient_n=assembled,
                lumped_mass_kg=masses)


def export_contact(feature, view, geometry_type, vertex_count, dt):
    parts = []
    for kind in feature.contact_primitive_types():
        gradient, energy = geometry_type(), geometry_type()
        feature.contact_gradient(kind, gradient)
        feature.contact_energy(kind, energy)
        ids = np.asarray(view(gradient.instances().find('i'))).copy()
        values = np.asarray(view(gradient.instances().find('grad'))).copy()
        energies = np.asarray(view(energy.instances().find('energy'))).copy()
        parts.append((ids, values, energies))
    return assemble_contact(parts, vertex_count, dt)
