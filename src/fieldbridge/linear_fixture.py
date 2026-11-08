"""Small-strain isotropic P1 tetrahedral reference, with consistent mass.

This is a controlled identifiability fixture, not the original MPM model or
a validated model for finite deformation, plasticity or tool contact.
"""
import numpy as np
from scipy.linalg import eigh, solve


def assemble(rest, tets, young, poisson, density):
    if young <= 0 or density <= 0 or not -1 < poisson < .5:
        raise ValueError('Invalid linear elastic material')
    n=3*len(rest);stiffness=np.zeros((n,n));mass=np.zeros((n,n))
    mu=young/(2*(1+poisson));lam=young*poisson/((1+poisson)*(1-2*poisson))
    elasticity=np.diag([2*mu]*3+[mu]*3);elasticity[:3,:3]+=lam
    for tet in tets:
        xyz=rest[tet];basis=np.column_stack((np.ones(4),xyz))
        volume=abs(np.linalg.det(basis))/6
        if volume<=0:raise ValueError('Degenerate tetrahedron')
        gradients=np.linalg.inv(basis)[1:,:].T;strain=np.zeros((6,12))
        for i,(x,y,z) in enumerate(gradients):
            strain[:,3*i:3*i+3]=[[x,0,0],[0,y,0],[0,0,z],[y,x,0],[0,z,y],[z,0,x]]
        dofs=(3*np.asarray(tet)[:,None]+np.arange(3)).flatten();ix=np.ix_(dofs,dofs)
        stiffness[ix]+=volume*strain.T@elasticity@strain
        mass[ix]+=density*volume/20*np.kron(np.ones((4,4))+np.eye(4),np.eye(3))
    return stiffness,mass


def equilibrium(stiffness, force, prescribed):
    """Return displacement, full residual and normalized free-DOF residual."""
    fixed=np.array(sorted(prescribed),dtype=int);free=np.setdiff1d(np.arange(len(force)),fixed)
    u=np.zeros(len(force));u[fixed]=[prescribed[i] for i in fixed]
    u[free]=solve(stiffness[np.ix_(free,free)],force[free]-stiffness[np.ix_(free,fixed)]@u[fixed],assume_a='pos')
    residual=stiffness@u-force
    scale=max(np.linalg.norm(force),np.linalg.norm(stiffness@u),1e-30)
    return u,residual,float(np.linalg.norm(residual[free])/scale)


def frequencies(stiffness,mass,fixed):
    free=np.setdiff1d(np.arange(len(stiffness)),fixed)
    values=eigh(stiffness[np.ix_(free,free)],mass[np.ix_(free,free)],subset_by_index=(0,2),eigvals_only=True)
    if np.any(values<=0):raise ValueError('Non-positive clamped vibration mode')
    return np.sqrt(values)/(2*np.pi)
