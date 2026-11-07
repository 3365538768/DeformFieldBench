"""3D small-strain elasticity forms, using NVIDIA Warp FEM's public API."""
import warp as wp
import warp.fem as fem


@fem.integrand
def stiffness(s: fem.Sample, u: fem.Field, v: fem.Field, mu: wp.float32, lam: wp.float32):
    gu = fem.grad(u, s); gv = fem.grad(v, s)
    eu = wp.float32(0.5) * (gu + wp.transpose(gu))
    ev = wp.float32(0.5) * (gv + wp.transpose(gv))
    return wp.float32(2.0) * mu * wp.ddot(eu, ev) + lam * wp.trace(gu) * wp.trace(gv)


@fem.integrand
def clamp_projector(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field):
    position = domain(s)
    if position[0] < wp.float32(1e-12):
        return wp.dot(u(s), v(s))
    return wp.float32(0.0)


@fem.integrand
def tip_traction(s: fem.Sample, domain: fem.Domain, v: fem.Field, traction: wp.vec3, length: wp.float32):
    normal = fem.normal(domain, s)
    if normal[0] > 0.99:
        return wp.dot(traction, v(s))
    return wp.float32(0.0)
