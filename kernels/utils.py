"""Shared math utilities for physics kernels."""
import taichi as ti


# =============================================================================
# Quaternion helpers (w, x, y, z format)
# =============================================================================

@ti.func
def quat_mul(q1: ti.types.vector(4, ti.f32), q2: ti.types.vector(4, ti.f32)) -> ti.types.vector(4, ti.f32):
    """Multiply two quaternions: q1 * q2"""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return ti.Vector([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


@ti.func
def quat_rotate(q: ti.types.vector(4, ti.f32), v: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    """Rotate vector v by quaternion q."""
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Optimized quaternion-vector rotation (faster than q * v * q_conj)
    t = 2.0 * ti.Vector([y*v[2] - z*v[1], z*v[0] - x*v[2], x*v[1] - y*v[0]])
    return v + w * t + ti.Vector([y*t[2] - z*t[1], z*t[0] - x*t[2], x*t[1] - y*t[0]])


@ti.func
def quat_normalize(q: ti.types.vector(4, ti.f32)) -> ti.types.vector(4, ti.f32):
    """Normalize quaternion to unit length."""
    length = ti.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    result = ti.Vector([1.0, 0.0, 0.0, 0.0])  # Identity as fallback
    if length > 1e-8:
        result = q / length
    return result


@ti.func
def quat_from_angular_velocity(omega: ti.types.vector(3, ti.f32), dt: ti.f32) -> ti.types.vector(4, ti.f32):
    """Create quaternion from angular velocity * dt (small angle approximation)."""
    half_angle = omega * (dt * 0.5)
    return ti.Vector([1.0, half_angle[0], half_angle[1], half_angle[2]])


@ti.func
def quat_conjugate(q: ti.types.vector(4, ti.f32)) -> ti.types.vector(4, ti.f32):
    """Return conjugate of quaternion (inverse for unit quaternions)."""
    return ti.Vector([q[0], -q[1], -q[2], -q[3]])


@ti.func
def quat_rotate_inverse(q: ti.types.vector(4, ti.f32), v: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    """Rotate vector v by inverse of quaternion q (world to local)."""
    return quat_rotate(quat_conjugate(q), v)


# =============================================================================
# Ray-tracing helpers
# =============================================================================


@ti.func
def intersect_aabb(ray_o, ray_d, bmin, bmax, closest_t):
    """Ray-AABB intersection test"""
    inv_d = 1.0 / ray_d

    tx1 = (bmin[0] - ray_o[0]) * inv_d[0]
    tx2 = (bmax[0] - ray_o[0]) * inv_d[0]
    tmin = ti.min(tx1, tx2)
    tmax = ti.max(tx1, tx2)

    ty1 = (bmin[1] - ray_o[1]) * inv_d[1]
    ty2 = (bmax[1] - ray_o[1]) * inv_d[1]
    tmin = ti.max(tmin, ti.min(ty1, ty2))
    tmax = ti.min(tmax, ti.max(ty1, ty2))

    tz1 = (bmin[2] - ray_o[2]) * inv_d[2]
    tz2 = (bmax[2] - ray_o[2]) * inv_d[2]
    tmin = ti.max(tmin, ti.min(tz1, tz2))
    tmax = ti.min(tmax, ti.max(tz1, tz2))

    return tmax >= tmin and tmin < closest_t and tmax > 0.0


@ti.func
def ray_triangle_intersect(ray_o, ray_d, v0, v1, v2):
    """Moller-Trumbore algorithm"""
    e1 = v1 - v0
    e2 = v2 - v0
    h = ray_d.cross(e2)
    a = e1.dot(h)
    t = -1.0

    if ti.abs(a) > 1e-8:
        f = 1.0 / a
        s = ray_o - v0
        u = f * s.dot(h)
        if 0.0 <= u <= 1.0:
            q = s.cross(e1)
            v = f * ray_d.dot(q)
            if v >= 0.0 and u + v <= 1.0:
                t = f * e2.dot(q)
                if t < 0.001:
                    t = -1.0
    return t


@ti.func
def get_triangle_normal(v0, v1, v2):
    e1 = v1 - v0
    e2 = v2 - v0
    return e1.cross(e2).normalized()
