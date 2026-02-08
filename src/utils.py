import taichi as ti

@ti.func
def quat_from_angular_velocity(omega: ti.types.vector(3, ti.f32), dt: ti.f32) -> ti.types.vector(4, ti.f32):
    """Create quaternion from angular velocity * dt (small angle approximation)."""
    half_angle = omega * (dt * 0.5)
    return ti.Vector([1.0, half_angle[0], half_angle[1], half_angle[2]])

@ti.func
def quat_normalize(q: ti.types.vector(4, ti.f32)) -> ti.types.vector(4, ti.f32):
    """Normalize quaternion to unit length."""
    length = ti.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    result = ti.Vector([1.0, 0.0, 0.0, 0.0])  # Identity as fallback
    if length > 1e-8:
        result = q / length
    return result

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
