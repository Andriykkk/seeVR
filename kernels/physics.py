"""Physics simulation kernels for rigid body dynamics."""
import taichi as ti
import kernels.data as data


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
    # For small rotations: q ≈ [1, omega*dt/2]
    half_angle = omega * (dt * 0.5)
    return ti.Vector([1.0, half_angle[0], half_angle[1], half_angle[2]])


# =============================================================================
# Physics kernels
# =============================================================================

@ti.kernel
def compute_local_vertices(num_bodies: ti.i32):
    """Compute local-space vertices from world-space vertices.

    local_vertex = world_vertex - body.pos

    Call ONCE after scene setup, before simulation starts.
    This converts world-space render vertices to body-local coordinates.
    """
    for b in range(num_bodies):
        body_pos = data.bodies[b].pos
        start = data.bodies[b].vert_start
        count = data.bodies[b].vert_count

        for v in range(start, start + count):
            data.original_vertices[v] = data.vertices[v] - body_pos


@ti.kernel
def apply_gravity(num_bodies: ti.i32, dt: ti.f32):
    """Apply gravity to velocity for all dynamic bodies.

    Only affects bodies with inv_mass > 0 (dynamic bodies).
    Static bodies (inv_mass == 0) are unaffected.
    """
    g = data.gravity[None]
    for i in range(num_bodies):
        if data.bodies[i].inv_mass > 0:
            data.bodies[i].vel += g * dt


@ti.kernel
def integrate_bodies(num_bodies: ti.i32, dt: ti.f32):
    """Integrate positions and orientations from velocities.

    pos += vel * dt
    quat = normalize(quat * quat_from_omega(omega, dt))
    """
    for i in range(num_bodies):
        if data.bodies[i].inv_mass > 0:
            # Integrate position
            data.bodies[i].pos += data.bodies[i].vel * dt

            # Integrate orientation (quaternion)
            omega = data.bodies[i].omega
            if omega[0] != 0.0 or omega[1] != 0.0 or omega[2] != 0.0:
                dq = quat_from_angular_velocity(omega, dt)
                q = data.bodies[i].quat
                new_q = quat_mul(q, dq)
                data.bodies[i].quat = quat_normalize(new_q)


@ti.kernel
def update_render_vertices(num_bodies: ti.i32):
    """Transform render mesh vertices from local to world space.

    world_vertex = body.pos + quat_rotate(body.quat, local_vertex)
    """
    for b in range(num_bodies):
        body_pos = data.bodies[b].pos
        body_quat = data.bodies[b].quat
        start = data.bodies[b].vert_start
        count = data.bodies[b].vert_count

        for v in range(start, start + count):
            local_pos = data.original_vertices[v]
            data.vertices[v] = body_pos + quat_rotate(body_quat, local_pos)
