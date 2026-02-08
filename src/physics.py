import taichi as ti
from data import data
from utils import quat_from_angular_velocity, quat_mul, quat_normalize, quat_rotate

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


@ti.func
def compute_aabb(verts: ti.template(), start: ti.i32, end: ti.i32, center: ti.types.vector(3, ti.f32)):
    """Compute local AABB from verts[start..end) relative to center."""
    aabb_min = ti.Vector([1e10, 1e10, 1e10])
    aabb_max = ti.Vector([-1e10, -1e10, -1e10])
    for v in range(start, end):
        p = verts[v]
        aabb_min = ti.min(aabb_min, p)
        aabb_max = ti.max(aabb_max, p)
    return aabb_min - center, aabb_max - center

@ti.func
def get_world_aabb(gi: ti.i32):
    """Transform geom's local AABB to world space using body pos + quat."""
    geom = data.geoms[gi]
    body = data.bodies[geom.body_idx]

    local_center = (geom.aabb_min + geom.aabb_max) * 0.5
    local_half = (geom.aabb_max - geom.aabb_min) * 0.5

    world_center = body.pos + quat_rotate(body.quat, local_center)

    # Project rotated half-extents onto each world axis
    ex = ti.abs(quat_rotate(body.quat, ti.Vector([local_half[0], 0.0, 0.0])))
    ey = ti.abs(quat_rotate(body.quat, ti.Vector([0.0, local_half[1], 0.0])))
    ez = ti.abs(quat_rotate(body.quat, ti.Vector([0.0, 0.0, local_half[2]])))
    world_half = ex + ey + ez

    return world_center - world_half, world_center + world_half