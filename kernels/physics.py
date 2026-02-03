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


@ti.func
def quat_conjugate(q: ti.types.vector(4, ti.f32)) -> ti.types.vector(4, ti.f32):
    """Return conjugate of quaternion (inverse for unit quaternions)."""
    return ti.Vector([q[0], -q[1], -q[2], -q[3]])


@ti.func
def quat_rotate_inverse(q: ti.types.vector(4, ti.f32), v: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    """Rotate vector v by inverse of quaternion q (world to local)."""
    return quat_rotate(quat_conjugate(q), v)


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


# =============================================================================
# Collision geometry transforms (Step 3 from physics pipeline)
# =============================================================================

@ti.func
def compute_sphere_aabb(center: ti.types.vector(3, ti.f32), radius: ti.f32):
    """Compute AABB for a sphere."""
    r = ti.Vector([radius, radius, radius])
    return center - r, center + r


@ti.func
def compute_box_aabb(center: ti.types.vector(3, ti.f32),
                     quat: ti.types.vector(4, ti.f32),
                     half_extents: ti.types.vector(3, ti.f32)):
    """Compute AABB for an oriented box.

    For each axis, project the rotated box extents.
    """
    # Get rotation matrix columns from quaternion
    # These are the world-space directions of the box's local axes
    ax = quat_rotate(quat, ti.Vector([1.0, 0.0, 0.0]))
    ay = quat_rotate(quat, ti.Vector([0.0, 1.0, 0.0]))
    az = quat_rotate(quat, ti.Vector([0.0, 0.0, 1.0]))

    # For each world axis, compute the extent as sum of absolute projections
    extent = ti.Vector([
        ti.abs(ax[0]) * half_extents[0] + ti.abs(ay[0]) * half_extents[1] + ti.abs(az[0]) * half_extents[2],
        ti.abs(ax[1]) * half_extents[0] + ti.abs(ay[1]) * half_extents[1] + ti.abs(az[1]) * half_extents[2],
        ti.abs(ax[2]) * half_extents[0] + ti.abs(ay[2]) * half_extents[1] + ti.abs(az[2]) * half_extents[2],
    ])

    return center - extent, center + extent


@ti.func
def compute_capsule_aabb(center: ti.types.vector(3, ti.f32),
                         quat: ti.types.vector(4, ti.f32),
                         radius: ti.f32, half_length: ti.f32):
    """Compute AABB for a capsule (oriented along local Y axis)."""
    # Capsule axis in world space
    axis = quat_rotate(quat, ti.Vector([0.0, half_length, 0.0]))

    # AABB of the two sphere centers
    p1 = center + axis
    p2 = center - axis
    aabb_min = ti.min(p1, p2)
    aabb_max = ti.max(p1, p2)

    # Expand by radius
    r = ti.Vector([radius, radius, radius])
    return aabb_min - r, aabb_max + r


@ti.kernel
def update_geom_transforms(num_geoms: ti.i32):
    """Update world transforms and AABBs for all collision geometries.

    For each geom:
        world_pos = body.pos + rotate(body.quat, geom.local_pos)
        world_quat = body.quat * geom.local_quat
        aabb = compute based on geom type
    """
    for i in range(num_geoms):
        body_idx = data.geoms[i].body_idx
        body_pos = data.bodies[body_idx].pos
        body_quat = data.bodies[body_idx].quat

        # Compute world transform
        local_pos = data.geoms[i].local_pos
        local_quat = data.geoms[i].local_quat

        world_pos = body_pos + quat_rotate(body_quat, local_pos)
        world_quat = quat_mul(body_quat, local_quat)

        data.geoms[i].world_pos = world_pos
        data.geoms[i].world_quat = world_quat

        # Compute AABB based on geom type
        geom_type = data.geoms[i].geom_type
        geom_data = data.geoms[i].data

        aabb_min = ti.Vector([0.0, 0.0, 0.0])
        aabb_max = ti.Vector([0.0, 0.0, 0.0])

        if geom_type == data.GEOM_SPHERE:
            radius = geom_data[0]
            aabb_min, aabb_max = compute_sphere_aabb(world_pos, radius)

        elif geom_type == data.GEOM_BOX:
            half_extents = ti.Vector([geom_data[0], geom_data[1], geom_data[2]])
            aabb_min, aabb_max = compute_box_aabb(world_pos, world_quat, half_extents)

        elif geom_type == data.GEOM_CAPSULE:
            radius = geom_data[0]
            half_length = geom_data[1]
            aabb_min, aabb_max = compute_capsule_aabb(world_pos, world_quat, radius, half_length)

        elif geom_type == data.GEOM_PLANE:
            # Infinite plane - use large AABB
            aabb_min = ti.Vector([-1e6, -1e6, -1e6])
            aabb_max = ti.Vector([1e6, 1e6, 1e6])

        data.geoms[i].aabb_min = aabb_min
        data.geoms[i].aabb_max = aabb_max


# =============================================================================
# Broad phase collision detection (Step 4 from physics pipeline)
# =============================================================================

@ti.func
def aabb_overlap(min_a: ti.types.vector(3, ti.f32), max_a: ti.types.vector(3, ti.f32),
                 min_b: ti.types.vector(3, ti.f32), max_b: ti.types.vector(3, ti.f32)) -> ti.i32:
    """Check if two AABBs overlap. Returns 1 if overlap, 0 otherwise."""
    # No overlap if separated along any axis
    overlap = 1
    if max_a[0] < min_b[0] or max_b[0] < min_a[0]:
        overlap = 0
    if max_a[1] < min_b[1] or max_b[1] < min_a[1]:
        overlap = 0
    if max_a[2] < min_b[2] or max_b[2] < min_a[2]:
        overlap = 0
    return overlap


@ti.kernel
def broad_phase_n_squared(num_geoms: ti.i32):
    """Find all geom pairs with overlapping AABBs (N² algorithm).

    For each pair (i, j) where i < j:
        - Skip if same body (can't collide with yourself)
        - Skip if both bodies are static (neither can move)
        - Check AABB overlap
        - If overlap, add to collision_pairs using atomic counter

    Output: collision_pairs[0..num_collision_pairs] contains candidate pairs
    """
    # N² loop: each thread handles one geom, checks against all higher-indexed geoms
    for i in range(num_geoms):
        body_a = data.geoms[i].body_idx
        inv_mass_a = data.bodies[body_a].inv_mass

        for j in range(i + 1, num_geoms):
            body_b = data.geoms[j].body_idx

            # Skip self-collision (same body)
            if body_a == body_b:
                continue

            # Skip static-static pairs (neither can move)
            inv_mass_b = data.bodies[body_b].inv_mass
            if inv_mass_a == 0.0 and inv_mass_b == 0.0:
                continue

            # Check AABB overlap
            if aabb_overlap(data.geoms[i].aabb_min, data.geoms[i].aabb_max,
                           data.geoms[j].aabb_min, data.geoms[j].aabb_max):
                # Atomically get slot in collision_pairs array
                slot = ti.atomic_add(data.num_collision_pairs[None], 1)
                if slot < data.MAX_COLLISION_PAIRS:
                    data.collision_pairs[slot] = ti.Vector([i, j])


# =============================================================================
# Narrow phase collision detection (Step 5 from physics pipeline)
# =============================================================================

@ti.func
def collide_sphere_sphere(pos_a: ti.types.vector(3, ti.f32), radius_a: ti.f32,
                          pos_b: ti.types.vector(3, ti.f32), radius_b: ti.f32,
                          body_a: ti.i32, body_b: ti.i32,
                          geom_a: ti.i32, geom_b: ti.i32):
    """Sphere-sphere collision detection.

    Returns: (has_contact, contact_point, normal, depth)
    Normal points from A to B.
    """
    diff = pos_b - pos_a
    dist_sq = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
    sum_radii = radius_a + radius_b

    has_contact = 0
    contact_point = ti.Vector([0.0, 0.0, 0.0])
    normal = ti.Vector([0.0, 1.0, 0.0])  # Default up
    depth = 0.0

    if dist_sq < sum_radii * sum_radii:
        dist = ti.sqrt(dist_sq)
        if dist > 1e-8:
            normal = diff / dist
        else:
            normal = ti.Vector([0.0, 1.0, 0.0])

        depth = sum_radii - dist
        # Contact point on surface of A (or midpoint if overlapping)
        contact_point = pos_a + normal * (radius_a - depth * 0.5)
        has_contact = 1

    return has_contact, contact_point, normal, depth


@ti.func
def collide_sphere_box(sphere_pos: ti.types.vector(3, ti.f32), sphere_radius: ti.f32,
                       box_pos: ti.types.vector(3, ti.f32), box_quat: ti.types.vector(4, ti.f32),
                       box_half: ti.types.vector(3, ti.f32),
                       body_a: ti.i32, body_b: ti.i32,
                       geom_a: ti.i32, geom_b: ti.i32):
    """Sphere-box collision detection.

    Returns: (has_contact, contact_point, normal, depth)
    Normal points from box to sphere.
    """
    # Transform sphere center to box's local space
    local_sphere = quat_rotate_inverse(box_quat, sphere_pos - box_pos)

    # Find closest point on box to sphere (clamping to box bounds)
    closest_local = ti.Vector([
        ti.max(-box_half[0], ti.min(box_half[0], local_sphere[0])),
        ti.max(-box_half[1], ti.min(box_half[1], local_sphere[1])),
        ti.max(-box_half[2], ti.min(box_half[2], local_sphere[2]))
    ])

    # Distance from sphere center to closest point
    diff = local_sphere - closest_local
    dist_sq = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]

    has_contact = 0
    contact_point = ti.Vector([0.0, 0.0, 0.0])
    normal = ti.Vector([0.0, 1.0, 0.0])
    depth = 0.0

    if dist_sq < sphere_radius * sphere_radius:
        has_contact = 1

        # Check if sphere center is inside box
        inside = (ti.abs(local_sphere[0]) <= box_half[0] and
                  ti.abs(local_sphere[1]) <= box_half[1] and
                  ti.abs(local_sphere[2]) <= box_half[2])

        if inside:
            # Sphere center inside box - find nearest face
            # Distance to each face
            dx_pos = box_half[0] - local_sphere[0]
            dx_neg = box_half[0] + local_sphere[0]
            dy_pos = box_half[1] - local_sphere[1]
            dy_neg = box_half[1] + local_sphere[1]
            dz_pos = box_half[2] - local_sphere[2]
            dz_neg = box_half[2] + local_sphere[2]

            min_dist = dx_pos
            local_normal = ti.Vector([1.0, 0.0, 0.0])

            if dx_neg < min_dist:
                min_dist = dx_neg
                local_normal = ti.Vector([-1.0, 0.0, 0.0])
            if dy_pos < min_dist:
                min_dist = dy_pos
                local_normal = ti.Vector([0.0, 1.0, 0.0])
            if dy_neg < min_dist:
                min_dist = dy_neg
                local_normal = ti.Vector([0.0, -1.0, 0.0])
            if dz_pos < min_dist:
                min_dist = dz_pos
                local_normal = ti.Vector([0.0, 0.0, 1.0])
            if dz_neg < min_dist:
                min_dist = dz_neg
                local_normal = ti.Vector([0.0, 0.0, -1.0])

            depth = min_dist + sphere_radius
            normal = quat_rotate(box_quat, local_normal)
            contact_point = sphere_pos - normal * sphere_radius
        else:
            # Sphere center outside box
            dist = ti.sqrt(dist_sq)
            local_normal = diff / dist
            depth = sphere_radius - dist
            normal = quat_rotate(box_quat, local_normal)
            contact_point = box_pos + quat_rotate(box_quat, closest_local)

    return has_contact, contact_point, normal, depth


@ti.func
def test_sat_axis(axis: ti.types.vector(3, ti.f32), d: ti.types.vector(3, ti.f32),
                  ra: ti.f32, rb: ti.f32,
                  has_contact: ti.i32, min_overlap: ti.f32, best_axis: ti.types.vector(3, ti.f32)):
    """Test one SAT axis and update collision state."""
    dist = ti.abs(d.dot(axis))
    overlap = ra + rb - dist

    new_has_contact = has_contact
    new_min_overlap = min_overlap
    new_best_axis = best_axis

    if overlap < 0:
        new_has_contact = 0
    elif overlap < min_overlap:
        new_min_overlap = overlap
        if d.dot(axis) < 0:
            new_best_axis = -axis
        else:
            new_best_axis = axis

    return new_has_contact, new_min_overlap, new_best_axis


@ti.func
def collide_box_box(pos_a: ti.types.vector(3, ti.f32), quat_a: ti.types.vector(4, ti.f32),
                    half_a: ti.types.vector(3, ti.f32),
                    pos_b: ti.types.vector(3, ti.f32), quat_b: ti.types.vector(4, ti.f32),
                    half_b: ti.types.vector(3, ti.f32),
                    body_a: ti.i32, body_b: ti.i32,
                    geom_a: ti.i32, geom_b: ti.i32):
    """Box-box collision using Separating Axis Theorem (SAT).

    Tests 15 axes: 3 from box A, 3 from box B, 9 edge cross products.
    Returns: (has_contact, contact_point, normal, depth)
    Normal points from A to B.
    """
    # Get axes of both boxes as separate vectors (can't use vector of vectors in Taichi)
    ax_a0 = quat_rotate(quat_a, ti.Vector([1.0, 0.0, 0.0]))
    ax_a1 = quat_rotate(quat_a, ti.Vector([0.0, 1.0, 0.0]))
    ax_a2 = quat_rotate(quat_a, ti.Vector([0.0, 0.0, 1.0]))

    ax_b0 = quat_rotate(quat_b, ti.Vector([1.0, 0.0, 0.0]))
    ax_b1 = quat_rotate(quat_b, ti.Vector([0.0, 1.0, 0.0]))
    ax_b2 = quat_rotate(quat_b, ti.Vector([0.0, 0.0, 1.0]))

    # Vector from A to B
    d = pos_b - pos_a

    has_contact = 1
    min_overlap = 1e30
    best_axis = ti.Vector([0.0, 1.0, 0.0])

    # Test 3 face axes of box A
    # Axis ax_a0
    ra = half_a[0]
    rb = (ti.abs(ax_a0.dot(ax_b0)) * half_b[0] +
          ti.abs(ax_a0.dot(ax_b1)) * half_b[1] +
          ti.abs(ax_a0.dot(ax_b2)) * half_b[2])
    has_contact, min_overlap, best_axis = test_sat_axis(ax_a0, d, ra, rb, has_contact, min_overlap, best_axis)

    # Axis ax_a1
    ra = half_a[1]
    rb = (ti.abs(ax_a1.dot(ax_b0)) * half_b[0] +
          ti.abs(ax_a1.dot(ax_b1)) * half_b[1] +
          ti.abs(ax_a1.dot(ax_b2)) * half_b[2])
    has_contact, min_overlap, best_axis = test_sat_axis(ax_a1, d, ra, rb, has_contact, min_overlap, best_axis)

    # Axis ax_a2
    ra = half_a[2]
    rb = (ti.abs(ax_a2.dot(ax_b0)) * half_b[0] +
          ti.abs(ax_a2.dot(ax_b1)) * half_b[1] +
          ti.abs(ax_a2.dot(ax_b2)) * half_b[2])
    has_contact, min_overlap, best_axis = test_sat_axis(ax_a2, d, ra, rb, has_contact, min_overlap, best_axis)

    # Test 3 face axes of box B
    # Axis ax_b0
    ra = (ti.abs(ax_b0.dot(ax_a0)) * half_a[0] +
          ti.abs(ax_b0.dot(ax_a1)) * half_a[1] +
          ti.abs(ax_b0.dot(ax_a2)) * half_a[2])
    rb = half_b[0]
    has_contact, min_overlap, best_axis = test_sat_axis(ax_b0, d, ra, rb, has_contact, min_overlap, best_axis)

    # Axis ax_b1
    ra = (ti.abs(ax_b1.dot(ax_a0)) * half_a[0] +
          ti.abs(ax_b1.dot(ax_a1)) * half_a[1] +
          ti.abs(ax_b1.dot(ax_a2)) * half_a[2])
    rb = half_b[1]
    has_contact, min_overlap, best_axis = test_sat_axis(ax_b1, d, ra, rb, has_contact, min_overlap, best_axis)

    # Axis ax_b2
    ra = (ti.abs(ax_b2.dot(ax_a0)) * half_a[0] +
          ti.abs(ax_b2.dot(ax_a1)) * half_a[1] +
          ti.abs(ax_b2.dot(ax_a2)) * half_a[2])
    rb = half_b[2]
    has_contact, min_overlap, best_axis = test_sat_axis(ax_b2, d, ra, rb, has_contact, min_overlap, best_axis)

    # Test 9 edge cross products (only if still potentially colliding)
    if has_contact == 1:
        # ax_a0 x ax_b0
        axis = ax_a0.cross(ax_b0)
        axis_len = axis.norm()
        if axis_len > 1e-6:
            axis = axis / axis_len
            ra = ti.abs(axis.dot(ax_a1)) * half_a[1] + ti.abs(axis.dot(ax_a2)) * half_a[2]
            rb = ti.abs(axis.dot(ax_b1)) * half_b[1] + ti.abs(axis.dot(ax_b2)) * half_b[2]
            has_contact, min_overlap, best_axis = test_sat_axis(axis, d, ra, rb, has_contact, min_overlap, best_axis)

        # ax_a0 x ax_b1
        axis = ax_a0.cross(ax_b1)
        axis_len = axis.norm()
        if axis_len > 1e-6:
            axis = axis / axis_len
            ra = ti.abs(axis.dot(ax_a1)) * half_a[1] + ti.abs(axis.dot(ax_a2)) * half_a[2]
            rb = ti.abs(axis.dot(ax_b0)) * half_b[0] + ti.abs(axis.dot(ax_b2)) * half_b[2]
            has_contact, min_overlap, best_axis = test_sat_axis(axis, d, ra, rb, has_contact, min_overlap, best_axis)

        # ax_a0 x ax_b2
        axis = ax_a0.cross(ax_b2)
        axis_len = axis.norm()
        if axis_len > 1e-6:
            axis = axis / axis_len
            ra = ti.abs(axis.dot(ax_a1)) * half_a[1] + ti.abs(axis.dot(ax_a2)) * half_a[2]
            rb = ti.abs(axis.dot(ax_b0)) * half_b[0] + ti.abs(axis.dot(ax_b1)) * half_b[1]
            has_contact, min_overlap, best_axis = test_sat_axis(axis, d, ra, rb, has_contact, min_overlap, best_axis)

        # ax_a1 x ax_b0
        axis = ax_a1.cross(ax_b0)
        axis_len = axis.norm()
        if axis_len > 1e-6:
            axis = axis / axis_len
            ra = ti.abs(axis.dot(ax_a0)) * half_a[0] + ti.abs(axis.dot(ax_a2)) * half_a[2]
            rb = ti.abs(axis.dot(ax_b1)) * half_b[1] + ti.abs(axis.dot(ax_b2)) * half_b[2]
            has_contact, min_overlap, best_axis = test_sat_axis(axis, d, ra, rb, has_contact, min_overlap, best_axis)

        # ax_a1 x ax_b1
        axis = ax_a1.cross(ax_b1)
        axis_len = axis.norm()
        if axis_len > 1e-6:
            axis = axis / axis_len
            ra = ti.abs(axis.dot(ax_a0)) * half_a[0] + ti.abs(axis.dot(ax_a2)) * half_a[2]
            rb = ti.abs(axis.dot(ax_b0)) * half_b[0] + ti.abs(axis.dot(ax_b2)) * half_b[2]
            has_contact, min_overlap, best_axis = test_sat_axis(axis, d, ra, rb, has_contact, min_overlap, best_axis)

        # ax_a1 x ax_b2
        axis = ax_a1.cross(ax_b2)
        axis_len = axis.norm()
        if axis_len > 1e-6:
            axis = axis / axis_len
            ra = ti.abs(axis.dot(ax_a0)) * half_a[0] + ti.abs(axis.dot(ax_a2)) * half_a[2]
            rb = ti.abs(axis.dot(ax_b0)) * half_b[0] + ti.abs(axis.dot(ax_b1)) * half_b[1]
            has_contact, min_overlap, best_axis = test_sat_axis(axis, d, ra, rb, has_contact, min_overlap, best_axis)

        # ax_a2 x ax_b0
        axis = ax_a2.cross(ax_b0)
        axis_len = axis.norm()
        if axis_len > 1e-6:
            axis = axis / axis_len
            ra = ti.abs(axis.dot(ax_a0)) * half_a[0] + ti.abs(axis.dot(ax_a1)) * half_a[1]
            rb = ti.abs(axis.dot(ax_b1)) * half_b[1] + ti.abs(axis.dot(ax_b2)) * half_b[2]
            has_contact, min_overlap, best_axis = test_sat_axis(axis, d, ra, rb, has_contact, min_overlap, best_axis)

        # ax_a2 x ax_b1
        axis = ax_a2.cross(ax_b1)
        axis_len = axis.norm()
        if axis_len > 1e-6:
            axis = axis / axis_len
            ra = ti.abs(axis.dot(ax_a0)) * half_a[0] + ti.abs(axis.dot(ax_a1)) * half_a[1]
            rb = ti.abs(axis.dot(ax_b0)) * half_b[0] + ti.abs(axis.dot(ax_b2)) * half_b[2]
            has_contact, min_overlap, best_axis = test_sat_axis(axis, d, ra, rb, has_contact, min_overlap, best_axis)

        # ax_a2 x ax_b2
        axis = ax_a2.cross(ax_b2)
        axis_len = axis.norm()
        if axis_len > 1e-6:
            axis = axis / axis_len
            ra = ti.abs(axis.dot(ax_a0)) * half_a[0] + ti.abs(axis.dot(ax_a1)) * half_a[1]
            rb = ti.abs(axis.dot(ax_b0)) * half_b[0] + ti.abs(axis.dot(ax_b1)) * half_b[1]
            has_contact, min_overlap, best_axis = test_sat_axis(axis, d, ra, rb, has_contact, min_overlap, best_axis)

    # Compute contact point (approximate: midpoint along collision axis)
    contact_point = ti.Vector([0.0, 0.0, 0.0])
    depth = 0.0
    normal = best_axis

    if has_contact == 1:
        depth = min_overlap
        # Contact point: support point on A in direction of normal, offset by half depth
        support_a = pos_a

        sign0 = 1.0
        if ax_a0.dot(normal) < 0:
            sign0 = -1.0
        support_a = support_a + ax_a0 * half_a[0] * sign0

        sign1 = 1.0
        if ax_a1.dot(normal) < 0:
            sign1 = -1.0
        support_a = support_a + ax_a1 * half_a[1] * sign1

        sign2 = 1.0
        if ax_a2.dot(normal) < 0:
            sign2 = -1.0
        support_a = support_a + ax_a2 * half_a[2] * sign2

        contact_point = support_a + normal * (depth * 0.5)

    return has_contact, contact_point, normal, depth


@ti.kernel
def narrow_phase(num_pairs: ti.i32):
    """Process collision pairs and generate contacts.

    For each candidate pair from broad phase:
        - Dispatch to appropriate collision function based on geom types
        - If contact found, add to contacts array
    """
    for p in range(num_pairs):
        geom_a_idx = data.collision_pairs[p][0]
        geom_b_idx = data.collision_pairs[p][1]

        type_a = data.geoms[geom_a_idx].geom_type
        type_b = data.geoms[geom_b_idx].geom_type

        pos_a = data.geoms[geom_a_idx].world_pos
        pos_b = data.geoms[geom_b_idx].world_pos
        quat_a = data.geoms[geom_a_idx].world_quat
        quat_b = data.geoms[geom_b_idx].world_quat
        data_a = data.geoms[geom_a_idx].data
        data_b = data.geoms[geom_b_idx].data

        body_a = data.geoms[geom_a_idx].body_idx
        body_b = data.geoms[geom_b_idx].body_idx

        has_contact = 0
        contact_point = ti.Vector([0.0, 0.0, 0.0])
        normal = ti.Vector([0.0, 1.0, 0.0])
        depth = 0.0

        # Sphere-Sphere
        if type_a == data.GEOM_SPHERE and type_b == data.GEOM_SPHERE:
            radius_a = data_a[0]
            radius_b = data_b[0]
            has_contact, contact_point, normal, depth = collide_sphere_sphere(
                pos_a, radius_a, pos_b, radius_b, body_a, body_b, geom_a_idx, geom_b_idx)

        # Sphere-Box: collide_sphere_box returns normal from box to sphere
        # Convention: normal from body_a to body_b, so need to flip
        elif type_a == data.GEOM_SPHERE and type_b == data.GEOM_BOX:
            radius = data_a[0]
            half_b = ti.Vector([data_b[0], data_b[1], data_b[2]])
            has_contact, contact_point, normal, depth = collide_sphere_box(
                pos_a, radius, pos_b, quat_b, half_b, body_a, body_b, geom_a_idx, geom_b_idx)
            # Flip: collide returns box→sphere, we need sphere→box (A→B)
            normal = -normal

        elif type_a == data.GEOM_BOX and type_b == data.GEOM_SPHERE:
            radius = data_b[0]
            half_a = ti.Vector([data_a[0], data_a[1], data_a[2]])
            has_contact, contact_point, normal, depth = collide_sphere_box(
                pos_b, radius, pos_a, quat_a, half_a, body_b, body_a, geom_b_idx, geom_a_idx)
            # No flip needed: collide returns box→sphere, which is A→B here

        # Box-Box
        elif type_a == data.GEOM_BOX and type_b == data.GEOM_BOX:
            half_a = ti.Vector([data_a[0], data_a[1], data_a[2]])
            half_b = ti.Vector([data_b[0], data_b[1], data_b[2]])
            has_contact, contact_point, normal, depth = collide_box_box(
                pos_a, quat_a, half_a, pos_b, quat_b, half_b, body_a, body_b, geom_a_idx, geom_b_idx)

        # Add contact if found
        if has_contact == 1 and depth > 0:
            slot = ti.atomic_add(data.num_contacts[None], 1)
            if slot < data.MAX_CONTACTS:
                data.contacts[slot].point = contact_point
                data.contacts[slot].normal = normal
                data.contacts[slot].depth = depth
                data.contacts[slot].body_a = body_a
                data.contacts[slot].body_b = body_b
                data.contacts[slot].geom_a = geom_a_idx
                data.contacts[slot].geom_b = geom_b_idx


# =============================================================================
# Contact Solver (Step 6 from physics pipeline)
# Sequential Impulse / Projected Gauss-Seidel solver
# =============================================================================

@ti.func
def compute_world_inv_inertia(body_idx: ti.i32) -> ti.types.matrix(3, 3, ti.f32):
    """Compute world-space inverse inertia tensor.

    I_world^-1 = R * I_local^-1 * R^T
    where R is the rotation matrix from body quaternion.

    Since we store diagonal inertia, this simplifies to:
    I_world^-1 = R * diag(inv_inertia) * R^T
    """
    inv_I = data.bodies[body_idx].inv_inertia
    quat = data.bodies[body_idx].quat

    # Get rotation matrix columns
    r0 = quat_rotate(quat, ti.Vector([1.0, 0.0, 0.0]))
    r1 = quat_rotate(quat, ti.Vector([0.0, 1.0, 0.0]))
    r2 = quat_rotate(quat, ti.Vector([0.0, 0.0, 1.0]))

    # I_world^-1 = R * diag(inv_I) * R^T
    # = sum over i: inv_I[i] * outer(r_i, r_i)
    result = ti.Matrix.zero(ti.f32, 3, 3)
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            result[i, j] = (inv_I[0] * r0[i] * r0[j] +
                           inv_I[1] * r1[i] * r1[j] +
                           inv_I[2] * r2[i] * r2[j])
    return result


@ti.func
def apply_impulse_to_body(body_idx: ti.i32, impulse: ti.types.vector(3, ti.f32),
                          r: ti.types.vector(3, ti.f32), inv_inertia_world: ti.types.matrix(3, 3, ti.f32)):
    """Apply impulse to body at point offset r from center of mass.

    Δv = impulse * inv_mass
    Δω = I^-1 * (r × impulse)
    """
    inv_mass = data.bodies[body_idx].inv_mass
    if inv_mass > 0:
        data.bodies[body_idx].vel += impulse * inv_mass

        # Angular impulse: torque = r × impulse
        torque = r.cross(impulse)
        # Δω = I^-1 * torque
        delta_omega = inv_inertia_world @ torque
        data.bodies[body_idx].omega += delta_omega


@ti.kernel
def solve_contacts(num_contacts: ti.i32, num_iterations: ti.i32):
    """Solve contact constraints using Sequential Impulse method.

    For each contact:
    1. Compute relative velocity at contact point
    2. Compute effective mass K
    3. Compute impulse to resolve constraint
    4. Clamp impulse (no pulling - contacts can only push)
    5. Apply impulse to both bodies

    Repeat for multiple iterations for convergence.

    Parameters:
        bias_factor: Baumgarte stabilization factor (0.1-0.3 typical)
        slop: Allowed penetration before correction (0.01 typical)
        restitution: Bounciness (0 = no bounce, 1 = perfect bounce)
    """
    # Solver parameters
    bias_factor = 0.2  # Baumgarte stabilization
    slop = 0.01  # Allowed penetration (meters)
    restitution = 0.3  # Bounciness

    for _ in range(num_iterations):
        for c in range(num_contacts):
            body_a = data.contacts[c].body_a
            body_b = data.contacts[c].body_b
            normal = data.contacts[c].normal
            contact_point = data.contacts[c].point
            depth = data.contacts[c].depth

            # Get body states
            pos_a = data.bodies[body_a].pos
            pos_b = data.bodies[body_b].pos
            vel_a = data.bodies[body_a].vel
            vel_b = data.bodies[body_b].vel
            omega_a = data.bodies[body_a].omega
            omega_b = data.bodies[body_b].omega
            inv_mass_a = data.bodies[body_a].inv_mass
            inv_mass_b = data.bodies[body_b].inv_mass

            # Vectors from body centers to contact point
            r_a = contact_point - pos_a
            r_b = contact_point - pos_b

            # Compute world-space inverse inertia tensors
            inv_I_a = compute_world_inv_inertia(body_a)
            inv_I_b = compute_world_inv_inertia(body_b)

            # Relative velocity at contact point
            # v_rel = (v_b + omega_b × r_b) - (v_a + omega_a × r_a)
            v_contact_a = vel_a + omega_a.cross(r_a)
            v_contact_b = vel_b + omega_b.cross(r_b)
            v_rel = v_contact_b - v_contact_a

            # Normal component of relative velocity (positive = separating)
            v_n = v_rel.dot(normal)

            # Compute effective mass K along normal
            # K = inv_mass_a + inv_mass_b + n · ((I_a^-1 * (r_a × n)) × r_a) + n · ((I_b^-1 * (r_b × n)) × r_b)
            r_a_cross_n = r_a.cross(normal)
            r_b_cross_n = r_b.cross(normal)

            angular_a = (inv_I_a @ r_a_cross_n).cross(r_a)
            angular_b = (inv_I_b @ r_b_cross_n).cross(r_b)

            K = inv_mass_a + inv_mass_b + normal.dot(angular_a) + normal.dot(angular_b)

            # Avoid division by zero for static-static (shouldn't happen, but be safe)
            if K > 1e-8:
                # Baumgarte bias: push apart if penetrating more than slop
                bias = 0.0
                penetration = depth - slop
                if penetration > 0:
                    bias = bias_factor * penetration

                # Restitution bias (only if approaching)
                if v_n < 0:
                    bias += -restitution * v_n

                # Compute impulse magnitude
                # We want: v_n_new = 0 (or positive for bounce)
                # impulse = -(v_n + bias) / K
                impulse_magnitude = -(v_n + bias) / K

                # Clamp: contacts can only push, not pull
                # (In full solver, we'd accumulate and clamp accumulated impulse)
                if impulse_magnitude < 0:
                    impulse_magnitude = 0.0

                # Apply impulse to both bodies
                impulse = normal * impulse_magnitude

                # Body A gets negative impulse (pushed away from contact)
                apply_impulse_to_body(body_a, -impulse, r_a, inv_I_a)

                # Body B gets positive impulse (pushed in normal direction)
                apply_impulse_to_body(body_b, impulse, r_b, inv_I_b)
