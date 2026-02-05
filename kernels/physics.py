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


@ti.kernel
def highlight_contact_bodies(num_contacts: ti.i32, num_bodies: ti.i32):
    """Highlight bodies that are in contact by changing their vertex colors.

    Bodies in contact get a red tint, others keep their original color.
    """
    # First, reset all dynamic body colors to original (slight desaturation to show difference)
    for b in range(num_bodies):
        if data.bodies[b].inv_mass > 0:  # Dynamic bodies only
            start = data.bodies[b].vert_start
            count = data.bodies[b].vert_count
            for v in range(start, start + count):
                # Reset to slightly dimmed original color
                data.vertex_colors[v] = data.vertex_colors[v] * 0.8

    # Then highlight bodies in contact with red
    for c in range(num_contacts):
        body_a = data.contacts[c].body_a
        body_b = data.contacts[c].body_b

        # Color body A vertices red (if dynamic)
        if data.bodies[body_a].inv_mass > 0:
            start_a = data.bodies[body_a].vert_start
            count_a = data.bodies[body_a].vert_count
            for v in range(start_a, start_a + count_a):
                data.vertex_colors[v] = ti.Vector([1.0, 0.3, 0.3])  # Red tint

        # Color body B vertices red (if dynamic)
        if data.bodies[body_b].inv_mass > 0:
            start_b = data.bodies[body_b].vert_start
            count_b = data.bodies[body_b].vert_count
            for v in range(start_b, start_b + count_b):
                data.vertex_colors[v] = ti.Vector([1.0, 0.3, 0.3])  # Red tint


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

        elif geom_type == data.GEOM_MESH:
            # Compute AABB from hull vertices transformed to world space
            vert_start = ti.cast(geom_data[0], ti.i32)
            vert_count = ti.cast(geom_data[1], ti.i32)
            aabb_min = ti.Vector([1e10, 1e10, 1e10])
            aabb_max = ti.Vector([-1e10, -1e10, -1e10])
            for vi in range(vert_count):
                local_v = data.collision_verts[vert_start + vi]
                world_v = world_pos + quat_rotate(world_quat, local_v)
                aabb_min = ti.min(aabb_min, world_v)
                aabb_max = ti.max(aabb_max, world_v)

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
def collide_sphere_plane(sphere_pos: ti.types.vector(3, ti.f32), sphere_radius: ti.f32,
                         plane_pos: ti.types.vector(3, ti.f32), plane_normal: ti.types.vector(3, ti.f32),
                         body_a: ti.i32, body_b: ti.i32,
                         geom_a: ti.i32, geom_b: ti.i32):
    """Sphere-plane collision detection.

    Returns: (has_contact, contact_point, normal, depth)
    Normal points from plane to sphere (A to B).
    """
    # Distance from sphere center to plane (signed)
    dist = (sphere_pos - plane_pos).dot(plane_normal)

    has_contact = 0
    contact_point = ti.Vector([0.0, 0.0, 0.0])
    normal = plane_normal
    depth = 0.0

    # Check if sphere penetrates plane
    if dist < sphere_radius:
        has_contact = 1
        depth = sphere_radius - dist
        contact_point = sphere_pos - plane_normal * dist
        # Normal points from plane (A) to sphere (B)
        normal = plane_normal

    return has_contact, contact_point, normal, depth


@ti.func
def collide_box_plane(box_pos: ti.types.vector(3, ti.f32), box_quat: ti.types.vector(4, ti.f32),
                      box_half: ti.types.vector(3, ti.f32),
                      plane_pos: ti.types.vector(3, ti.f32), plane_normal: ti.types.vector(3, ti.f32),
                      body_a: ti.i32, body_b: ti.i32,
                      geom_a: ti.i32, geom_b: ti.i32):
    """Box-plane collision detection.

    Returns: (has_contact, contact_point, normal, depth)
    Normal points from plane to box (A to B).
    """
    # Get box axes
    ax0 = quat_rotate(box_quat, ti.Vector([1.0, 0.0, 0.0]))
    ax1 = quat_rotate(box_quat, ti.Vector([0.0, 1.0, 0.0]))
    ax2 = quat_rotate(box_quat, ti.Vector([0.0, 0.0, 1.0]))

    # Find the box corner furthest in the -normal direction (most penetrating)
    # This is the support point in the -normal direction
    sign0 = -1.0 if ax0.dot(plane_normal) > 0.0 else 1.0
    sign1 = -1.0 if ax1.dot(plane_normal) > 0.0 else 1.0
    sign2 = -1.0 if ax2.dot(plane_normal) > 0.0 else 1.0

    # Most penetrating corner
    corner = box_pos + ax0 * box_half[0] * sign0 + ax1 * box_half[1] * sign1 + ax2 * box_half[2] * sign2

    # Distance from corner to plane
    dist = (corner - plane_pos).dot(plane_normal)

    has_contact = 0
    contact_point = ti.Vector([0.0, 0.0, 0.0])
    normal = plane_normal
    depth = 0.0

    # If corner is below plane, we have contact
    if dist < 0.0:
        has_contact = 1
        depth = -dist
        contact_point = corner - plane_normal * dist
        normal = plane_normal

    return has_contact, contact_point, normal, depth


@ti.func
def collide_mesh_plane(mesh_pos: ti.types.vector(3, ti.f32), mesh_quat: ti.types.vector(4, ti.f32),
                       vert_start: ti.i32, vert_count: ti.i32,
                       plane_pos: ti.types.vector(3, ti.f32), plane_normal: ti.types.vector(3, ti.f32),
                       body_a: ti.i32, body_b: ti.i32, geom_a: ti.i32, geom_b: ti.i32):
    """Mesh-plane collision - find deepest vertex below plane."""
    has_contact = 0
    contact_point = ti.Vector([0.0, 0.0, 0.0])
    normal = plane_normal
    max_depth = 0.0

    for vi in range(vert_count):
        local_v = data.collision_verts[vert_start + vi]
        world_v = mesh_pos + quat_rotate(mesh_quat, local_v)
        dist = (world_v - plane_pos).dot(plane_normal)

        if dist < 0.0:
            depth = -dist
            if depth > max_depth:
                max_depth = depth
                contact_point = world_v - plane_normal * dist
                has_contact = 1

    return has_contact, contact_point, normal, max_depth


@ti.func
def collide_mesh_sphere(mesh_pos: ti.types.vector(3, ti.f32), mesh_quat: ti.types.vector(4, ti.f32),
                        face_start: ti.i32, face_count: ti.i32,
                        sphere_pos: ti.types.vector(3, ti.f32), sphere_radius: ti.f32,
                        body_a: ti.i32, body_b: ti.i32, geom_a: ti.i32, geom_b: ti.i32):
    """Mesh-sphere collision - check sphere against hull faces. A=sphere, B=mesh."""
    has_contact = 0
    contact_point = ti.Vector([0.0, 0.0, 0.0])
    normal = ti.Vector([0.0, 1.0, 0.0])
    depth = 0.0

    # For convex hull: find closest face to sphere center
    # Signed distance: positive = outside hull, negative = inside hull
    max_dist = -1e10
    best_normal = ti.Vector([0.0, 1.0, 0.0])
    best_point = ti.Vector([0.0, 0.0, 0.0])

    for fi in range(face_count):
        face = data.collision_faces[face_start + fi]
        v0 = mesh_pos + quat_rotate(mesh_quat, data.collision_verts[face[0]])
        v1 = mesh_pos + quat_rotate(mesh_quat, data.collision_verts[face[1]])
        v2 = mesh_pos + quat_rotate(mesh_quat, data.collision_verts[face[2]])

        # Face normal (outward from hull)
        fn = (v1 - v0).cross(v2 - v0)
        fn_len = fn.norm()
        if fn_len > 1e-10:
            fn = fn / fn_len
            dist = (sphere_pos - v0).dot(fn)

            if dist > max_dist:
                max_dist = dist
                best_normal = fn
                best_point = sphere_pos - fn * dist

    # Contact if sphere penetrates: max_dist < sphere_radius
    if max_dist < sphere_radius:
        depth = sphere_radius - max_dist
        # Normal points from A (sphere) to B (mesh) = -outward
        normal = -best_normal
        contact_point = best_point
        has_contact = 1

    return has_contact, contact_point, normal, depth


@ti.func
def collide_mesh_box(mesh_pos: ti.types.vector(3, ti.f32), mesh_quat: ti.types.vector(4, ti.f32),
                     vert_start: ti.i32, vert_count: ti.i32,
                     box_pos: ti.types.vector(3, ti.f32), box_quat: ti.types.vector(4, ti.f32),
                     box_half: ti.types.vector(3, ti.f32),
                     body_a: ti.i32, body_b: ti.i32, geom_a: ti.i32, geom_b: ti.i32):
    """Mesh-box collision. A=box, B=mesh. Normal points from A to B."""
    has_contact = 0
    contact_point = ti.Vector([0.0, 0.0, 0.0])
    normal = ti.Vector([0.0, 1.0, 0.0])
    max_depth = 0.0

    ax0 = quat_rotate(box_quat, ti.Vector([1.0, 0.0, 0.0]))
    ax1 = quat_rotate(box_quat, ti.Vector([0.0, 1.0, 0.0]))
    ax2 = quat_rotate(box_quat, ti.Vector([0.0, 0.0, 1.0]))

    for vi in range(vert_count):
        local_v = data.collision_verts[vert_start + vi]
        world_v = mesh_pos + quat_rotate(mesh_quat, local_v)

        rel = world_v - box_pos
        local_x = rel.dot(ax0)
        local_y = rel.dot(ax1)
        local_z = rel.dot(ax2)

        dx = ti.abs(local_x) - box_half[0]
        dy = ti.abs(local_y) - box_half[1]
        dz = ti.abs(local_z) - box_half[2]

        if dx < 0 and dy < 0 and dz < 0:
            depth = -ti.max(dx, ti.max(dy, dz))
            if depth > max_depth:
                max_depth = depth
                has_contact = 1
                contact_point = world_v
                # Normal points from box (A) toward mesh vertex (B)
                if dx > dy and dx > dz:
                    normal = ax0 if local_x > 0 else -ax0
                elif dy > dz:
                    normal = ax1 if local_y > 0 else -ax1
                else:
                    normal = ax2 if local_z > 0 else -ax2

    return has_contact, contact_point, normal, max_depth


@ti.func
def collide_mesh_mesh(pos_a: ti.types.vector(3, ti.f32), quat_a: ti.types.vector(4, ti.f32),
                      vert_start_a: ti.i32, vert_count_a: ti.i32, face_start_a: ti.i32, face_count_a: ti.i32,
                      pos_b: ti.types.vector(3, ti.f32), quat_b: ti.types.vector(4, ti.f32),
                      vert_start_b: ti.i32, vert_count_b: ti.i32, face_start_b: ti.i32, face_count_b: ti.i32,
                      body_a: ti.i32, body_b: ti.i32, geom_a: ti.i32, geom_b: ti.i32):
    """Mesh-mesh collision - check vertices of A against hull B and vice versa."""
    has_contact = 0
    contact_point = ti.Vector([0.0, 0.0, 0.0])
    normal = ti.Vector([0.0, 1.0, 0.0])
    max_depth = 0.0

    # Check vertices of mesh A against faces of mesh B
    for vi in range(vert_count_a):
        local_v = data.collision_verts[vert_start_a + vi]
        world_v = pos_a + quat_rotate(quat_a, local_v)

        # Check if inside hull B by testing against all faces
        inside = 1
        min_dist = 1e10
        closest_normal = ti.Vector([0.0, 1.0, 0.0])

        for fi in range(face_count_b):
            face = data.collision_faces[face_start_b + fi]
            v0 = pos_b + quat_rotate(quat_b, data.collision_verts[face[0]])
            v1 = pos_b + quat_rotate(quat_b, data.collision_verts[face[1]])
            v2 = pos_b + quat_rotate(quat_b, data.collision_verts[face[2]])

            # Face normal (outward)
            fn = (v1 - v0).cross(v2 - v0)
            fn_len = fn.norm()
            if fn_len > 1e-10:
                fn = fn / fn_len

            # Signed distance from vertex to face plane
            dist = (world_v - v0).dot(fn)

            if dist > 0:
                inside = 0  # Outside this face = outside hull
            elif -dist < min_dist:
                min_dist = -dist
                closest_normal = -fn  # Points from B to A

        if inside == 1 and min_dist < 1e10:
            if min_dist > max_depth:
                max_depth = min_dist
                contact_point = world_v
                normal = closest_normal
                has_contact = 1

    # Check vertices of mesh B against faces of mesh A
    for vi in range(vert_count_b):
        local_v = data.collision_verts[vert_start_b + vi]
        world_v = pos_b + quat_rotate(quat_b, local_v)

        inside = 1
        min_dist = 1e10
        closest_normal = ti.Vector([0.0, 1.0, 0.0])

        for fi in range(face_count_a):
            face = data.collision_faces[face_start_a + fi]
            v0 = pos_a + quat_rotate(quat_a, data.collision_verts[face[0]])
            v1 = pos_a + quat_rotate(quat_a, data.collision_verts[face[1]])
            v2 = pos_a + quat_rotate(quat_a, data.collision_verts[face[2]])

            fn = (v1 - v0).cross(v2 - v0)
            fn_len = fn.norm()
            if fn_len > 1e-10:
                fn = fn / fn_len

            dist = (world_v - v0).dot(fn)

            if dist > 0:
                inside = 0
            elif -dist < min_dist:
                min_dist = -dist
                closest_normal = fn  # Points from A to B

        if inside == 1 and min_dist < 1e10:
            if min_dist > max_depth:
                max_depth = min_dist
                contact_point = world_v
                normal = closest_normal
                has_contact = 1

    return has_contact, contact_point, normal, max_depth


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

    # Compute contact point
    contact_point = ti.Vector([0.0, 0.0, 0.0])
    depth = 0.0
    normal = best_axis

    if has_contact == 1:
        depth = min_overlap
        # Contact point: project box B's center onto the contact surface of box A.
        #
        # For face-face contact (box sitting on ground), this gives a point
        # directly under box B on the surface of box A - the actual contact region.
        #
        # The contact surface of A in the normal direction is at:
        #   pos_a + normal * (extent of A along normal)
        extent_a_along_normal = (ti.abs(ax_a0.dot(normal)) * half_a[0] +
                                  ti.abs(ax_a1.dot(normal)) * half_a[1] +
                                  ti.abs(ax_a2.dot(normal)) * half_a[2])
        surface_height_a = pos_a.dot(normal) + extent_a_along_normal

        # Project pos_b onto the contact plane (keep x,z of B, set height to surface)
        # contact_point = pos_b projected onto plane with normal at surface_height_a
        pos_b_along_normal = pos_b.dot(normal)
        contact_point = pos_b + normal * (surface_height_a - pos_b_along_normal + depth * 0.5)

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

        # Canonicalize: ensure type_a <= type_b to avoid duplicate collision cases
        # Swap if type_a > type_b
        if type_a > type_b:
            geom_a_idx, geom_b_idx = geom_b_idx, geom_a_idx
            type_a, type_b = type_b, type_a

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

        # Box-Box: Generate 4 corner contacts to prevent rotation
        elif type_a == data.GEOM_BOX and type_b == data.GEOM_BOX:
            half_a = ti.Vector([data_a[0], data_a[1], data_a[2]])
            half_b = ti.Vector([data_b[0], data_b[1], data_b[2]])
            has_contact, contact_point, normal, depth = collide_box_box(
                pos_a, quat_a, half_a, pos_b, quat_b, half_b, body_a, body_b, geom_a_idx, geom_b_idx)

            if has_contact == 1 and depth > 0:
                # Generate 4 contact points at corners of box B's contact face
                # Find the two axes of box B perpendicular to the contact normal
                ax_b0 = quat_rotate(quat_b, ti.Vector([1.0, 0.0, 0.0]))
                ax_b1 = quat_rotate(quat_b, ti.Vector([0.0, 1.0, 0.0]))
                ax_b2 = quat_rotate(quat_b, ti.Vector([0.0, 0.0, 1.0]))

                # Find which face of B is closest to facing -normal (the contact face)
                dot0 = ti.abs(ax_b0.dot(normal))
                dot1 = ti.abs(ax_b1.dot(normal))
                dot2 = ti.abs(ax_b2.dot(normal))

                # Initialize tangent vectors (will be set based on contact face)
                tangent1 = ti.Vector([0.0, 0.0, 0.0])
                tangent2 = ti.Vector([0.0, 0.0, 0.0])

                # The face normal is the axis most aligned with collision normal
                # The other two axes span the contact face
                if dot0 >= dot1 and dot0 >= dot2:
                    # X-face: span by Y and Z
                    tangent1 = ax_b1 * half_b[1]
                    tangent2 = ax_b2 * half_b[2]
                elif dot1 >= dot0 and dot1 >= dot2:
                    # Y-face: span by X and Z
                    tangent1 = ax_b0 * half_b[0]
                    tangent2 = ax_b2 * half_b[2]
                else:
                    # Z-face: span by X and Y
                    tangent1 = ax_b0 * half_b[0]
                    tangent2 = ax_b1 * half_b[1]

                # 4 corners of box B's contact face (relative to center contact point)
                # Use the center contact_point as base and offset to corners
                for i in ti.static(range(4)):
                    sign1 = 1.0 if (i & 1) == 0 else -1.0
                    sign2 = 1.0 if (i & 2) == 0 else -1.0
                    corner_offset = tangent1 * sign1 + tangent2 * sign2
                    corner_point = contact_point + corner_offset

                    slot = ti.atomic_add(data.num_contacts[None], 1)
                    if slot < data.MAX_CONTACTS:
                        data.contacts[slot].point = corner_point
                        data.contacts[slot].normal = normal
                        data.contacts[slot].depth = depth
                        data.contacts[slot].body_a = body_a
                        data.contacts[slot].body_b = body_b
                        data.contacts[slot].geom_a = geom_a_idx
                        data.contacts[slot].geom_b = geom_b_idx

            has_contact = 0  # Already handled, skip generic add below

        # Sphere-Plane
        elif type_a == data.GEOM_SPHERE and type_b == data.GEOM_PLANE:
            radius = data_a[0]
            plane_normal = ti.Vector([data_b[0], data_b[1], data_b[2]])
            has_contact, contact_point, normal, depth = collide_sphere_plane(
                pos_a, radius, pos_b, plane_normal, body_a, body_b, geom_a_idx, geom_b_idx)

        # Box-Plane
        elif type_a == data.GEOM_BOX and type_b == data.GEOM_PLANE:
            half_a = ti.Vector([data_a[0], data_a[1], data_a[2]])
            plane_normal = ti.Vector([data_b[0], data_b[1], data_b[2]])
            has_contact, contact_point, normal, depth = collide_box_plane(
                pos_a, quat_a, half_a, pos_b, plane_normal, body_a, body_b, geom_a_idx, geom_b_idx)

        # Mesh-Plane
        elif type_a == data.GEOM_MESH and type_b == data.GEOM_PLANE:
            vert_start = ti.cast(data_a[0], ti.i32)
            vert_count = ti.cast(data_a[1], ti.i32)
            plane_normal = ti.Vector([data_b[0], data_b[1], data_b[2]])
            has_contact, contact_point, normal, depth = collide_mesh_plane(
                pos_a, quat_a, vert_start, vert_count, pos_b, plane_normal,
                body_a, body_b, geom_a_idx, geom_b_idx)

        # Sphere-Mesh (SPHERE=1 < MESH=5)
        elif type_a == data.GEOM_SPHERE and type_b == data.GEOM_MESH:
            radius = data_a[0]
            face_start = ti.cast(data_b[2], ti.i32)
            face_count = ti.cast(data_b[3], ti.i32)
            has_contact, contact_point, normal, depth = collide_mesh_sphere(
                pos_b, quat_b, face_start, face_count, pos_a, radius,
                body_a, body_b, geom_a_idx, geom_b_idx)

        # Box-Mesh (BOX=2 < MESH=5)
        elif type_a == data.GEOM_BOX and type_b == data.GEOM_MESH:
            half_a = ti.Vector([data_a[0], data_a[1], data_a[2]])
            vert_start = ti.cast(data_b[0], ti.i32)
            vert_count = ti.cast(data_b[1], ti.i32)
            has_contact, contact_point, normal, depth = collide_mesh_box(
                pos_b, quat_b, vert_start, vert_count, pos_a, quat_a, half_a,
                body_a, body_b, geom_a_idx, geom_b_idx)

        # Mesh-Mesh (MESH=5, MESH=5) - skip if same body
        elif type_a == data.GEOM_MESH and type_b == data.GEOM_MESH:
            if body_a != body_b:  # Don't collide geoms on same body
                vert_start_a = ti.cast(data_a[0], ti.i32)
                vert_count_a = ti.cast(data_a[1], ti.i32)
                face_start_a = ti.cast(data_a[2], ti.i32)
                face_count_a = ti.cast(data_a[3], ti.i32)
                vert_start_b = ti.cast(data_b[0], ti.i32)
                vert_count_b = ti.cast(data_b[1], ti.i32)
                face_start_b = ti.cast(data_b[2], ti.i32)
                face_count_b = ti.cast(data_b[3], ti.i32)
                has_contact, contact_point, normal, depth = collide_mesh_mesh(
                    pos_a, quat_a, vert_start_a, vert_count_a, face_start_a, face_count_a,
                    pos_b, quat_b, vert_start_b, vert_count_b, face_start_b, face_count_b,
                    body_a, body_b, geom_a_idx, geom_b_idx)

        # Add contact if found (for non-box-box collisions)
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
# NCP Contact Solver - Step by Step
# =============================================================================
#
# This solver finds impulses λ that prevent penetration while respecting:
#   λ ≥ 0        (can only push, not pull)
#   v_n ≥ 0      (bodies separate or stay in contact)
#   λ · v_n = 0  (complementarity: impulse only when in contact)
#
# =============================================================================


@ti.func
def compute_world_inv_inertia(body_idx: ti.i32) -> ti.types.matrix(3, 3, ti.f32): 
    inv_I_local = data.bodies[body_idx].inv_inertia  # vec3
    quat = data.bodies[body_idx].quat                 # orientation quaternion

    r0 = quat_rotate(quat, ti.Vector([1.0, 0.0, 0.0]))  # Local X in world
    r1 = quat_rotate(quat, ti.Vector([0.0, 1.0, 0.0]))  # Local Y in world
    r2 = quat_rotate(quat, ti.Vector([0.0, 0.0, 1.0]))  # Local Z in world

    result = ti.Matrix.zero(ti.f32, 3, 3)

    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            result[i, j] = (inv_I_local[0] * r0[i] * r0[j] +
                           inv_I_local[1] * r1[i] * r1[j] +
                           inv_I_local[2] * r2[i] * r2[j])

    return result


@ti.kernel
def solve_contacts_prestep(num_contacts: ti.i32, dt: ti.f32):
    beta = 0.2      # Baumgarte factor: how aggressively to fix penetration (0.1-0.3)
    slop = 0.001    # Allow 5mm penetration before correction (prevents jitter)
    restitution = 0.3  # Bounciness (0 = clay, 1 = superball)

    for c in range(num_contacts):
        contact_point = data.contacts[c].point   # Where bodies touch (world space)
        normal = data.contacts[c].normal          # Direction to push (A → B)
        depth = data.contacts[c].depth            # How deep they overlap (positive)

        body_a = data.contacts[c].body_a          # Index of first body
        body_b = data.contacts[c].body_b          # Index of second body

        pos_a = data.bodies[body_a].pos           # Center of mass position
        pos_b = data.bodies[body_b].pos
        vel_a = data.bodies[body_a].vel           # Linear velocity
        vel_b = data.bodies[body_b].vel
        omega_a = data.bodies[body_a].omega       # Angular velocity (rad/s)
        omega_b = data.bodies[body_b].omega
        inv_mass_a = data.bodies[body_a].inv_mass # 1/mass (0 for static bodies)
        inv_mass_b = data.bodies[body_b].inv_mass

        # Step 3: Compute r vectors (contact point relative to body center)
        r_a = contact_point - pos_a
        r_b = contact_point - pos_b

        inv_I_a = compute_world_inv_inertia(body_a)
        inv_I_b = compute_world_inv_inertia(body_b)

        # =====================================================================
        # Step 5: Compute effective mass K along normal
        #
        # When we apply impulse J = λ·n, velocity changes by:
        #   Δv_n = K · λ
        #
        # K has two parts:
        #   Linear:  inv_mass_a + inv_mass_b  (both bodies get pushed)
        #   Angular: n · (I⁻¹ · (r × n)) × r  (rotation contributes to contact velocity)
        #
        # We store 1/K so iteration is just: Δλ = (1/K) · violation
        # =====================================================================

        # r × n: torque direction when impulse n is applied at offset r
        r_a_cross_n = r_a.cross(normal)
        r_b_cross_n = r_b.cross(normal)

        # I⁻¹ · (r × n): angular velocity change from impulse
        # Then cross with r again: contribution to linear velocity at contact
        # Then dot with n: only the normal component matters
        angular_a = (inv_I_a @ r_a_cross_n).cross(r_a)
        angular_b = (inv_I_b @ r_b_cross_n).cross(r_b)

        K = inv_mass_a + inv_mass_b + normal.dot(angular_a) + normal.dot(angular_b)

        # Store effective mass (guard against division by zero for static-static)
        if K > 1e-8:
            data.contacts[c].mass_normal = 1.0 / K
        else:
            data.contacts[c].mass_normal = 0.0

        # =====================================================================
        # Step 6: Compute current relative velocity at contact
        #
        # v_contact = v_center + ω × r  (velocity at contact point)
        # v_rel = v_b - v_a             (relative velocity)
        # v_n = v_rel · n               (normal component: + = separating)
        # =====================================================================
        v_contact_a = vel_a + omega_a.cross(r_a)
        v_contact_b = vel_b + omega_b.cross(r_b)
        v_rel = v_contact_b - v_contact_a
        v_n = v_rel.dot(normal)

        # =====================================================================
        # Step 7: Compute bias (target velocity we want after solving)
        #
        # Two reasons to add bias:
        #
        # A) Position correction (Baumgarte stabilization):
        #    If bodies overlap, we want small velocity INTO each other
        #    to push them apart over several frames.
        #    bias_pos = -β/dt · max(0, depth - slop)
        #
        # B) Restitution (bounce):
        #    If approaching fast, reflect some velocity.
        #    bias_restitution = -e · v_n  (only if v_n < 0, i.e., approaching)
        #
        # We use max() to pick the larger effect (not both, to avoid energy gain)
        # =====================================================================
        inv_dt = 1.0 / dt if dt > 1e-8 else 0.0

        # Position correction bias
        penetration = depth - slop
        bias_position = 0.0
        if penetration > 0:
            bias_position = beta * inv_dt * penetration

        # Restitution bias (only for fast impacts, not resting contact)
        bias_restitution = 0.0
        if v_n < -0.5:  # Threshold to avoid jitter at rest
            bias_restitution = -restitution * v_n

        # Take the larger of the two
        data.contacts[c].bias = ti.max(bias_position, bias_restitution)

        # Initialize accumulated impulse to zero
        data.contacts[c].Pn = 0.0


@ti.kernel
def solve_contacts_iterate_once(num_contacts: ti.i32):
    """
    Single iteration of the constraint solver.

    Goes through ALL contacts sequentially (Gauss-Seidel).
    Called multiple times from Python to complete the solve.

    The Jacobian J maps body velocities to constraint velocity:
        v_n = J · V

    For one contact, J has the form:
        J = [-n, -(r_a × n), +n, +(r_b × n)]

    Newton step to find impulse:
        constraint_error = v_n - bias  (how much we violate target)
        Δλ = -error / K = mass_normal · (-v_n + bias)

    Complementarity projection:
        λ_new = max(0, λ_old + Δλ)  (can't pull, only push)

    Apply impulse:
        body_a: vel -= λ·n/m,  ω -= I⁻¹·(r_a × λ·n)
        body_b: vel += λ·n/m,  ω += I⁻¹·(r_b × λ·n)
    """
    # CRITICAL: Sequential iteration (Gauss-Seidel)
    # Each contact sees updated velocities from previous contacts
    ti.loop_config(serialize=True)
    for c in range(num_contacts):
            # Skip if no effective mass (static-static collision)
            mass_normal = data.contacts[c].mass_normal
            if mass_normal < 1e-8:
                continue

            # =================================================================
            # Step 1: Get contact and body data
            # =================================================================
            contact_point = data.contacts[c].point
            normal = data.contacts[c].normal
            bias = data.contacts[c].bias  # Pre-computed in prestep

            body_a = data.contacts[c].body_a
            body_b = data.contacts[c].body_b

            # Get CURRENT velocities (updated by previous contacts this iteration)
            pos_a = data.bodies[body_a].pos
            pos_b = data.bodies[body_b].pos
            vel_a = data.bodies[body_a].vel
            vel_b = data.bodies[body_b].vel
            omega_a = data.bodies[body_a].omega
            omega_b = data.bodies[body_b].omega
            inv_mass_a = data.bodies[body_a].inv_mass
            inv_mass_b = data.bodies[body_b].inv_mass

            # r vectors (contact point relative to body center)
            r_a = contact_point - pos_a
            r_b = contact_point - pos_b

            # =================================================================
            # Step 2: Compute current relative velocity v_n
            #
            # This is the Jacobian applied to current velocities:
            #   v_n = J · V = n · (v_b + ω_b × r_b - v_a - ω_a × r_a)
            # =================================================================
            v_contact_a = vel_a + omega_a.cross(r_a)
            v_contact_b = vel_b + omega_b.cross(r_b)
            v_rel = v_contact_b - v_contact_a
            v_n = v_rel.dot(normal)

            # =================================================================
            # Step 3: Newton step - compute impulse change
            #
            # We want: v_n_new = bias (target velocity)
            # Currently: v_n (actual velocity)
            # Error: v_n - bias
            #
            # Impulse to fix: Δλ = -error · (1/K) = mass_normal · (-v_n + bias)
            # =================================================================
            dPn = mass_normal * (-v_n + bias)

            # =================================================================
            # Step 4: Accumulated impulse clamping (complementarity)
            #
            # Key insight from Box2D: clamp the TOTAL impulse, not the delta.
            #
            # Why? If we clamp delta only:
            #   - Iteration 1: Pn=0, dPn=-5 → clamp to 0 → Pn=0
            #   - Iteration 2: Pn=0, dPn=-5 → clamp to 0 → Pn=0
            #   (stuck at zero even if we need positive impulse)
            #
            # With accumulated clamping:
            #   - Iteration 1: Pn=0, dPn=3 → Pn_new=max(0,3)=3, apply dPn=3
            #   - Iteration 2: Pn=3, dPn=-5 → Pn_new=max(0,-2)=0, apply dPn=-3
            #   (correctly removes excess impulse)
            # =================================================================
            Pn_old = data.contacts[c].Pn
            Pn_new = ti.max(0.0, Pn_old + dPn)
            data.contacts[c].Pn = Pn_new
            dPn = Pn_new - Pn_old  # Actual impulse to apply

            # =================================================================
            # Step 5: Apply impulse to both bodies
            #
            # Impulse vector: P = dPn · n
            #
            # Body A receives -P (pushed opposite to normal):
            #   Δvel_a = -P · inv_mass_a
            #   Δω_a = I_a⁻¹ · (r_a × -P) = -I_a⁻¹ · (r_a × P)
            #
            # Body B receives +P (pushed along normal):
            #   Δvel_b = +P · inv_mass_b
            #   Δω_b = I_b⁻¹ · (r_b × P)
            # =================================================================
            impulse = normal * dPn

            # Apply linear impulse
            data.bodies[body_a].vel = vel_a - impulse * inv_mass_a
            data.bodies[body_b].vel = vel_b + impulse * inv_mass_b

            # Apply angular impulse
            # Need world-space inverse inertia for angular velocity change
            inv_I_a = compute_world_inv_inertia(body_a)
            inv_I_b = compute_world_inv_inertia(body_b)

            # Torque from impulse at offset r: τ = r × P
            torque_a = r_a.cross(impulse)
            torque_b = r_b.cross(impulse)

            # Angular velocity change: Δω = I⁻¹ · τ
            data.bodies[body_a].omega = omega_a - inv_I_a @ torque_a
            data.bodies[body_b].omega = omega_b + inv_I_b @ torque_b


def solve_contacts(num_contacts: int, num_iterations: int, dt: float):
    """
    Full contact solver: prestep + iterations.

    Call this from the physics loop after narrow phase.
    """
    if num_contacts == 0:
        return

    # Phase 1: Compute constants (mass_normal, bias)
    solve_contacts_prestep(num_contacts, dt)

    # Phase 2: Iteratively apply impulses (Python loop ensures sequential iterations)
    for _ in range(num_iterations):
        solve_contacts_iterate_once(num_contacts)


# =============================================================================
# Debug rendering for collision geoms
# =============================================================================

@ti.kernel
def build_debug_geom_verts(num_geoms: ti.i32, num_collision_verts: ti.i32, num_collision_faces: ti.i32):
    """Transform collision vertices to world space and copy face indices for debug rendering."""
    # Transform all collision vertices to world space
    for gi in range(num_geoms):
        geom = data.geoms[gi]
        if geom.geom_type == data.GEOM_MESH:
            vert_start = ti.cast(geom.data[0], ti.i32)
            vert_count = ti.cast(geom.data[1], ti.i32)

            body_idx = geom.body_idx
            body_pos = data.bodies[body_idx].pos
            body_quat = data.bodies[body_idx].quat

            # Random color per geom
            r = ti.cast((gi * 123) % 255, ti.f32) / 255.0
            g = ti.cast((gi * 456 + 100) % 255, ti.f32) / 255.0
            b = ti.cast((gi * 789 + 50) % 255, ti.f32) / 255.0
            color = ti.Vector([r, g, b])

            for vi in range(vert_count):
                idx = vert_start + vi
                local_v = data.collision_verts[idx]
                world_v = quat_rotate(body_quat, local_v) + body_pos
                data.debug_geom_verts[idx] = world_v
                data.debug_geom_colors[idx] = color

    # Copy all collision face indices to debug indices (flattened)
    # Also compute normal arrows for each face
    for fi in range(num_collision_faces):
        src_face = data.collision_faces[fi]
        data.debug_geom_indices[fi * 3 + 0] = src_face[0]
        data.debug_geom_indices[fi * 3 + 1] = src_face[1]
        data.debug_geom_indices[fi * 3 + 2] = src_face[2]

        # Compute face center and normal for debug arrow
        v0 = data.debug_geom_verts[src_face[0]]
        v1 = data.debug_geom_verts[src_face[1]]
        v2 = data.debug_geom_verts[src_face[2]]
        center = (v0 + v1 + v2) / 3.0
        fn = (v1 - v0).cross(v2 - v0)
        fn_len = fn.norm()
        if fn_len > 1e-10:
            fn = fn / fn_len
        # Arrow from center to center + normal * 0.1
        data.debug_normal_verts[fi * 2 + 0] = center
        data.debug_normal_verts[fi * 2 + 1] = center + fn * 0.1
        data.debug_normal_colors[fi * 2 + 0] = ti.Vector([0.0, 1.0, 0.0])
        data.debug_normal_colors[fi * 2 + 1] = ti.Vector([1.0, 0.0, 0.0])


@ti.kernel
def build_debug_contacts(num_contacts: ti.i32):
    """Build debug visualization for contact points and normals."""
    for i in range(num_contacts):
        pt = data.contacts[i].point
        n = data.contacts[i].normal
        data.debug_contact_points[i] = pt
        # Normal arrow from contact point
        data.debug_contact_normals[i * 2 + 0] = pt
        data.debug_contact_normals[i * 2 + 1] = pt + n * 0.2  # 0.2 length arrow

