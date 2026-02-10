import taichi as ti
from data import data, GEOM_SPHERE, GEOM_BOX, MPR_EPS, MPR_TOLERANCE, MPR_MAX_ITERATIONS
from utils import quat_conjugate, quat_from_angular_velocity, quat_mul, quat_normalize, quat_rotate

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
def update_geom_transforms(num_geoms: ti.i32):
    for i in range(num_geoms):
        body = data.bodies[data.geoms[i].body_idx]
        data.geoms[i].world_pos = body.pos + quat_rotate(body.quat, data.geoms[i].local_pos)
        data.geoms[i].world_quat = quat_mul(body.quat, data.geoms[i].local_quat)

# --- MPR support functions ---

@ti.func
def support_sphere(gi: ti.i32, direction: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    pos = data.geoms[gi].world_pos
    radius = data.geoms[gi].data[0]
    return pos + direction.normalized() * radius

@ti.func
def support_box(gi: ti.i32, direction: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    pos = data.geoms[gi].world_pos
    q = data.geoms[gi].world_quat
    d_local = quat_rotate(quat_conjugate(q), direction)
    corner = ti.Vector([
        (1.0 if d_local[0] >= 0.0 else -1.0) * data.geoms[gi].data[0],
        (1.0 if d_local[1] >= 0.0 else -1.0) * data.geoms[gi].data[1],
        (1.0 if d_local[2] >= 0.0 else -1.0) * data.geoms[gi].data[2],
    ])
    return pos + quat_rotate(q, corner)

@ti.func
def support(gi: ti.i32, direction: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    v = ti.Vector([0.0, 0.0, 0.0])
    if data.geoms[gi].geom_type == GEOM_SPHERE:
        v = support_sphere(gi, direction)
    elif data.geoms[gi].geom_type == GEOM_BOX:
        v = support_box(gi, direction)
    return v

@ti.func
def compute_support(direction: ti.types.vector(3, ti.f32), gi_a: ti.i32, gi_b: ti.i32):
    v1 = support(gi_a, direction)
    v2 = support(gi_b, -direction)
    v = v1 - v2
    return v, v1, v2

@ti.func
def mpr_discover_portal(gi_a: ti.i32, gi_b: ti.i32):
    # v0: Minkowski difference of centers
    center_a = data.geoms[gi_a].world_pos
    center_b = data.geoms[gi_b].world_pos
    v0 = center_a - center_b
    v0_a = center_a
    v0_b = center_b

    # Nudge if centers nearly overlap
    if ti.abs(v0[0]) < MPR_EPS and ti.abs(v0[1]) < MPR_EPS and ti.abs(v0[2]) < MPR_EPS:
        v0[0] += 10.0 * MPR_EPS

    # v1: support in direction toward origin
    direction = -v0.normalized()
    v1, v1_a, v1_b = compute_support(direction, gi_a, gi_b)

    # Init v2, v3 placeholders
    v2 = ti.Vector([0.0, 0.0, 0.0])
    v2_a = ti.Vector([0.0, 0.0, 0.0])
    v2_b = ti.Vector([0.0, 0.0, 0.0])
    v3 = ti.Vector([0.0, 0.0, 0.0])
    v3_a = ti.Vector([0.0, 0.0, 0.0])
    v3_b = ti.Vector([0.0, 0.0, 0.0])

    ret = 0
    # Check if v1 reached past origin
    if v1.dot(direction) < MPR_EPS:
        ret = -1
    else:
        # Find v2: perpendicular to v0-v1 line
        direction = v0.cross(v1)
        if direction.dot(direction) < MPR_EPS:
            # Collinear
            if ti.abs(v1[0]) < MPR_EPS and ti.abs(v1[1]) < MPR_EPS and ti.abs(v1[2]) < MPR_EPS:
                ret = 1  # touching
            else:
                ret = 2  # origin on segment
        else:
            direction = direction.normalized()
            v2, v2_a, v2_b = compute_support(direction, gi_a, gi_b)
            if v2.dot(direction) < MPR_EPS:
                ret = -1
            else:
                # Orient triangle normal toward origin
                va = v1 - v0
                vb = v2 - v0
                direction = va.cross(vb).normalized()

                if direction.dot(v0) > 0.0:
                    # Swap v1 and v2
                    tmp = v1; v1 = v2; v2 = tmp
                    tmp = v1_a; v1_a = v2_a; v2_a = tmp
                    tmp = v1_b; v1_b = v2_b; v2_b = tmp
                    direction = -direction

                # Find v3: expand triangle to tetrahedron
                num_trials = 0
                found = 0
                while found == 0:
                    v3, v3_a, v3_b = compute_support(direction, gi_a, gi_b)
                    if v3.dot(direction) < MPR_EPS:
                        ret = -1
                        break

                    # Check if origin is outside v0-v1 edge
                    cont = 0
                    if v1.cross(v3).dot(v0) < -MPR_EPS:
                        v2 = v3; v2_a = v3_a; v2_b = v3_b
                        cont = 1

                    # Check if origin is outside v0-v2 edge
                    if cont == 0:
                        if v3.cross(v2).dot(v0) < -MPR_EPS:
                            v1 = v3; v1_a = v3_a; v1_b = v3_b
                            cont = 1

                    if cont == 1:
                        # Recompute triangle normal
                        va = v1 - v0
                        vb = v2 - v0
                        direction = va.cross(vb).normalized()
                        num_trials += 1
                        if num_trials == 15:
                            ret = -1
                            break
                    else:
                        # v3 completes the tetrahedron
                        found = 1

    return ret, v0, v1, v2, v3, v0_a, v1_a, v2_a, v3_a, v0_b, v1_b, v2_b, v3_b

@ti.func
def mpr_find_penetr_touch(v0, v1_a, v1_b):
    normal = -v0.normalized()
    pos = (v1_a + v1_b) * 0.5
    penetration = 0.0
    return normal, pos, penetration

@ti.func
def mpr_portal_dir(v1, v2, v3):
    # Portal face normal (v1-v2-v3 triangle)
    return (v2 - v1).cross(v3 - v1).normalized()

@ti.func
def mpr_portal_encapsules_origin(v1, direction):
    # Is origin on the portal side? (v1 projected onto normal >= 0)
    return v1.dot(direction) > -MPR_EPS

@ti.func
def mpr_portal_can_encapsule_origin(v, direction):
    # Did new support point reach past origin?
    return v.dot(direction) > -MPR_EPS

@ti.func
def mpr_portal_reach_tolerance(v1, v2, v3, v, direction):
    # Is new point basically same distance as current portal?
    dv1 = v1.dot(direction)
    dv2 = v2.dot(direction)
    dv3 = v3.dot(direction)
    dv4 = v.dot(direction)
    dot1 = ti.min(dv4 - dv1, dv4 - dv2, dv4 - dv3)
    return dot1 < MPR_TOLERANCE + MPR_EPS * ti.max(1.0, dot1)

@ti.func
def mpr_expand_portal(v0, v1, v2, v3, v1_a, v2_a, v3_a, v1_b, v2_b, v3_b, v, va, vb):
    # Determine which portal vertex to replace with new point
    v4v0 = v.cross(v0)
    dot = v1.dot(v4v0)
    if dot > 0:
        dot = v2.dot(v4v0)
        if dot > 0:
            # Replace v1
            v1 = v; v1_a = va; v1_b = vb
        else:
            # Replace v3
            v3 = v; v3_a = va; v3_b = vb
    else:
        dot = v3.dot(v4v0)
        if dot > 0:
            # Replace v2
            v2 = v; v2_a = va; v2_b = vb
        else:
            # Replace v1
            v1 = v; v1_a = va; v1_b = vb
    return v1, v2, v3, v1_a, v2_a, v3_a, v1_b, v2_b, v3_b

@ti.func
def mpr_refine_portal(v0, v1, v2, v3, v1_a, v2_a, v3_a, v1_b, v2_b, v3_b, gi_a, gi_b):
    ret = -1
    i = 0
    while i < MPR_MAX_ITERATIONS:
        # Step 1: portal face normal
        direction = mpr_portal_dir(v1, v2, v3)

        # Step 2: is origin on the portal face? (converged)
        if mpr_portal_encapsules_origin(v1, direction):
            ret = 0
            break

        # Step 3: new support point beyond portal face
        v4, v4_a, v4_b = compute_support(direction, gi_a, gi_b)

        # Step 4: can't reach origin or converged to tolerance
        if not mpr_portal_can_encapsule_origin(v4, direction):
            ret = -1
            break
        if mpr_portal_reach_tolerance(v1, v2, v3, v4, direction):
            ret = -1
            break

        # Step 5: replace one portal vertex with new point
        v1, v2, v3, v1_a, v2_a, v3_a, v1_b, v2_b, v3_b = mpr_expand_portal(
            v0, v1, v2, v3, v1_a, v2_a, v3_a, v1_b, v2_b, v3_b, v4, v4_a, v4_b
        )
        i += 1

    return ret, v1, v2, v3, v1_a, v2_a, v3_a, v1_b, v2_b, v3_b

@ti.func
def mpr_find_pos(v0, v1, v2, v3, v0_a, v1_a, v2_a, v3_a, v0_b, v1_b, v2_b, v3_b):
    # Barycentric coords of origin projected onto tetrahedron
    # b[i] = volume of sub-tetrahedron opposite vertex i
    b = ti.Vector([0.0, 0.0, 0.0, 0.0])
    verts = ti.Matrix.rows([v0, v1, v2, v3])
    v1s = ti.Matrix.rows([v0_a, v1_a, v2_a, v3_a])
    v2s = ti.Matrix.rows([v0_b, v1_b, v2_b, v3_b])

    for i in range(4):
        i1 = (i % 2) + 1
        i2 = (i + 2) % 4
        i3 = 3 * ((i + 1) % 2)
        sign = 1.0 - 2.0 * float(((i + 1) // 2) % 2)
        vec = ti.Vector([verts[i1, 0], verts[i1, 1], verts[i1, 2]]).cross(
              ti.Vector([verts[i2, 0], verts[i2, 1], verts[i2, 2]]))
        b[i] = vec.dot(ti.Vector([verts[i3, 0], verts[i3, 1], verts[i3, 2]])) * sign

    total = b.sum()

    if total < MPR_EPS:
        # Fallback: project onto portal triangle using portal normal
        direction = mpr_portal_dir(v1, v2, v3)
        b[0] = 0.0
        for i in range(1, 4):
            i1 = i % 3 + 1
            i2 = (i + 1) % 3 + 1
            vec = ti.Vector([verts[i1, 0], verts[i1, 1], verts[i1, 2]]).cross(
                  ti.Vector([verts[i2, 0], verts[i2, 1], verts[i2, 2]]))
            b[i] = vec.dot(direction)
        total = b.sum()

    p1 = ti.Vector([0.0, 0.0, 0.0])
    p2 = ti.Vector([0.0, 0.0, 0.0])
    for i in range(4):
        p1 += b[i] * ti.Vector([v1s[i, 0], v1s[i, 1], v1s[i, 2]])
        p2 += b[i] * ti.Vector([v2s[i, 0], v2s[i, 1], v2s[i, 2]])

    return (0.5 / total) * (p1 + p2)

@ti.func
def mpr_find_penetration(v0, v1, v2, v3, v0_a, v1_a, v2_a, v3_a, v0_b, v1_b, v2_b, v3_b, gi_a, gi_b):
    # Second refinement loop: find exact penetration depth, normal, contact pos
    normal = ti.Vector([0.0, 0.0, 0.0])
    pos = ti.Vector([0.0, 0.0, 0.0])
    penetration = 0.0
    iterations = 0

    while True:
        direction = mpr_portal_dir(v1, v2, v3)
        v4, v4_a, v4_b = compute_support(direction, gi_a, gi_b)

        if mpr_portal_reach_tolerance(v1, v2, v3, v4, direction) or iterations > MPR_MAX_ITERATIONS:
            # Converged: extract contact info
            penetration = direction.dot(v1)
            normal = -direction
            pos = mpr_find_pos(v0, v1, v2, v3, v0_a, v1_a, v2_a, v3_a, v0_b, v1_b, v2_b, v3_b)
            break

        v1, v2, v3, v1_a, v2_a, v3_a, v1_b, v2_b, v3_b = mpr_expand_portal(
            v0, v1, v2, v3, v1_a, v2_a, v3_a, v1_b, v2_b, v3_b, v4, v4_a, v4_b
        )
        iterations += 1

    return normal, pos, penetration

@ti.func
def mpr_find_penetr_segment(v1, v1_a, v1_b):
    penetration = v1.norm()
    normal = -v1.normalized()
    pos = (v1_a + v1_b) * 0.5
    return normal, pos, penetration

@ti.func
def mpr_collide(gi_a: ti.i32, gi_b: ti.i32):
    ret, v0, v1, v2, v3, v0_a, v1_a, v2_a, v3_a, v0_b, v1_b, v2_b, v3_b = mpr_discover_portal(gi_a, gi_b)

    is_col = 0
    normal = ti.Vector([0.0, 0.0, 0.0])
    pos = ti.Vector([0.0, 0.0, 0.0])
    penetration = 0.0

    if ret == 1:
        is_col = 1
        normal, pos, penetration = mpr_find_penetr_touch(v0, v1_a, v1_b)
    elif ret == 2:
        is_col = 1
        normal, pos, penetration = mpr_find_penetr_segment(v1, v1_a, v1_b)
    elif ret == 0:
        ret2, v1, v2, v3, v1_a, v2_a, v3_a, v1_b, v2_b, v3_b = mpr_refine_portal(
            v0, v1, v2, v3, v1_a, v2_a, v3_a, v1_b, v2_b, v3_b, gi_a, gi_b
        )
        if ret2 >= 0:
            is_col = 1
            normal, pos, penetration = mpr_find_penetration(
                v0, v1, v2, v3, v0_a, v1_a, v2_a, v3_a, v0_b, v1_b, v2_b, v3_b, gi_a, gi_b
            )

    return is_col, normal, pos, penetration

@ti.kernel
def narrow_phase(num_pairs: ti.i32):
    data.num_contacts[None] = 0
    for i in range(num_pairs):
        gi_a = data.collision_pairs[i][0]
        gi_b = data.collision_pairs[i][1]
        is_col, normal, pos, penetration = mpr_collide(gi_a, gi_b)
        if is_col == 1:
            idx = ti.atomic_add(data.num_contacts[None], 1)
            if idx < data.contacts.shape[0]:
                data.contacts[idx].pos = pos
                data.contacts[idx].normal = normal
                data.contacts[idx].penetration = penetration
                data.contacts[idx].geom_a = gi_a
                data.contacts[idx].geom_b = gi_b

# --- AABB ---

@ti.func
def compute_aabb(verts: ti.template(), start: ti.i32, end: ti.i32, center: ti.types.vector(3, ti.f32)):
    """Compute local AABB from verts[start..end) relative to center."""
    aabb_min = ti.Vector([1e10, 1e10, 1e10])
    aabb_max = ti.Vector([-1e10, -1e10, -1e10])
    v = start
    while v < end:
        p = verts[v]
        aabb_min = ti.min(aabb_min, p)
        aabb_max = ti.max(aabb_max, p)
        v += 1
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

@ti.func
def aabb_overlap(min_a: ti.types.vector(3, ti.f32), max_a: ti.types.vector(3, ti.f32),
                 min_b: ti.types.vector(3, ti.f32), max_b: ti.types.vector(3, ti.f32)) -> ti.i32:
    """Check if two AABBs overlap. Returns 1 if overlapping, 0 otherwise."""
    overlap = 1
    if max_a[0] < min_b[0] or max_b[0] < min_a[0]:
        overlap = 0
    if max_a[1] < min_b[1] or max_b[1] < min_a[1]:
        overlap = 0
    if max_a[2] < min_b[2] or max_b[2] < min_a[2]:
        overlap = 0
    return overlap

@ti.kernel
def broad_phase(num_geoms: ti.i32):
    """N^2 broad phase: test all geom pairs for AABB overlap."""
    data.num_collision_pairs[None] = 0
    for i in range(num_geoms):
        min_a, max_a = get_world_aabb(i)
        for j in range(i + 1, num_geoms):
            min_b, max_b = get_world_aabb(j)
            if aabb_overlap(min_a, max_a, min_b, max_b):
                idx = ti.atomic_add(data.num_collision_pairs[None], 1)
                if idx < data.collision_pairs.shape[0]:
                    data.collision_pairs[idx] = ti.Vector([i, j])