"""MPR (Minkowski Portal Refinement) collision detection.

MPR is a general-purpose collision detection algorithm for convex shapes.
It works by finding if the origin is inside the Minkowski Difference of two shapes.

Key concepts:
- Minkowski Difference: A - B contains origin iff A and B overlap
- Support function: returns furthest point on shape in given direction
- Portal: triangle on Minkowski Difference surface, refined until collision found or ruled out
"""
import taichi as ti
import kernels.data as data
from kernels.utils import quat_rotate, quat_rotate_inverse


# =============================================================================
# Support functions - return furthest point on shape in given direction
# =============================================================================

@ti.func
def support_box(pos: ti.types.vector(3, ti.f32),
                quat: ti.types.vector(4, ti.f32),
                half_extents: ti.types.vector(3, ti.f32),
                direction: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    """Support function for oriented box.

    Returns the point on the box surface furthest in the given direction.

    Args:
        pos: Box center position (world space)
        quat: Box orientation quaternion (w, x, y, z)
        half_extents: Half-sizes along local x, y, z axes
        direction: Direction to find furthest point (world space, doesn't need to be normalized)

    Returns:
        World-space point on box surface furthest in direction
    """
    # Transform direction to box's local space
    local_dir = quat_rotate_inverse(quat, direction)

    # In local space, support point is simply the corner with matching signs
    # sign(d.x) * half_x, sign(d.y) * half_y, sign(d.z) * half_z
    local_support = ti.Vector([
        half_extents[0] if local_dir[0] >= 0 else -half_extents[0],
        half_extents[1] if local_dir[1] >= 0 else -half_extents[1],
        half_extents[2] if local_dir[2] >= 0 else -half_extents[2]
    ])

    # Transform back to world space
    world_support = pos + quat_rotate(quat, local_support)

    return world_support


@ti.func
def support_sphere(center: ti.types.vector(3, ti.f32),
                   radius: ti.f32,
                   direction: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    """Support function for sphere.

    Returns the point on the sphere surface furthest in the given direction.

    Args:
        center: Sphere center position (world space)
        radius: Sphere radius
        direction: Direction to find furthest point (world space)

    Returns:
        World-space point on sphere surface furthest in direction
    """
    # Normalize direction (handle zero vector)
    dir_len = ti.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
    result = center
    if dir_len > 1e-8:
        result = center + (direction / dir_len) * radius
    return result


@ti.func
def support_hull(pos: ti.types.vector(3, ti.f32),
                 quat: ti.types.vector(4, ti.f32),
                 vert_start: ti.i32,
                 vert_count: ti.i32,
                 direction: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    """Support function for convex hull mesh.

    Returns the vertex on the hull furthest in the given direction.
    For a convex shape, the support point is always a vertex.

    Args:
        pos: Hull center position (world space)
        quat: Hull orientation quaternion (w, x, y, z)
        vert_start: Start index in data.collision_verts
        vert_count: Number of vertices in this hull
        direction: Direction to find furthest point (world space)

    Returns:
        World-space point on hull surface furthest in direction
    """
    # Transform direction to hull's local space
    local_dir = quat_rotate_inverse(quat, direction)

    # Find vertex with maximum dot product with direction
    best_dot = -1e10
    best_local_vertex = ti.Vector([0.0, 0.0, 0.0])

    for i in range(vert_count):
        local_v = data.collision_verts[vert_start + i]
        d = local_v.dot(local_dir)
        if d > best_dot:
            best_dot = d
            best_local_vertex = local_v

    # Transform best vertex to world space
    world_support = pos + quat_rotate(quat, best_local_vertex)

    return world_support


# =============================================================================
# Generic support function - dispatches based on geometry type
# =============================================================================

@ti.func
def support_geom(geom_idx: ti.i32, direction: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    """Generic support function that dispatches based on geometry type.

    Args:
        geom_idx: Index into data.geoms array
        direction: Direction to find furthest point (world space)

    Returns:
        World-space point on geometry surface furthest in direction
    """
    geom_type = data.geoms[geom_idx].geom_type
    pos = data.geoms[geom_idx].world_pos
    quat = data.geoms[geom_idx].world_quat
    geom_data = data.geoms[geom_idx].data

    result = pos  # Default fallback

    if geom_type == data.GEOM_SPHERE:
        radius = geom_data[0]
        result = support_sphere(pos, radius, direction)

    elif geom_type == data.GEOM_BOX:
        half_extents = ti.Vector([geom_data[0], geom_data[1], geom_data[2]])
        result = support_box(pos, quat, half_extents, direction)

    elif geom_type == data.GEOM_MESH:
        vert_start = ti.cast(geom_data[0], ti.i32)
        vert_count = ti.cast(geom_data[1], ti.i32)
        result = support_hull(pos, quat, vert_start, vert_count, direction)

    return result

@ti.func
def support_minkowski(geom_a: ti.i32, geom_b: ti.i32,
                      direction: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    # Get support point on A in direction d
    support_a = support_geom(geom_a, direction)

    # Get support point on B in direction -d (opposite direction!)
    support_b = support_geom(geom_b, -direction)

    # Minkowski difference: A - B
    return support_a - support_b


# =============================================================================
# MPR Algorithm - Minkowski Portal Refinement
# =============================================================================
MPR_MAX_ITERATIONS = 32
MPR_TOLERANCE = 1e-6


@ti.func
def triple_product(a: ti.types.vector(3, ti.f32),
                   b: ti.types.vector(3, ti.f32),
                   c: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    """Compute (a × b) × c = b(a·c) - a(b·c)"""
    return b * a.dot(c) - a * b.dot(c)


@ti.func
def portal_normal(v1: ti.types.vector(3, ti.f32),
                  v2: ti.types.vector(3, ti.f32),
                  v3: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    """Compute outward normal of portal triangle (v1, v2, v3)."""
    edge1 = v2 - v1
    edge2 = v3 - v1
    n = edge1.cross(edge2)
    length = n.norm()
    if length > 1e-10:
        n = n / length
    return n


@ti.func
def mpr_collide(geom_a: ti.i32, geom_b: ti.i32):
    has_collision = 0
    contact_point = ti.Vector([0.0, 0.0, 0.0])
    contact_normal = ti.Vector([0.0, 1.0, 0.0])
    penetration_depth = 0.0
    done = 0

    center_a = data.geoms[geom_a].world_pos
    center_b = data.geoms[geom_b].world_pos

    # v0: interior point of Minkowski Difference
    v0 = center_a - center_b
    if v0.norm() < MPR_TOLERANCE:
        v0 = ti.Vector([1.0, 0.0, 0.0])

    # v1: support toward origin (direction = -v0)
    v1 = support_minkowski(geom_a, geom_b, -v0)
    if v1.dot(-v0) < MPR_TOLERANCE:
        done = 1

    # v2: perpendicular to v0-v1 line
    v2 = ti.Vector([0.0, 0.0, 0.0])
    v3 = ti.Vector([0.0, 0.0, 0.0])

    dir2 = ti.Vector([0.0, 0.0, 0.0])
    if done == 0:
        dir2 = v0.cross(v1)
        if dir2.dot(dir2) < MPR_TOLERANCE:
            if v1.norm() < MPR_TOLERANCE:
                has_collision = 1
                contact_normal = -v0.normalized()
                sa = support_geom(geom_a, contact_normal)
                sb = support_geom(geom_b, -contact_normal)
                contact_point = (sa + sb) * 0.5
            done = 1

    if done == 0:
        dir2 = dir2.normalized()
        v2 = support_minkowski(geom_a, geom_b, dir2)
        if v2.dot(dir2) < MPR_TOLERANCE:
            done = 1

    # v3: perpendicular to v0-v1-v2 plane, toward origin (with retries)
    if done == 0:
        n = (v1 - v0).cross(v2 - v0).normalized()
        if n.dot(v0) > 0:
            v1, v2 = v2, v1
            n = -n

        for _discover in range(15):
            if done == 0:
                v3 = support_minkowski(geom_a, geom_b, n)
                if v3.dot(n) < MPR_TOLERANCE:
                    done = 1
                if done == 0:
                    va = v1.cross(v3)
                    if va.dot(v0) < -MPR_TOLERANCE:
                        v2 = v3
                        n = (v1 - v0).cross(v2 - v0).normalized()
                        if n.dot(v0) > 0:
                            v1, v2 = v2, v1
                            n = -n
                    else:
                        vb = v3.cross(v2)
                        if vb.dot(v0) < -MPR_TOLERANCE:
                            v1 = v3
                            n = (v1 - v0).cross(v2 - v0).normalized()
                            if n.dot(v0) > 0:
                                v1, v2 = v2, v1
                                n = -n
                        else:
                            done = 2  # tetrahedron complete

    # =====================================================================
    # PHASE 3: Refine portal — confirm collision
    # =====================================================================
    if done == 2:
        done = 0
        for _ in range(MPR_MAX_ITERATIONS):
            if done == 0:
                n = portal_normal(v1, v2, v3)

                # portal encapsulates origin? (origin behind portal)
                if v1.dot(n) > -MPR_TOLERANCE:
                    has_collision = 1
                    done = 1
                else:
                    v4 = support_minkowski(geom_a, geom_b, n)

                    # can new point reach origin?
                    if v4.dot(n) < MPR_TOLERANCE:
                        done = 1  # no collision

                    if done == 0:
                        # convergence check
                        dv1 = v1.dot(n)
                        dv2 = v2.dot(n)
                        dv3 = v3.dot(n)
                        dv4 = v4.dot(n)
                        progress = ti.min(dv4 - dv1, ti.min(dv4 - dv2, dv4 - dv3))
                        if progress < MPR_TOLERANCE:
                            done = 1  # no collision

                    if done == 0:
                        # expand portal
                        v4v0 = v4.cross(v0)
                        d1 = v1.dot(v4v0)
                        if d1 > 0:
                            d2 = v2.dot(v4v0)
                            if d2 > 0:
                                v1 = v4
                            else:
                                v3 = v4
                        else:
                            d3 = v3.dot(v4v0)
                            if d3 > 0:
                                v2 = v4
                            else:
                                v1 = v4

    # =====================================================================
    # PHASE 4: Find penetration depth and contact point
    # =====================================================================
    if has_collision == 1:
        done = 0
        for _ in range(MPR_MAX_ITERATIONS):
            if done == 0:
                n = portal_normal(v1, v2, v3)
                v4 = support_minkowski(geom_a, geom_b, n)

                # convergence check
                dv1 = v1.dot(n)
                dv2 = v2.dot(n)
                dv3 = v3.dot(n)
                dv4 = v4.dot(n)
                progress = ti.min(dv4 - dv1, ti.min(dv4 - dv2, dv4 - dv3))
                if progress < MPR_TOLERANCE:
                    penetration_depth = n.dot(v1)
                    contact_normal = -n
                    # contact point from support points along final normal
                    sa = support_geom(geom_a, n)
                    sb = support_geom(geom_b, -n)
                    contact_point = (sa + sb) * 0.5
                    done = 1
                else:
                    # expand portal for better accuracy
                    v4v0 = v4.cross(v0)
                    d1 = v1.dot(v4v0)
                    if d1 > 0:
                        d2 = v2.dot(v4v0)
                        if d2 > 0:
                            v1 = v4
                        else:
                            v3 = v4
                    else:
                        d3 = v3.dot(v4v0)
                        if d3 > 0:
                            v2 = v4
                        else:
                            v1 = v4

    return has_collision, contact_point, contact_normal, penetration_depth
