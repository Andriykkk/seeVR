"""Mesh collision processor - determines collision strategy based on complexity (GPU)."""
import taichi as ti
import numpy as np
from numba import njit
import kernels.data as data


@ti.kernel
def compute_mesh_volume_gpu(vertices: ti.types.ndarray(dtype=ti.f32, ndim=2),
                            faces: ti.types.ndarray(dtype=ti.i32, ndim=2)) -> ti.f32:
    """Compute signed volume of triangle mesh on GPU."""
    volume = 0.0
    for i in range(faces.shape[0]):
        v0 = ti.Vector([vertices[faces[i, 0], 0], vertices[faces[i, 0], 1], vertices[faces[i, 0], 2]])
        v1 = ti.Vector([vertices[faces[i, 1], 0], vertices[faces[i, 1], 1], vertices[faces[i, 1], 2]])
        v2 = ti.Vector([vertices[faces[i, 2], 0], vertices[faces[i, 2], 1], vertices[faces[i, 2], 2]])
        volume += v0.dot(v1.cross(v2))
    return ti.abs(volume) / 6.0


# =============================================================================
# Quickhull Algorithm (Pure Python/NumPy - no Numba for simplicity)
# =============================================================================

def quickhull_3d(input_vertices):
    """
    Compute 3D convex hull using Quickhull algorithm.

    Args:
        input_vertices: Nx3 numpy array of 3D points

    Returns:
        hull_vertices: Mx3 array of hull vertex coordinates (direct copy from input)
        hull_faces: Kx3 array of triangle face indices (into hull_vertices)
    """
    # Keep reference to original for final output
    original_verts = np.asarray(input_vertices)
    # Work with float64 for precision in calculations
    points = original_verts.astype(np.float64)
    n = len(points)

    if n < 4:
        return points.astype(np.float32), np.array([[0, 1, 2]], dtype=np.int32)

    # Step 1: Find extreme points to build initial tetrahedron
    # Find min/max along each axis
    min_x, max_x = np.argmin(points[:, 0]), np.argmax(points[:, 0])
    min_y, max_y = np.argmin(points[:, 1]), np.argmax(points[:, 1])
    min_z, max_z = np.argmin(points[:, 2]), np.argmax(points[:, 2])

    extreme = [min_x, max_x, min_y, max_y, min_z, max_z]

    # Find two most distant extreme points
    max_dist = -1
    p0, p1 = 0, 1
    for i in range(len(extreme)):
        for j in range(i + 1, len(extreme)):
            d = np.linalg.norm(points[extreme[i]] - points[extreme[j]])
            if d > max_dist:
                max_dist = d
                p0, p1 = extreme[i], extreme[j]

    # Find point furthest from line p0-p1
    line_dir = points[p1] - points[p0]
    line_len = np.linalg.norm(line_dir)
    if line_len < 1e-10:
        return points[:1].astype(np.float32), np.array([[0, 0, 0]], dtype=np.int32)
    line_dir /= line_len

    max_dist = -1
    p2 = 0
    for i in range(n):
        if i == p0 or i == p1:
            continue
        v = points[i] - points[p0]
        proj = np.dot(v, line_dir) * line_dir
        dist = np.linalg.norm(v - proj)
        if dist > max_dist:
            max_dist = dist
            p2 = i

    # Find point furthest from plane (p0, p1, p2)
    v1 = points[p1] - points[p0]
    v2 = points[p2] - points[p0]
    normal = np.cross(v1, v2)
    normal_len = np.linalg.norm(normal)
    if normal_len < 1e-10:
        return points[[p0, p1, p2]].astype(np.float32), np.array([[0, 1, 2]], dtype=np.int32)
    normal /= normal_len

    max_dist = -1
    p3 = 0
    for i in range(n):
        if i in [p0, p1, p2]:
            continue
        dist = abs(np.dot(points[i] - points[p0], normal))
        if dist > max_dist:
            max_dist = dist
            p3 = i

    # Step 2: Build initial tetrahedron with 4 faces
    # Each face stores: [v0, v1, v2] indices into points array
    # Faces are oriented with normals pointing outward

    centroid = (points[p0] + points[p1] + points[p2] + points[p3]) / 4.0

    # Initial faces (will be reoriented to point outward)
    initial_faces = [
        [p0, p1, p2],
        [p0, p2, p3],
        [p0, p3, p1],
        [p1, p3, p2]
    ]

    faces = []
    for face in initial_faces:
        v0, v1, v2 = points[face[0]], points[face[1]], points[face[2]]
        normal = np.cross(v1 - v0, v2 - v0)
        normal_len = np.linalg.norm(normal)
        if normal_len > 1e-10:
            normal /= normal_len
        # Check if normal points outward (away from centroid)
        face_center = (v0 + v1 + v2) / 3.0
        if np.dot(normal, face_center - centroid) < 0:
            # Flip winding
            face = [face[0], face[2], face[1]]
        faces.append(face)

    # Track which points are on the hull
    on_hull = set([p0, p1, p2, p3])

    # Step 3: Assign outside points to faces
    def get_face_normal(face):
        v0, v1, v2 = points[face[0]], points[face[1]], points[face[2]]
        normal = np.cross(v1 - v0, v2 - v0)
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-10:
            normal /= norm_len
        return normal, v0

    def point_above_face(pt_idx, face):
        normal, v0 = get_face_normal(face)
        return np.dot(points[pt_idx] - v0, normal) > 1e-10

    # Build outside sets for each face
    outside_sets = []
    for face in faces:
        outside = []
        for i in range(n):
            if i not in on_hull and point_above_face(i, face):
                outside.append(i)
        outside_sets.append(outside)

    # Step 4: Main loop - process faces with outside points
    max_iterations = n * 3
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Find face with furthest outside point
        best_face_idx = -1
        best_point_idx = -1
        best_dist = 1e-10

        for fi, face in enumerate(faces):
            if not outside_sets[fi]:
                continue
            normal, v0 = get_face_normal(face)
            for pi in outside_sets[fi]:
                dist = np.dot(points[pi] - v0, normal)
                if dist > best_dist:
                    best_dist = dist
                    best_face_idx = fi
                    best_point_idx = pi

        if best_face_idx == -1:
            # No more outside points - done!
            break

        new_point = best_point_idx
        on_hull.add(new_point)

        # Find all faces visible from new point
        visible = []
        for fi, face in enumerate(faces):
            if point_above_face(new_point, face):
                visible.append(fi)

        if not visible:
            continue

        # Collect all outside points from visible faces
        all_outside = set()
        for fi in visible:
            all_outside.update(outside_sets[fi])
        all_outside.discard(new_point)

        # Find horizon edges (edges between visible and non-visible faces)
        edge_count = {}
        for fi in visible:
            face = faces[fi]
            for i in range(3):
                e = (face[i], face[(i + 1) % 3])
                # Store edge with canonical ordering for counting
                e_key = (min(e), max(e))
                e_directed = e  # Keep original direction
                if e_key not in edge_count:
                    edge_count[e_key] = []
                edge_count[e_key].append(e_directed)

        # Horizon edges appear only once (not shared between visible faces)
        horizon = []
        for e_key, edges in edge_count.items():
            if len(edges) == 1:
                # This edge is on the horizon - keep original winding
                horizon.append(edges[0])

        # Remove visible faces (in reverse order to preserve indices)
        for fi in sorted(visible, reverse=True):
            faces.pop(fi)
            outside_sets.pop(fi)

        # Create new faces from new point to horizon edges
        new_faces = []
        new_outside_sets = []
        for e0, e1 in horizon:
            # Create face with correct winding (e1, e0, new_point to match horizon edge direction)
            new_face = [e1, e0, new_point]

            # Verify orientation
            v0, v1, v2 = points[new_face[0]], points[new_face[1]], points[new_face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            normal_len = np.linalg.norm(normal)
            if normal_len > 1e-10:
                normal /= normal_len
            face_center = (v0 + v1 + v2) / 3.0
            if np.dot(normal, face_center - centroid) < 0:
                new_face = [new_face[0], new_face[2], new_face[1]]

            # Assign outside points to new face
            new_outside = []
            for pi in all_outside:
                if pi not in on_hull and point_above_face(pi, new_face):
                    new_outside.append(pi)

            new_faces.append(new_face)
            new_outside_sets.append(new_outside)

        faces.extend(new_faces)
        outside_sets.extend(new_outside_sets)

    # Step 5: Convert to output format
    # Map original point indices to hull vertex indices
    hull_point_indices = sorted(on_hull)
    index_map = {old: new for new, old in enumerate(hull_point_indices)}

    # Copy ORIGINAL vertices directly (no conversion, exact same values)
    hull_vertices = original_verts[hull_point_indices].astype(np.float32)
    hull_faces = np.array([[index_map[f[0]], index_map[f[1]], index_map[f[2]]] for f in faces], dtype=np.int32)

    return hull_vertices, hull_faces


def convex_hull(vertices):
    """Compute convex hull using Quickhull algorithm (CPU with Numba JIT)."""
    hull_verts, hull_faces = quickhull_3d(vertices)

    # Compute hull volume on GPU
    hull_volume = compute_mesh_volume_gpu(hull_verts, hull_faces)

    return {
        'vertices': hull_verts,
        'faces': hull_faces,
        'volume': hull_volume
    }


def volume_error(hull, original_verts, original_faces):
    """Compute volume error between hull and original mesh (GPU)."""
    mesh_volume = compute_mesh_volume_gpu(original_verts, original_faces)
    hull_volume = hull['volume']
    return abs(hull_volume - mesh_volume) / max(mesh_volume, 1e-8)


def load_collision_mesh(vertices, faces, convexify=True, threshold=0.05):
    """
    Load mesh and choose collision strategy (GPU accelerated).

    Args:
        vertices: Nx3 numpy array
        faces: Mx3 numpy array (triangle indices)
        convexify: If True, use convex hulls. If False, use SDF.
        threshold: Max volume error for single hull

    Returns:
        dict with collision data including original mesh volume
    """
    # Calculate original mesh volume (needed for proper inertia)
    mesh_volume = compute_mesh_volume_gpu(vertices, faces)

    if convexify:
        # Try single convex hull first (computed on GPU)
        hull = convex_hull(vertices)

        error = volume_error(hull, vertices, faces)

        if error < threshold:
            # Single hull is good enough
            return {
                'geom_type': data.GEOM_MESH,
                'mesh_subtype': data.MESH_SINGLE_HULL,
                'hulls': [hull],
                'mesh_volume': mesh_volume,
                'stats': {
                    'hull_count': 1,
                    'volume_error': error,
                }
            }
        else:
            # Need decomposition
            # TODO: Implement CoACD on GPU
            hulls = run_coacd_placeholder(vertices, faces, threshold)
            return {
                'geom_type': data.GEOM_MESH,
                'mesh_subtype': data.MESH_DECOMPOSED,
                'hulls': hulls,
                'mesh_volume': mesh_volume,
                'stats': {
                    'hull_count': len(hulls),
                    'volume_error': 0.0,  # TODO: Compute from CoACD
                }
            }
    else:
        # Use SDF for exact non-convex collision
        sdf_data = build_sdf_placeholder(vertices, faces)
        return {
            'geom_type': data.GEOM_SDF,
            'sdf': sdf_data,
            'mesh_volume': mesh_volume,
            'stats': {
                'resolution': sdf_data['resolution'],
            }
        }


# ============================================================================
# Placeholder functions - to be implemented on GPU
# ============================================================================

def run_coacd_placeholder(vertices, faces, threshold):
    """
    TODO: Implement CoACD convex decomposition on GPU.

    For now, just return single hull.
    """
    print("  WARNING: CoACD not implemented yet - using single hull")
    hull = convex_hull(vertices)
    return [hull]


@ti.kernel
def build_sdf_grid_gpu(vertices: ti.types.ndarray(dtype=ti.f32, ndim=2),
                       faces: ti.types.ndarray(dtype=ti.i32, ndim=2),
                       sdf_grid: ti.types.ndarray(dtype=ti.f32, ndim=3),
                       min_corner: ti.types.vector(3, ti.f32),
                       voxel_size: ti.types.vector(3, ti.f32)):
    """
    Build SDF grid on GPU.

    TODO: Implement proper SDF computation using distance to mesh.
    For now, placeholder that sets all values to 0.
    """
    for i, j, k in sdf_grid:
        # TODO: Compute actual signed distance to mesh surface
        sdf_grid[i, j, k] = 0.0


def build_sdf_placeholder(vertices, faces, resolution=64):
    """
    TODO: Implement SDF generation on GPU.

    For now, return empty placeholder.
    """
    print("  WARNING: SDF not implemented yet")
    min_corner = np.min(vertices, axis=0)
    max_corner = np.max(vertices, axis=0)

    voxel_size = (max_corner - min_corner) / resolution
    sdf_grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    # Run GPU kernel (placeholder)
    build_sdf_grid_gpu(vertices, faces, sdf_grid,
                       ti.Vector(min_corner), ti.Vector(voxel_size))

    return {
        'grid': sdf_grid,
        'min_corner': min_corner,
        'max_corner': max_corner,
        'resolution': resolution,
        'voxel_size': voxel_size,
    }
