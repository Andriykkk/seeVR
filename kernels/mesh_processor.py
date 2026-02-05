"""Mesh collision processor - determines collision strategy based on complexity (GPU)."""
import taichi as ti
import numpy as np
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
# Quickhull Algorithm
# =============================================================================

def _face_normal(points, f):
    """Get outward normal of face."""
    n = np.cross(points[f[1]] - points[f[0]], points[f[2]] - points[f[0]])
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-10 else n

def _point_dist(points, pi, f):
    """Signed distance from point to face plane."""
    return np.dot(points[pi] - points[f[0]], _face_normal(points, f))

def _orient_face(points, f, centroid):
    """Orient face outward (away from centroid)."""
    fc = (points[f[0]] + points[f[1]] + points[f[2]]) / 3.0
    if np.dot(_face_normal(points, f), fc - centroid) < 0:
        return [f[0], f[2], f[1]]
    return f

def quickhull_3d(verts):
    """Compute 3D convex hull. Returns (hull_verts, hull_faces)."""
    pts = np.asarray(verts, dtype=np.float64)
    n = len(pts)
    if n < 4:
        return pts.astype(np.float32), np.array([[0,1,2]], dtype=np.int32)

    # Find initial tetrahedron from extreme points
    ext = [np.argmin(pts[:,i//2]) if i%2==0 else np.argmax(pts[:,i//2]) for i in range(6)]
    p0, p1 = max(((ext[i], ext[j]) for i in range(6) for j in range(i+1,6)),
                  key=lambda p: np.linalg.norm(pts[p[0]] - pts[p[1]]))

    line = pts[p1] - pts[p0]
    line /= np.linalg.norm(line)
    p2 = max((i for i in range(n) if i not in (p0,p1)),
             key=lambda i: np.linalg.norm(pts[i] - pts[p0] - np.dot(pts[i]-pts[p0], line)*line))

    norm = np.cross(pts[p1]-pts[p0], pts[p2]-pts[p0])
    norm /= np.linalg.norm(norm)
    p3 = max((i for i in range(n) if i not in (p0,p1,p2)),
             key=lambda i: abs(np.dot(pts[i]-pts[p0], norm)))

    centroid = (pts[p0] + pts[p1] + pts[p2] + pts[p3]) / 4.0
    on_hull = {p0, p1, p2, p3}

    # Initial faces with outside sets
    faces = [_orient_face(pts, f, centroid) for f in [[p0,p1,p2], [p0,p2,p3], [p0,p3,p1], [p1,p3,p2]]]
    outside = [[i for i in range(n) if i not in on_hull and _point_dist(pts, i, f) > 1e-10] for f in faces]

    # Main loop
    while True:
        # Find furthest outside point
        best = (-1, -1, 0)
        for fi, f in enumerate(faces):
            for pi in outside[fi]:
                d = _point_dist(pts, pi, f)
                if d > best[2]:
                    best = (fi, pi, d)
        if best[0] < 0:
            break

        new_pt = best[1]
        on_hull.add(new_pt)

        # Find visible faces and horizon edges
        visible = [fi for fi, f in enumerate(faces) if _point_dist(pts, new_pt, f) > 1e-10]
        edges = {}
        for fi in visible:
            for i in range(3):
                e = (faces[fi][i], faces[fi][(i+1)%3])
                k = (min(e), max(e))
                edges[k] = edges.get(k, []) + [e]
        horizon = [es[0] for es in edges.values() if len(es) == 1]

        # Collect outside points from visible faces
        all_out = {pi for fi in visible for pi in outside[fi]} - {new_pt}

        # Remove visible, add new faces
        for fi in sorted(visible, reverse=True):
            faces.pop(fi)
            outside.pop(fi)

        for e0, e1 in horizon:
            nf = _orient_face(pts, [e1, e0, new_pt], centroid)
            faces.append(nf)
            outside.append([pi for pi in all_out if pi not in on_hull and _point_dist(pts, pi, nf) > 1e-10])

    # Output
    hull_idx = sorted(on_hull)
    idx_map = {old: new for new, old in enumerate(hull_idx)}
    return (np.asarray(verts)[hull_idx].astype(np.float32),
            np.array([[idx_map[f[0]], idx_map[f[1]], idx_map[f[2]]] for f in faces], dtype=np.int32))


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
