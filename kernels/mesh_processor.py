"""Mesh collision processor - determines collision strategy based on complexity."""
import numpy as np
from scipy.spatial import ConvexHull
import kernels.data as data


def load_collision_mesh(vertices, faces, convexify=True, threshold=0.05):
    """
    Load mesh and choose collision strategy.

    Args:
        vertices: Nx3 numpy array
        faces: Mx3 numpy array (triangle indices)
        convexify: If True, use convex hulls. If False, use SDF.
        threshold: Max volume error for single hull

    Returns:
        dict with collision data
    """
    if convexify:
        # Try single convex hull first
        hull = convex_hull(vertices)

        if volume_error(hull, vertices, faces) < threshold:
            # Single hull is good enough
            return {
                'geom_type': data.GEOM_MESH,
                'mesh_subtype': data.MESH_SINGLE_HULL,
                'hulls': [hull],
                'stats': {
                    'hull_count': 1,
                    'volume_error': volume_error(hull, vertices, faces),
                }
            }
        else:
            # Need decomposition
            # TODO: Implement CoACD here
            hulls = run_coacd_placeholder(vertices, faces, threshold)
            return {
                'geom_type': data.GEOM_MESH,
                'mesh_subtype': data.MESH_DECOMPOSED,
                'hulls': hulls,
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
            'stats': {
                'resolution': sdf_data['resolution'],
            }
        }


def convex_hull(vertices):
    """Build single convex hull from vertices."""
    hull = ConvexHull(vertices)
    return {
        'vertices': vertices[hull.vertices],
        'faces': hull.simplices,
        'volume': hull.volume,
    }


def volume_error(hull, original_verts, original_faces):
    """Compute volume error between hull and original mesh."""
    mesh_volume = compute_mesh_volume(original_verts, original_faces)
    hull_volume = hull['volume']
    return abs(hull_volume - mesh_volume) / max(mesh_volume, 1e-8)


def compute_mesh_volume(vertices, faces):
    """Compute signed volume of triangle mesh."""
    volume = 0.0
    for face in faces:
        v0, v1, v2 = vertices[face]
        volume += v0.dot(np.cross(v1, v2))
    return abs(volume) / 6.0


# ============================================================================
# Placeholder functions - to be implemented
# ============================================================================

def run_coacd_placeholder(vertices, faces, threshold):
    """
    TODO: Implement CoACD convex decomposition.

    For now, just return single hull.
    """
    print("  WARNING: CoACD not implemented yet - using single hull")
    hull = convex_hull(vertices)
    return [hull]


def build_sdf_placeholder(vertices, faces, resolution=64):
    """
    TODO: Implement SDF generation.

    For now, return empty placeholder.
    """
    print("  WARNING: SDF not implemented yet")
    min_corner = np.min(vertices, axis=0)
    max_corner = np.max(vertices, axis=0)

    return {
        'grid': None,  # TODO: Implement
        'min_corner': min_corner,
        'max_corner': max_corner,
        'resolution': resolution,
        'voxel_size': (max_corner - min_corner) / resolution,
    }
