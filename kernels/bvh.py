import taichi as ti
import numpy as np
from numba import njit
import kernels.data as data

# BVH construction kernels
@ti.kernel
def bvh_init_centroids(n: ti.i32):
    """Initialize triangle centroids and prim indices (parallel)"""
    for i in range(n):
        idx = i * 3
        v0 = data.vertices[data.indices[idx]]
        v1 = data.vertices[data.indices[idx + 1]]
        v2 = data.vertices[data.indices[idx + 2]]
        data.tri_centroids[i] = (v0 + v1 + v2) / 3.0
        data.bvh_prim_indices[i] = i

@njit(cache=True)
def _build_bvh_njit(n, centroids, vertices, indices,
                    node_aabb_min, node_aabb_max, node_left_first, node_tri_count,
                    prim_indices):
    """Numba-compiled BVH build - 10-50x faster than pure Python"""
    # Initialize root
    node_left_first[0] = 0
    node_tri_count[0] = n

    # Stack for iterative build (pre-allocated)
    stack_node = np.zeros(64, dtype=np.int32)
    stack_first = np.zeros(64, dtype=np.int32)
    stack_count = np.zeros(64, dtype=np.int32)
    stack_ptr = 1
    stack_node[0] = 0
    stack_first[0] = 0
    stack_count[0] = n
    nodes_used = 1

    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = stack_node[stack_ptr]
        first = stack_first[stack_ptr]
        count = stack_count[stack_ptr]

        # Calculate bounds
        aabb_min = np.array([1e30, 1e30, 1e30], dtype=np.float32)
        aabb_max = np.array([-1e30, -1e30, -1e30], dtype=np.float32)
        for i in range(first, first + count):
            tri_idx = prim_indices[i]
            idx = tri_idx * 3
            for vi in range(3):
                v_idx = indices[idx + vi]
                for k in range(3):
                    val = vertices[v_idx, k]
                    if val < aabb_min[k]:
                        aabb_min[k] = val
                    if val > aabb_max[k]:
                        aabb_max[k] = val

        node_aabb_min[node_idx] = aabb_min
        node_aabb_max[node_idx] = aabb_max

        # Leaf if <= 10 triangles
        if count <= 10:
            node_left_first[node_idx] = first
            node_tri_count[node_idx] = count
            continue

        # Find split axis (longest extent)
        extent = aabb_max - aabb_min
        axis = 0
        if extent[1] > extent[0]:
            axis = 1
        if extent[2] > extent[axis]:
            axis = 2
        split_pos = aabb_min[axis] + extent[axis] * 0.5

        # Partition
        i = first
        j = first + count - 1
        while i <= j:
            if centroids[prim_indices[i], axis] < split_pos:
                i += 1
            else:
                tmp = prim_indices[i]
                prim_indices[i] = prim_indices[j]
                prim_indices[j] = tmp
                j -= 1

        left_count = i - first
        if left_count == 0 or left_count == count:
            # Degenerate - keep as leaf
            node_left_first[node_idx] = first
            node_tri_count[node_idx] = count
            continue

        # Create children
        left_idx = nodes_used
        right_idx = nodes_used + 1
        nodes_used += 2

        # Mark parent as interior
        node_left_first[node_idx] = left_idx
        node_tri_count[node_idx] = 0

        # Push children to stack
        stack_node[stack_ptr] = right_idx
        stack_first[stack_ptr] = i
        stack_count[stack_ptr] = count - left_count
        stack_ptr += 1

        stack_node[stack_ptr] = left_idx
        stack_first[stack_ptr] = first
        stack_count[stack_ptr] = left_count
        stack_ptr += 1

    return nodes_used


def build_bvh(num_triangles: int, num_vertices: int):
    """Build BVH acceleration structure using shared data fields.

    Args:
        num_triangles: Number of triangles in the scene
        num_vertices: Number of vertices in the scene
    """
    n = num_triangles
    if n == 0:
        return

    # GPU: compute centroids and init prim_indices
    bvh_init_centroids(n)
    ti.sync()  # Ensure GPU is done before CPU reads

    # CPU: build tree (deterministic)
    _build_bvh_cpu(n, num_vertices)


def _build_bvh_cpu(n: int, num_vertices: int):
    """Build BVH on CPU using Numba for speed"""
    # Copy data to numpy
    centroids = data.tri_centroids.to_numpy()[:n]
    vertices = data.vertices.to_numpy()[:num_vertices]
    indices = data.indices.to_numpy()[:n * 3]

    # Allocate output arrays
    max_nodes = n * 2
    node_aabb_min = np.zeros((max_nodes, 3), dtype=np.float32)
    node_aabb_max = np.zeros((max_nodes, 3), dtype=np.float32)
    node_left_first = np.zeros(max_nodes, dtype=np.uint32)
    node_tri_count = np.zeros(max_nodes, dtype=np.uint32)
    prim_indices = np.arange(n, dtype=np.int32)

    # Run njit-compiled build
    nodes_used = _build_bvh_njit(
        n, centroids, vertices, indices,
        node_aabb_min, node_aabb_max, node_left_first, node_tri_count,
        prim_indices
    )

    # Copy results back to Taichi fields
    for i in range(nodes_used):
        data.bvh_nodes[i].aabb_min = node_aabb_min[i].tolist()
        data.bvh_nodes[i].aabb_max = node_aabb_max[i].tolist()
        data.bvh_nodes[i].left_first = int(node_left_first[i])
        data.bvh_nodes[i].tri_count = int(node_tri_count[i])

    for i in range(n):
        data.bvh_prim_indices[i] = int(prim_indices[i])

    data.num_bvh_nodes[None] = nodes_used
