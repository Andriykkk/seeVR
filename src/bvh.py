import taichi as ti
from data import data 
from radix_sort import radix_sort_morton

@ti.func
def expand_bits_10(v: ti.u32) -> ti.u32:
    v = v & ti.u32(0x3FF)
    v = (v | (v << 16)) & ti.u32(0x030000FF)
    v = (v | (v << 8)) & ti.u32(0x0300F00F)
    v = (v | (v << 4)) & ti.u32(0x030C30C3)
    v = (v | (v << 2)) & ti.u32(0x09249249)
    return v


@ti.func
def morton_3d(x: ti.f32, y: ti.f32, z: ti.f32) -> ti.u32:
    xi = ti.cast(ti.min(ti.max(x * 1024.0, 0.0), 1023.0), ti.u32)
    yi = ti.cast(ti.min(ti.max(y * 1024.0, 0.0), 1023.0), ti.u32)
    zi = ti.cast(ti.min(ti.max(z * 1024.0, 0.0), 1023.0), ti.u32)
    xx = expand_bits_10(xi)
    yy = expand_bits_10(yi)
    zz = expand_bits_10(zi)
    return (xx << 2) | (yy << 1) | zz


@ti.func
def clz(x: ti.u32) -> ti.i32:
    n = 0
    if x == 0:
        n = 32
    else:
        if (x & ti.u32(0xFFFF0000)) == 0:
            n += 16
            x <<= 16
        if (x & ti.u32(0xFF000000)) == 0:
            n += 8
            x <<= 8
        if (x & ti.u32(0xF0000000)) == 0:
            n += 4
            x <<= 4
        if (x & ti.u32(0xC0000000)) == 0:
            n += 2
            x <<= 2
        if (x & ti.u32(0x80000000)) == 0:
            n += 1
    return n


@ti.func
def delta(i: ti.i32, j: ti.i32, n: ti.i32) -> ti.i32:
    result = -1
    if j >= 0 and j < n:
        a = data.morton_codes[i]
        b = data.morton_codes[j]
        if a == b:
            result = 32 + clz(ti.cast(i, ti.u32) ^ ti.cast(j, ti.u32))
        else:
            result = clz(a ^ b)
    return result

@ti.kernel
def compute_morton_codes(n: ti.i32):
    aabb_min = data.scene_aabb_min[None]
    aabb_max = data.scene_aabb_max[None]
    extent = aabb_max - aabb_min

    inv_extent = ti.Vector([0.0, 0.0, 0.0])
    for k in ti.static(range(3)):
        if extent[k] > 1e-6:
            inv_extent[k] = 1.0 / extent[k]

    for i in range(n):
        centroid = data.tri_centroids[i]
        normalized = (centroid - aabb_min) * inv_extent
        data.morton_codes[i] = morton_3d(normalized[0], normalized[1], normalized[2])


@ti.kernel
def bvh_init_centroids(n: ti.i32):
    for i in range(n):
        idx = i * 3
        v0 = data.vertices[data.indices[idx]]
        v1 = data.vertices[data.indices[idx + 1]]
        v2 = data.vertices[data.indices[idx + 2]]
        data.tri_centroids[i] = (v0 + v1 + v2) / 3.0
        data.bvh_prim_indices[i] = i


@ti.kernel
def compute_scene_bounds(n: ti.i32):
    aabb_min = data.tri_centroids[0]
    aabb_max = aabb_min
    i = 1
    while i < n:
        c = data.tri_centroids[i]
        aabb_min = ti.min(aabb_min, c)
        aabb_max = ti.max(aabb_max, c)
        i += 1
    data.scene_aabb_min[None] = aabb_min
    data.scene_aabb_max[None] = aabb_max


@ti.kernel
def build_lbvh_hierarchy(n: ti.i32):
    # Create leaf nodes
    for i in range(n):
        leaf_idx = n - 1 + i
        data.bvh_nodes[leaf_idx].left_first = ti.cast(i, ti.u32)
        data.bvh_nodes[leaf_idx].tri_count = 1

    # Build internal nodes in parallel
    for i in range(n - 1):
        d_left = delta(i, i - 1, n)
        d_right = delta(i, i + 1, n)
        d = 1 if d_right > d_left else -1

        delta_min = delta(i, i - d, n)

        l_max = 2
        while delta(i, i + l_max * d, n) > delta_min:
            l_max *= 2

        l = 0
        t = l_max // 2
        while t >= 1:
            if delta(i, i + (l + t) * d, n) > delta_min:
                l += t
            t //= 2

        j = i + l * d

        first = ti.min(i, j)
        last = ti.max(i, j)

        delta_node = delta(first, last, n)
        split = first
        stride = last - first
        while stride > 1:
            stride = (stride + 1) // 2
            middle = split + stride
            if middle < last:
                if delta(first, middle, n) > delta_node:
                    split = middle

        left_child = split
        right_child = split + 1

        if first == split:
            left_child = split + n - 1

        if last == split + 1:
            right_child = split + 1 + n - 1

        data.bvh_nodes[i].left_first = ti.cast(left_child, ti.u32)
        data.bvh_nodes[i].right_child = ti.cast(right_child, ti.u32)
        data.bvh_nodes[i].tri_count = 0

        data.bvh_nodes[left_child].parent_idx = ti.cast(i, ti.u32)
        data.bvh_nodes[right_child].parent_idx = ti.cast(i, ti.u32)

    data.bvh_nodes[0].parent_idx = ti.u32(0xFFFFFFFF)


@ti.kernel
def compute_leaf_aabbs(n: ti.i32):
    for i in range(n):
        leaf_idx = n - 1 + i
        prim_idx = data.sort_indices[i]
        data.bvh_prim_indices[i] = prim_idx
        idx = prim_idx * 3

        v0 = data.vertices[data.indices[idx]]
        v1 = data.vertices[data.indices[idx + 1]]
        v2 = data.vertices[data.indices[idx + 2]]

        aabb_min = ti.min(ti.min(v0, v1), v2)
        aabb_max = ti.max(ti.max(v0, v1), v2)

        data.bvh_nodes[leaf_idx].aabb_min = aabb_min
        data.bvh_nodes[leaf_idx].aabb_max = aabb_max


@ti.kernel
def clear_aabb_flags(n: ti.i32):
    for i in range(n - 1):
        data.bvh_aabb_flags[i] = 0


@ti.kernel
def propagate_aabbs_atomic(n: ti.i32):
    for i in range(n):
        leaf_idx = n - 1 + i
        parent = data.bvh_nodes[leaf_idx].parent_idx

        while parent != ti.u32(0xFFFFFFFF):
            parent_i32 = ti.cast(parent, ti.i32)
            old = ti.atomic_or(data.bvh_aabb_flags[parent_i32], 1)

            if old == 0:
                break

            parent_i32 = ti.cast(parent, ti.i32)
            left_idx = ti.cast(data.bvh_nodes[parent_i32].left_first, ti.i32)
            right_idx = ti.cast(data.bvh_nodes[parent_i32].right_child, ti.i32)

            left_min = data.bvh_nodes[left_idx].aabb_min
            left_max = data.bvh_nodes[left_idx].aabb_max
            right_min = data.bvh_nodes[right_idx].aabb_min
            right_max = data.bvh_nodes[right_idx].aabb_max

            data.bvh_nodes[parent_i32].aabb_min = ti.min(left_min, right_min)
            data.bvh_nodes[parent_i32].aabb_max = ti.max(left_max, right_max)

            parent = data.bvh_nodes[parent_i32].parent_idx


def build_lbvh(num_triangles: int):
    """Build LBVH using Morton codes + radix sort + Karras construction.

    Full GPU-parallel BVH build.
    """

    n = num_triangles
    if n == 0:
        return

    # 1. Compute centroids (parallel)
    bvh_init_centroids(n)

    # 2. Compute scene bounds and Morton codes
    data.scene_aabb_min[None] = [1e30, 1e30, 1e30]
    data.scene_aabb_max[None] = [-1e30, -1e30, -1e30]
    compute_scene_bounds(n)
    ti.sync()  # Bounds must complete before morton codes
    compute_morton_codes(n)

    # 3. Radix sort Morton codes
    radix_sort_morton(n)

    # 4. Build tree hierarchy (parallel - each internal node independent)
    build_lbvh_hierarchy(n)

    # 5. Compute leaf AABBs + copy prim indices
    compute_leaf_aabbs(n)

    # 6. Compute internal AABBs bottom-up using atomic propagation
    clear_aabb_flags(n)
    propagate_aabbs_atomic(n)
    ti.sync()  # Final sync to ensure BVH is ready

    data.num_bvh_nodes[None] = 2 * n - 1
