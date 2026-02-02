import taichi as ti
import kernels.data as data

@ti.func
def expand_bits_10(v: ti.u32) -> ti.u32:
    # Mask to 10 bits (max 1023)
    v = v & ti.u32(0x3FF)

    v = (v | (v << 16)) & ti.u32(0x030000FF)

    v = (v | (v << 8)) & ti.u32(0x0300F00F)

    v = (v | (v << 4)) & ti.u32(0x030C30C3)

    v = (v | (v << 2)) & ti.u32(0x09249249)

    return v


@ti.func
def morton_3d(x: ti.f32, y: ti.f32, z: ti.f32) -> ti.u32:
    """Compute 30-bit Morton code from normalized [0,1] coordinates.

    The coordinates are quantized to 10 bits each (0-1023), then interleaved.
    """
    # Clamp to [0, 1] and scale to [0, 1023]
    xi = ti.cast(ti.min(ti.max(x * 1024.0, 0.0), 1023.0), ti.u32)
    yi = ti.cast(ti.min(ti.max(y * 1024.0, 0.0), 1023.0), ti.u32)
    zi = ti.cast(ti.min(ti.max(z * 1024.0, 0.0), 1023.0), ti.u32)

    # Expand each coordinate and interleave: x gets bits 2,5,8,...  y gets 1,4,7,...  z gets 0,3,6,...
    xx = expand_bits_10(xi)
    yy = expand_bits_10(yi)
    zz = expand_bits_10(zi)

    return (xx << 2) | (yy << 1) | zz


@ti.kernel
def compute_scene_bounds(n: ti.i32):
    """Compute scene AABB for normalizing Morton codes (parallel reduction)."""
    for i in range(n):
        centroid = data.tri_centroids[i]
        ti.atomic_min(data.scene_aabb_min[None][0], centroid[0])
        ti.atomic_min(data.scene_aabb_min[None][1], centroid[1])
        ti.atomic_min(data.scene_aabb_min[None][2], centroid[2])
        ti.atomic_max(data.scene_aabb_max[None][0], centroid[0])
        ti.atomic_max(data.scene_aabb_max[None][1], centroid[1])
        ti.atomic_max(data.scene_aabb_max[None][2], centroid[2])


@ti.kernel
def compute_morton_codes(n: ti.i32):
    """Compute Morton code for each triangle centroid (fully parallel)."""
    aabb_min = data.scene_aabb_min[None]
    aabb_max = data.scene_aabb_max[None]
    extent = aabb_max - aabb_min

    # Avoid division by zero
    inv_extent = ti.Vector([0.0, 0.0, 0.0])
    for k in ti.static(range(3)):
        if extent[k] > 1e-6:
            inv_extent[k] = 1.0 / extent[k]

    for i in range(n):
        centroid = data.tri_centroids[i]
        # Normalize to [0, 1]
        normalized = (centroid - aabb_min) * inv_extent
        data.morton_codes[i] = morton_3d(normalized[0], normalized[1], normalized[2])

def compute_all_morton_codes(num_triangles: int):
    """Compute Morton codes for all triangles (GPU parallel).

    Call after bvh_init_centroids() has computed centroids.
    Returns numpy array of morton codes for sorting.
    """
    n = num_triangles
    if n == 0:
        return np.array([], dtype=np.uint32)

    # Initialize bounds to extreme values
    data.scene_aabb_min[None] = [1e30, 1e30, 1e30]
    data.scene_aabb_max[None] = [-1e30, -1e30, -1e30]

    # Compute scene bounds (parallel reduction)
    compute_scene_bounds(n)
    ti.sync()

    # Compute Morton codes (fully parallel)
    compute_morton_codes(n)
    ti.sync()

    return data.morton_codes.to_numpy()[:n]


# ============================================================================
# LBVH (Linear BVH) - Karras 2012 parallel construction
# ============================================================================

@ti.func
def clz(x: ti.u32) -> ti.i32:
    """Count leading zeros in 32-bit integer."""
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
    """Common prefix length between Morton codes at indices i and j.

    Returns -1 if j is out of bounds (used to determine direction).
    """
    result = -1
    if j >= 0 and j < n:
        a = data.morton_codes[i]
        b = data.morton_codes[j]
        # If codes are equal, use index to break tie (append index bits)
        if a == b:
            result = 32 + clz(ti.cast(i, ti.u32) ^ ti.cast(j, ti.u32))
        else:
            result = clz(a ^ b)
    return result


@ti.kernel
def build_lbvh_tree(n: ti.i32):
    """Build BVH tree structure from sorted Morton codes (Karras 2012).

    For n leaves, creates n-1 internal nodes.
    Internal nodes: indices 0 to n-2
    Leaves: referenced by prim_indices (sorted order)

    Layout: internal nodes [0, n-1), leaves use prim_indices
    Node.left_first = left child index (internal) or first prim (leaf)
    Node.tri_count = 0 for internal, >0 for leaf
    """
    # Process each internal node in parallel
    for i in range(n - 1):
        # Determine direction: compare prefix with neighbors
        d_left = delta(i, i - 1, n)
        d_right = delta(i, i + 1, n)
        d = 1 if d_right > d_left else -1

        # Find upper bound for the range
        delta_min = delta(i, i - d, n)

        # Binary search for range length
        l_max = 2
        while delta(i, i + l_max * d, n) > delta_min:
            l_max *= 2

        # Binary search to find exact range end
        l = 0
        t = l_max // 2
        while t >= 1:
            if delta(i, i + (l + t) * d, n) > delta_min:
                l = l + t
            t = t // 2

        j = i + l * d  # Other end of range

        # Find split position using binary search
        delta_node = delta(i, j, n)
        s = 0
        t = l
        # Binary search for split
        divisor = 2
        while divisor <= l * 2:
            t = (l + divisor - 1) // divisor  # Ceiling division
            if t >= 1:
                if delta(i, i + (s + t) * d, n) > delta_node:
                    s = s + t
            divisor *= 2

        split = i + s * d + ti.min(d, 0)

        # Determine children
        left_idx = split
        right_idx = split + 1

        # Determine range bounds
        range_start = ti.min(i, j)
        range_end = ti.max(i, j)

        # Left child: leaf if split == range_start, else internal
        left_is_leaf = (split == range_start)
        # Right child: leaf if split + 1 == range_end, else internal
        right_is_leaf = (split + 1 == range_end)

        # Store internal node
        # For LBVH: internal nodes at [0, n-1), leaf refs at [n-1, 2n-1)
        # We'll use a simpler layout:
        # - Internal node i stores left child index
        # - Leaves are handled by tri_count > 0

        # Internal node at index i
        # Children: if child is leaf, point to leaf node at n-1 + leaf_idx
        #           if child is internal, point to internal node
        if left_is_leaf:
            # Left child is single leaf at position split in sorted order
            data.bvh_nodes[i].left_first = ti.cast(n - 1 + split, ti.u32)
        else:
            data.bvh_nodes[i].left_first = ti.cast(left_idx, ti.u32)

        data.bvh_nodes[i].tri_count = 0  # Internal node

        # Store right child info (we need parent pointers or store both)
        # For now, store in a way compatible with traversal:
        # left_first = left child, right = left_first + 1 for internal nodes
        # This requires restructuring... let's use standard layout instead


@ti.kernel
def build_lbvh_hierarchy(n: ti.i32):
    """Build LBVH hierarchy from sorted Morton codes.

    Uses simplified Karras approach:
    - n leaves (each is one triangle)
    - n-1 internal nodes
    - Node layout: internals [0..n-2], leaves [n-1..2n-2]
    """
    # Create leaf nodes first
    for i in range(n):
        leaf_idx = n - 1 + i
        data.bvh_nodes[leaf_idx].left_first = ti.cast(i, ti.u32)  # Points to sorted prim
        data.bvh_nodes[leaf_idx].tri_count = 1  # Single triangle leaf

    # Build internal nodes in parallel
    for i in range(n - 1):
        # Direction based on prefix comparison
        d_left = delta(i, i - 1, n)
        d_right = delta(i, i + 1, n)
        d = 1 if d_right > d_left else -1

        # Minimum prefix in opposite direction
        delta_min = delta(i, i - d, n)

        # Find range length with exponential search
        l_max = 2
        while delta(i, i + l_max * d, n) > delta_min:
            l_max *= 2

        # Binary search for exact range
        l = 0
        t = l_max // 2
        while t >= 1:
            if delta(i, i + (l + t) * d, n) > delta_min:
                l += t
            t //= 2

        j = i + l * d

        # Determine range bounds (first < last always)
        first = ti.min(i, j)
        last = ti.max(i, j)

        # Find split with binary search (matching CUDA reference)
        # Compare with first_code, search from first toward last
        delta_node = delta(first, last, n)
        split = first
        stride = last - first
        while stride > 1:
            stride = (stride + 1) // 2
            middle = split + stride
            if middle < last:
                if delta(first, middle, n) > delta_node:
                    split = middle

        # Assign children (matching CUDA reference)
        left_child = split
        right_child = split + 1

        # If left subtree contains single primitive, it's a leaf
        if first == split:
            left_child = split + n - 1  # Leaf node index

        # If right subtree contains single primitive, it's a leaf
        if last == split + 1:
            right_child = split + 1 + n - 1  # Leaf node index

        data.bvh_nodes[i].left_first = ti.cast(left_child, ti.u32)
        data.bvh_nodes[i].right_child = ti.cast(right_child, ti.u32)
        data.bvh_nodes[i].tri_count = 0

        # Set parent indices for children (for bottom-up AABB propagation)
        data.bvh_nodes[left_child].parent_idx = ti.cast(i, ti.u32)
        data.bvh_nodes[right_child].parent_idx = ti.cast(i, ti.u32)

    # Root has no parent (use max u32 = -1 in signed representation)
    data.bvh_nodes[0].parent_idx = ti.u32(0xFFFFFFFF)


@ti.kernel
def clear_aabb_flags(n: ti.i32):
    """Clear atomic flags before AABB propagation."""
    for i in range(n - 1):
        data.bvh_aabb_flags[i] = 0


@ti.kernel
def propagate_aabbs_atomic(n: ti.i32):
    """Bottom-up AABB propagation using atomic flags.

    Each leaf thread walks up the tree. At each parent node:
    - First thread to arrive sets flag and exits (other child not ready)
    - Second thread finds flag=1, computes AABB from both children, continues up
    """
    # Start from each leaf
    for i in range(n):
        leaf_idx = n - 1 + i
        parent = data.bvh_nodes[leaf_idx].parent_idx

        while parent != ti.u32(0xFFFFFFFF):
            # Atomic compare-and-swap: if flag==0, set to 1 and return old value
            parent_i32 = ti.cast(parent, ti.i32)
            old = ti.atomic_or(data.bvh_aabb_flags[parent_i32], 1)

            if old == 0:
                # First thread to arrive - other child not ready yet
                break

            # Second thread - both children are ready, compute AABB
            parent_i32 = ti.cast(parent, ti.i32)
            left_idx = ti.cast(data.bvh_nodes[parent_i32].left_first, ti.i32)
            right_idx = ti.cast(data.bvh_nodes[parent_i32].right_child, ti.i32)

            left_min = data.bvh_nodes[left_idx].aabb_min
            left_max = data.bvh_nodes[left_idx].aabb_max
            right_min = data.bvh_nodes[right_idx].aabb_min
            right_max = data.bvh_nodes[right_idx].aabb_max

            data.bvh_nodes[parent_i32].aabb_min = ti.min(left_min, right_min)
            data.bvh_nodes[parent_i32].aabb_max = ti.max(left_max, right_max)

            # Move up to grandparent
            parent = data.bvh_nodes[parent_i32].parent_idx


@ti.kernel
def compute_leaf_aabbs(n: ti.i32):
    """Compute AABBs for leaf nodes."""
    for i in range(n):
        leaf_idx = n - 1 + i
        prim_idx = data.sort_indices[i]  # Original triangle index at sorted position i
        data.bvh_prim_indices[i] = prim_idx  # Store for traversal
        idx = prim_idx * 3

        v0 = data.vertices[data.indices[idx]]
        v1 = data.vertices[data.indices[idx + 1]]
        v2 = data.vertices[data.indices[idx + 2]]

        aabb_min = ti.min(ti.min(v0, v1), v2)
        aabb_max = ti.max(ti.max(v0, v1), v2)

        data.bvh_nodes[leaf_idx].aabb_min = aabb_min
        data.bvh_nodes[leaf_idx].aabb_max = aabb_max


def build_lbvh(num_triangles: int):
    """Build LBVH using Morton codes + radix sort + Karras construction.

    Full GPU-parallel BVH build.
    """
    from kernels.radix_sort import radix_sort_morton

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