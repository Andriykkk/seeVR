3. optimisation
Great question! The issue is ray coherence:

First bounce (primary rays):

All rays start from camera, going similar directions
Neighboring pixels = similar rays
GPU threads in warp do same work
Same BVH nodes accessed = good cache
Fast
Second bounce (scattered rays):

Rays scatter to random directions from random positions
Neighboring pixels = completely different rays
GPU warp divergence (threads take different branches)
Random BVH node access = cache misses
Each thread traverses different tree paths
10-50x slower per ray
This is called "secondary ray incoherence" - a fundamental GPU path tracing problem. Solutions:

Ray sorting - reorder rays by direction before tracing
Wavefront rendering - batch similar rays together
Persistent threads - better GPU utilization
Want me to try adding simple ray sorting or just accept this is a limitation of naive path tracing?


4. test different bvh with different sizes and simulations
    For dynamic scenes with frequent BVH rebuilds, there are several strategies:

    1. BVH Refitting (fastest rebuild)

    Don't rebuild tree structure, just update AABBs bottom-up
    O(n) rebuild, works well if objects don't move far from original positions
    Quality degrades over time → needs periodic full rebuild
    2. LBVH (Linear BVH) - best for frequent full rebuilds

    Uses Morton codes to sort primitives along space-filling curve
    O(n log n) build, fully parallelizable on GPU
    ~10-20% slower traversal than SAH, but 10-100x faster build
    Good balance for dynamic scenes
    3. Two-Level (TLAS/BLAS) - best for rigid body simulation

    Build BVH for each object once (BLAS - static)
    Build lightweight top-level BVH over object bounds (TLAS - rebuilt per frame)
    Only rebuild TLAS when objects move
    This is what RTX hardware uses

5. calculate bvh quality
    cost =
    traversal_cost +
    (SA_left / SA_parent) * N_left +
    (SA_right / SA_parent) * N_right