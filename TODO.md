1. optimise several elements (Surface Area Heuristic) 
2. update bvh 
This is LBVH (Linear BVH) - a GPU-friendly algorithm. The key difference:

Morton codes - encode 3D position into 1D value preserving spatial locality
Radix sort - parallel-friendly sorting by Morton codes
Parallel tree build - tree structure determined mathematically from sorted Morton codes
Bottom-up AABB - propagate bounds from leaves to root in parallel

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