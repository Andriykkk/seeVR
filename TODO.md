1. update bvh 
This is LBVH (Linear BVH) - a GPU-friendly algorithm. The key difference:

Morton codes - encode 3D position into 1D value preserving spatial locality
Radix sort - parallel-friendly sorting by Morton codes
Parallel tree build - tree structure determined mathematically from sorted Morton codes
Bottom-up AABB - propagate bounds from leaves to root in parallel