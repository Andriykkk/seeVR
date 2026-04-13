The only ways to actually reduce the complexity:

Hill climbing O(√n) — needs adjacency data, random memory access
Gauss map O(log n) — complex preprocessing, poor GPU access patterns
Keep n small — cap hulls at 32-64 verts via convex decomposition
Analytical support for known shapes — spheres, capsules, boxes don't need vertex loops at all




Precomputed support field — 180x180 spherical grid per geom. At runtime, convert direction to (theta, phi), look up 4 neighbors, pick best. O(1) for any mesh.

6. Smaller 2D grid (32×32)
Memory: 1024 × 16 bytes = 16 KB per geom
Lookups: 4 reads + maybe check a few nearby cells = ~8 reads
Speed: O(1), ~8 dot products
Problem: Angular resolution ~5.6°, might miss for sharp hulls. Fine for physics.
7. Triple 1D lookup (3 perpendicular great circles)
3 axes × 64 bins = 192 entries per geom
Memory: 192 × 4 bytes = 768 bytes per geom
Lookups: Query all 3, pick the one whose equator is closest to the query direction. ~3 atan2 + 3 reads + ~6 neighbor checks = ~15 reads
Speed: O(1), ~15 dot products
Problem: Still has blind spots at ~35° from all three equators (the (1,1,1) direction). But the error at those spots is bounded by the bin resolution.


Contact via perturbation — run MPR once for 1 contact, then perturb geometry by small rotations around 4 axes and re-run MPR for up to 4 more contacts. No hull vert scanning.

Warm-starting — cache the contact normal per pair from the previous frame. Use it to bias the initial portal direction via guess_geoms_center (shifts center toward AABB intersection midpoint along cached normal). If warm-started MPR fails, retry without. Reduces iterations from ~25 to ~5 for persistent contacts.