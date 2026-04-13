The only ways to actually reduce the complexity:

Hill climbing O(√n) — needs adjacency data, random memory access
Gauss map O(log n) — complex preprocessing, poor GPU access patterns
Keep n small — cap hulls at 32-64 verts via convex decomposition
Analytical support for known shapes — spheres, capsules, boxes don't need vertex loops at all
 

Contact via perturbation — run MPR once for 1 contact, then perturb geometry by small rotations around 4 axes and re-run MPR for up to 4 more contacts. No hull vert scanning.

Warm-starting — cache the contact normal per pair from the previous frame. Use it to bias the initial portal direction via guess_geoms_center (shifts center toward AABB intersection midpoint along cached normal). If warm-started MPR fails, retry without. Reduces iterations from ~25 to ~5 for persistent contacts.







