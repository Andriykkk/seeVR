1. Graph Coloring + Batched PGS (Bullet GPU, Box2D v3)
Color contacts so no two same-color contacts share a body. Process each color sequentially, all contacts within a color in parallel.


for each iteration:
    for each color (10-15 colors):
        parallel: solve all contacts of this color
Pro: Maintains GS convergence quality, well-proven
Con: 10-15 serial color passes per iteration, need to rebuild coloring each frame
Used by: Bullet GPU, Box2D v3, Avian/Bevy
2. Mass-Split Jacobi (PhysX, Tonge 2012 SIGGRAPH)
Solve ALL contacts in parallel (pure Jacobi), but divide each body's effective mass by its contact count. This prevents overcorrection when multiple contacts push the same body simultaneously.


for each iteration:
    parallel: solve ALL contacts at once (each sees mass / num_contacts)
    apply impulses with real mass
Pro: Maximum parallelism, simple, no graph coloring needed
Con: Slower convergence, needs more iterations to match PGS quality
Used by: PhysX GPU (variant of this)
3. TGS — Temporal Gauss-Seidel (PhysX 5, Box2D v3)
Instead of many PGS iterations per timestep, use many substeps with 1-2 solver iterations each. Between substeps, re-integrate positions. This naturally improves convergence because positions are more accurate.


for each substep:
    solve contacts (1-2 iterations, graph-colored)
    integrate velocities + positions
Pro: Much better convergence and stability, especially for stacking
Con: Need to recompute contacts each substep (or at least update them). You already do 10 substeps — you could just reduce PGS iterations from 30 to 2-4.
Used by: PhysX 5.x, Box2D v3
4. Pure Jacobi with Over-Relaxation
Like #2 but instead of mass splitting, apply a relaxation factor ω > 1 to accelerate convergence. Simple to implement but can be unstable if ω is too large.

Pro: Trivially parallel, one-line change
Con: Finding stable ω is scene-dependent














1. Graph Coloring + Batched PGS (Bullet GPU, Box2D v3)
Color contacts so no two same-color contacts share a body. Process each color sequentially, all contacts within a color in parallel.


for each iteration:
    for each color (10-15 colors):
        parallel: solve all contacts of this color
Pro: Maintains GS convergence quality, well-proven
Con: 10-15 serial color passes per iteration, need to rebuild coloring each frame
Used by: Bullet GPU, Box2D v3, Avian/Bevy
2. Mass-Split Jacobi (PhysX, Tonge 2012 SIGGRAPH)
Solve ALL contacts in parallel (pure Jacobi), but divide each body's effective mass by its contact count. This prevents overcorrection when multiple contacts push the same body simultaneously.


for each iteration:
    parallel: solve ALL contacts at once (each sees mass / num_contacts)
    apply impulses with real mass
Pro: Maximum parallelism, simple, no graph coloring needed
Con: Slower convergence, needs more iterations to match PGS quality
Used by: PhysX GPU (variant of this)
3. TGS — Temporal Gauss-Seidel (PhysX 5, Box2D v3)
Instead of many PGS iterations per timestep, use many substeps with 1-2 solver iterations each. Between substeps, re-integrate positions. This naturally improves convergence because positions are more accurate.


for each substep:
    solve contacts (1-2 iterations, graph-colored)
    integrate velocities + positions
Pro: Much better convergence and stability, especially for stacking
Con: Need to recompute contacts each substep (or at least update them). You already do 10 substeps — you could just reduce PGS iterations from 30 to 2-4.
Used by: PhysX 5.x, Box2D v3
4. Pure Jacobi with Over-Relaxation
Like #2 but instead of mass splitting, apply a relaxation factor ω > 1 to accelerate convergence. Simple to implement but can be unstable if ω is too large.

Pro: Trivially parallel, one-line change
Con: Finding stable ω is scene-dependent
My Recommendation
You already have 10 substeps. The simplest win: reduce PGS iterations from 30 to 4, and switch to Jacobi (all contacts in parallel). With 10 substeps × 4 Jacobi iterations = 40 total passes, convergence should be comparable to 30 sequential PGS iterations. Each Jacobi pass dispatches one thread per contact — fully parallel.

The implementation is straightforward: each thread reads body velocities, computes its impulse independently, writes to per-contact accumulators, then a second pass sums and applies impulses per body. No graph coloring needed.

Want me to implement parallel Jacobi?