Your MPR is correct and functionally equivalent to Genesis in MuJoCo-compat mode. The two things worth adding:

point_tri_depth for more accurate penetration depth (project origin onto portal triangle instead of just dot(dir, v1))
Center guessing from previous normal for better convergence on deep penetrations





So Genesis's order is:

Forward dynamics (compute accelerations from forces, mass matrix) — BEFORE collision
Collision detection (broad + narrow phase)
Build constraints (Jacobian, aref, efc_D from contacts) — DIRECTLY after collision
Solve constraints (Newton iterations)
Integrate (update positions/velocities)
No intermediate steps between collision and constraints. They flow directly.

Our current pipeline:

Clear counters
Update geom transforms
Compute AABBs
Broad phase
Narrow phase (MPR)
Solver (penalty forces in qacc)
Apply qacc to velocities
Integrate positions
Update render vertices
We're missing the forward dynamics step (mass matrix computation) before collision. But for free rigid bodies, the mass matrix is just diag(m,m,m, I_world) which we compute in step 0 (world inertia). So the main missing piece is exactly steps 3-4: building constraints properly and solving them.

The constraint building (Jacobian + aref + efc_D) is in step_build_jacobian — but it has bugs. Want me to focus on fixing that shader to match Genesis exactly?