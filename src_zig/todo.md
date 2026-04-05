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









forward_dynamics is the full pipeline that runs BEFORE the constraint solver. It does:

func_compute_mass_matrix — builds the 6×6 mass matrix per body (mass + world inertia). Our step 0 (world inertia) is part of this.

func_factor_mass — Cholesky factorize the mass matrix. Needed for M⁻¹ operations. For free rigid bodies with diagonal mass + 3×3 inertia, this is trivial.

func_torque_and_passive_force — compute all forces: gravity, damping, applied forces, Coriolis. For us: just gravity.

func_update_acc — compute unconstrained acceleration qacc_smooth = M⁻¹ · force (what would happen without contacts). This is the "free fall" acceleration.

func_update_force — finalize the force buffer (qf_smooth).

func_bias_force — Coriolis/centrifugal bias. Zero for free bodies.

func_compute_qacc — sets qacc = acc_smooth as starting point.

For free rigid bodies, all of this collapses to:

Mass matrix = diag(m,m,m) + I_world (our step 0)
Force = mass * gravity (our step 2 init)
qacc_smooth = M⁻¹ · force = gravity (unconstrained acceleration)
Our steps 0 + 2 already cover what forward_dynamics does for free bodies. We don't need the full pipeline — no joints, no Coriolis, no parent links.