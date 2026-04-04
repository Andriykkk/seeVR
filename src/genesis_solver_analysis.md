# Genesis Collision Solver — Complete Step-by-Step Analysis

Source code: `Genesis/genesis/engine/solvers/rigid/constraint/solver.py`

## Overview: Gauss Principle Optimization

Genesis (following MuJoCo) formulates contact resolution as an **optimization problem**, not impulse iteration. It minimizes:

```
cost(qacc) = 0.5 * (qacc - qacc_smooth)^T * M * (qacc - qacc_smooth)
           + 0.5 * SUM_c [ D_c * (J_c * qacc - aref_c)^2 ]   for active constraints
```

- `qacc_smooth` = unconstrained acceleration (gravity + external forces)
- `qacc` = actual acceleration being solved for
- First term penalizes deviation from free motion (Gauss's principle)
- Second term penalizes constraint violation

---

<!-- ## Step 0: Collision Detection (before solver)

Broad phase + narrow phase produces contacts, each with:
`pos`, `normal`, `penetration`, `link_a`, `link_b`, `friction`, `sol_params`

Collision code: `Genesis/genesis/engine/solvers/rigid/collider/`
- `narrowphase.py` — narrow phase dispatch
- `gjk.py` — GJK algorithm
- `epa.py` — EPA for penetration depth
- `mpr.py` — Minkowski Portal Refinement

--- -->

## Step 1: Build Constraints

Source: `solver.py:498` — `add_collision_constraints()`

### 4 constraints per contact (friction pyramid)

For each contact, build two tangent vectors:

Source: `Genesis/genesis/utils/geom.py:360` — `ti_orthogonals()`

```
d1, d2 = orthogonals(normal)
```

Then 4 constraint directions:

```
i=0: n = +d1 * friction - normal
i=1: n = -d1 * friction - normal
i=2: n = +d2 * friction - normal
i=3: n = -d2 * friction - normal
```

This is a **linearized friction pyramid** — each edge of the pyramid is one constraint.
The normal force emerges implicitly from the combination of 4 edges.

### 1a. Build Jacobian row

Source: `solver.py:550-580`

Walk the **kinematic chain** from each link to root. For each DOF along the chain:

```
vel = transform_motion(cdof_ang, cdof_vel, contact_pos - link_COM)
jac[constraint, i_d] += sign * dot(vel, n)
```

Where `sign = -1` for link_a, `+1` for link_b.
Also accumulate: `jac_qvel = SUM( jac * vel[i_d] )`

### 1b. Compute impedance and aref

Source: `Genesis/genesis/utils/geom.py:378` — `imp_aref()`

From `sol_params = [timeconst, dampratio, dmin, dmax, width, mid, power]`:

```
imp_x = abs(penetration) / width              # normalized penetration

# Smooth interpolation between dmin and dmax:
if imp_x < mid:
    imp_y = (1/mid^(power-1)) * imp_x^power
else:
    imp_y = 1 - (1/(1-mid)^(power-1)) * (1-imp_x)^power

imp = clamp(dmin + imp_y * (dmax - dmin), dmin, dmax)
if imp_x > 1: imp = dmax

b = 2 / (dmax * timeconst)
k = 1 / (dmax^2 * timeconst^2 * dampratio^2)

aref = -b * jac_qvel - k * imp * (-penetration)
```

- `aref` = **target constraint acceleration**: damping (`-b * vel`) + position correction (`-k * imp * pos`)
- Impedance `imp` smoothly ramps from `dmin` to `dmax` as penetration increases past `width`

### 1c. Compute diagonal regularizer (efc_D)

Source: `solver.py:588-594`

```
diag = (invweight_a + friction^2 * invweight_b)
diag *= 2 * friction^2 * (1 - imp) / imp
diag = max(diag, EPS)
efc_D = 1 / diag
```

`efc_D` = **constraint stiffness** — how strongly each constraint pushes back.
Depends on effective mass (`invweight`) and impedance.

---

## Step 2: Solver Initialization

Source: `solver.py:2548` — `func_solve_init()`

### 2a. Choose starting qacc (warmstart)

Source: `solver.py:2559-2623`

Compare cost of `qacc_prev` (last frame's solution) vs `qacc_smooth` (unconstrained).
Pick whichever has lower cost. Critical for convergence speed.

### 2b. Compute Ma = M * qacc

Source: `solver.py:2523` — `initialize_Ma()`

```
Ma[i_d] = SUM_j( mass_mat[i_d, j] * qacc[j] )
```

### 2c. Compute Jaref (constraint residual)

Source: `solver.py:2500` — `initialize_Jaref()`

```
Jaref[c] = -aref[c] + SUM_d( jac[c, d] * qacc[d] )
```

`Jaref < 0` means constraint is violated (needs force).

### 2d. Determine active set + compute forces + cost

Source: `solver.py:2237` — `func_update_constraint_batch()`

For each **contact constraint**:

```
active[c] = (Jaref[c] < 0)       // violated -> active
efc_force[c] = -Jaref[c] * efc_D[c] * active[c]
```

Compute **qfrc_constraint** (force in DOF space):

```
qfrc_constraint[d] = SUM_c( jac[c, d] * efc_force[c] )
```

Compute **cost**:

```
cost = SUM_d 0.5*(Ma[d] - force[d]) * (qacc[d] - qacc_smooth[d])   // Gauss term
     + SUM_c 0.5 * Jaref[c]^2 * efc_D[c] * active[c]              // constraint term
```

### 2e. Build Hessian H = M + J^T D J

Source: `solver.py:1285` — `func_hessian_direct_batch()`

```
H[d1, d2] = mass_mat[d1, d2] + SUM_c( jac[c,d1] * jac[c,d2] * efc_D[c] * active[c] )
```

Only lower triangle stored (symmetric).

### 2f. Cholesky factor H = L L^T

Source: `solver.py:1467` — `func_cholesky_factor_direct_batch()`

Standard in-place Cholesky on lower triangle of H.
After this, `nt_H` stores L (overwritten).

```
for i_d in range(n_dofs):
    H[i_d, i_d] = sqrt(H[i_d, i_d] - SUM_j<i_d( H[i_d, j]^2 ))
    for j_d > i_d:
        H[j_d, i_d] = (H[j_d, i_d] - SUM_k<i_d( H[j_d,k]*H[i_d,k] )) / H[i_d, i_d]
```

### 2g. Compute gradient and search direction

Source: `solver.py:2339` — `func_update_gradient_batch()`

```
grad[d] = Ma[d] - force[d] - qfrc_constraint[d]
```

Solve `H * Mgrad = grad` via Cholesky forward+back substitution:

Source: `solver.py:1746` — `func_cholesky_solve_batch()`

```
// Forward: L * y = grad
for i_d:
    Mgrad[i_d] = (grad[i_d] - SUM_j<i_d( L[i_d,j] * Mgrad[j] )) / L[i_d,i_d]

// Backward: L^T * Mgrad = y
for i_d (reversed):
    Mgrad[i_d] = (Mgrad[i_d] - SUM_j>i_d( L[j,i_d] * Mgrad[j] )) / L[i_d,i_d]
```

Initial search direction:

```
search = -Mgrad       // Newton descent direction
```

---

## Step 3: Newton Iterations (up to 50)

Source: `solver.py:2750` — `func_solve_body()` calls `func_solve_iter()` in a loop

### 3a. Line search

Source: `solver.py:2048` — `func_linesearch_batch()`

Find step size `alpha` along `search` direction that minimizes cost.

Pre-compute (source: `solver.py:1888` — `func_ls_init()`):

```
mv[d] = M * search[d]           // mass matrix * search
jv[c] = J[c] * search           // Jacobian * search
```

The cost along the line is a **piecewise quadratic** (active set can change):

```
cost(alpha) = gauss_quad + SUM_c [ D_c * 0.5 * (Jaref + alpha*jv)^2 ] for active constraints
```

Source: `solver.py:1944` — `func_ls_point_fn()`

Each evaluation computes cost + first derivative + second derivative:

```
deriv_0 = 2*alpha*quad2 + quad1        // first derivative
deriv_1 = 2*quad2                      // second derivative
```

The line search uses Newton steps on this 1D function:
1. Evaluate at alpha=0
2. Step: `alpha_new = alpha - deriv_0 / deriv_1`
3. If not converged, bracket from both sides
4. Bisect + Newton within bracket
5. Converge when `|deriv_0| < gtol`
6. Up to `ls_iterations` (default 50) evaluations

### 3b. Update qacc, Ma, Jaref

Source: `solver.py:2692-2699`

```
qacc += search * alpha
Ma += mv * alpha
Jaref[c] += jv[c] * alpha
```

### 3c. Update constraints

Source: `solver.py:2706` — calls `func_update_constraint_batch()`

Re-evaluate active set, forces, qfrc, cost (same as step 2d).

### 3d. Update Hessian — incremental Cholesky

Source: `solver.py:1632` — `func_hessian_and_cholesky_factor_incremental_dense_batch()`

For each constraint whose active status **changed** (activated or deactivated):
- Rank-1 update (activate) or downdate (deactivate) of the Cholesky factor L
- `vec = jac[c] * sqrt(efc_D[c])`
- Apply Givens rotations to update L in-place:

```
for each changed constraint c:
    sign = +1 if newly active, -1 if deactivated
    vec = jac[c] * sqrt(efc_D[c])
    for k in range(n_dofs):
        tmp = L[k,k]^2 + sign * vec[k]^2
        r = sqrt(tmp)
        c_rot = r / L[k,k]
        s_rot = vec[k] / L[k,k]
        L[k,k] = r
        for i > k:
            L[i,k] = (L[i,k] + s_rot * vec[i] * sign) / c_rot
            vec[i] = vec[i] * c_rot - s_rot * L[i,k]
```

If the incremental update produces a degenerate matrix (`tmp < EPS`), fall back to full
Hessian rebuild + Cholesky from scratch (`func_hessian_and_cholesky_factor_direct_batch`).

### 3e. Compute gradient + new search direction

Source: `solver.py:2732`

```
grad = Ma - force - qfrc_constraint
solve H * Mgrad = grad via Cholesky
search = -Mgrad
```

### 3f. Check convergence

Source: `solver.py:2454` — `func_terminate_or_update_descent_batch()`

```
improvement = prev_cost - cost
grad_norm = sqrt(SUM( grad[d]^2 ))
tol_scaled = meaninertia * max(1, n_dofs) * tolerance

STOP if: grad_norm <= tol_scaled  OR  improvement <= tol_scaled
```

---

## Step 4: Finalization

Source: `solver.py:2824` — `func_update_qacc()`

```
dofs.acc = qacc                              // final accelerations
dofs.qf_constraint = qfrc_constraint        // constraint forces in DOF space
dofs.force = qf_smooth + qfrc_constraint    // total force
```

Save `qacc` for warmstart next frame (`qacc_ws = qacc`).

---

## Step 5: Extract Contact Forces

Source: `solver.py:2782` — `func_update_contact_force()`

For each contact, sum the 4 pyramid constraint forces back into a 3D force vector:

```
d1, d2 = orthogonals(normal)
force = vec3(0)
for i in range(4):
    d = (2*(i%2)-1) * (d1 if i<2 else d2)
    n = d * friction - normal
    force += n * efc_force[contact*4 + i]
```

---

## Optional: No-Slip Post-Processing

Source: `Genesis/genesis/engine/solvers/rigid/constraint/noslip.py`

If `noslip_iterations > 0`, runs a secondary solver that enforces zero-slip friction
(prevents sliding even when the pyramid approximation would allow it).

---

## Default Solver Parameters

Source: `Genesis/genesis/options/solvers.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| constraint_solver | Newton | Newton or CG |
| iterations | 50 | Max Newton iterations |
| tolerance | 1e-6 | Convergence tolerance |
| ls_iterations | 50 | Max line search evaluations |
| ls_tolerance | 1e-2 | Line search gradient tolerance (relative) |

Default impedance `sol_params` (from MuJoCo):

| Parameter | Default | Description |
|-----------|---------|-------------|
| timeconst | 0.02 | Time constant for damping |
| dampratio | 1.0 | Damping ratio |
| dmin | 0.9 | Minimum impedance |
| dmax | 0.95 | Maximum impedance |
| width | 0.001 | Penetration range for impedance ramp |
| mid | 0.5 | Midpoint of smooth curve |
| power | 2 | Smoothness power |
