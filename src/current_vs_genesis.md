# Current Solver vs Genesis — Comparison

## Architecture Comparison

| Aspect | Genesis | Current (solver.comp) |
|--------|---------|----------------------|
| Formulation | Gauss principle optimization | Gradient descent approximation |
| Friction | 4 constraints per contact (pyramid) | Only normal (friction built but unused in step 3) |
| Constraint dirs | `d*mu - normal` (combined) | Separate normal, t1, t2 |
| Hessian | Full `H = M + J^T*D*J`, Cholesky factored | No Hessian — uses `M^{-1}` directly |
| Line search | Newton line search (up to 50 iters) | None — fixed step of 1.0 |
| Incremental update | Rank-1 Cholesky update on active set change | Rebuild everything each iteration |
| Warmstart | Compare warmstart vs smooth, pick best | Init qacc to gravity |
| Convergence | Cost + gradient norm tolerance | Fixed 10 iterations |
| Parallelism | Per-environment parallel, solver serial per env | Single GPU thread for entire solve |
| Impedance | Full smooth curve (mid, power params) | Simplified (power=2, mid=0.5 hardcoded) |
| Angular gradient | Proper: I*qacc_ang via Hessian | Approximated as -qfrc_ang (skips I*qacc_ang) |
| Cost function | Explicit, tracked, used for convergence | Not computed |

## What Current Solver Is Missing

### Critical
1. **No Hessian** — gradient descent with M^{-1} converges much slower than Newton with H^{-1}
2. **No line search** — fixed step can overshoot or undershoot, causing instability
3. **No friction in solve loop** — step 3 only uses normal constraints, ignoring the t1/t2 rows built in step 1
4. **No warmstart** — starting from gravity each frame wastes iterations re-converging

### Important
5. **Angular acceleration gradient wrong** — `g_ang = -qfrc_ang` ignores `I*qacc_ang`, only works when angular acc is small
6. **No convergence check** — always runs 10 iterations even if converged or diverged
7. **Friction pyramid vs separate** — Genesis uses 4 combined dirs per contact; current uses 3 separate (1 normal + 2 tangent)

### Nice to Have
8. **Incremental Cholesky** — avoid full Hessian rebuild when only a few constraints change active status
9. **No-slip post-processing** — secondary solver for zero-slip friction
10. **Cost tracking** — useful for debugging and adaptive iteration count

## File Locations

### Genesis solver code
- Main solver: `Genesis/genesis/engine/solvers/rigid/constraint/solver.py`
  - `add_collision_constraints()` — line 498
  - `func_solve_init()` — line 2548
  - `func_solve_body()` — line 2750
  - `func_solve_iter()` — line 2671
  - `func_linesearch_batch()` — line 2048
  - `func_update_constraint_batch()` — line 2237
  - `func_hessian_direct_batch()` — line 1285
  - `func_cholesky_factor_direct_batch()` — line 1467
  - `func_hessian_and_cholesky_factor_incremental_dense_batch()` — line 1632
  - `func_cholesky_solve_batch()` — line 1746
  - `func_update_gradient_batch()` — line 2339
  - `func_terminate_or_update_descent_batch()` — line 2454
  - `func_update_contact_force()` — line 2782
  - `func_update_qacc()` — line 2824
- Impedance: `Genesis/genesis/utils/geom.py:378` — `imp_aref()`
- No-slip: `Genesis/genesis/engine/solvers/rigid/constraint/noslip.py`
- Solver options: `Genesis/genesis/options/solvers.py`

### Current solver code
- Pipeline orchestration: `src/physics.zig`
- Physics shader: `shaders/physics.comp` (transforms, AABB, broad phase, integration)
- Narrow phase: `shaders/narrow_phase.comp` (MPR collision detection)
- Constraint solver: `shaders/solver.comp` (Newton solver)
- Data layout: `src/data.zig` (GPU buffer definitions)
