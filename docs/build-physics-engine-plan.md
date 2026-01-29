# Building a GPU Physics Engine: Complete Plan

A roadmap for building a Genesis-like GPU-accelerated physics simulation engine.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Technology Stack Choices](#2-technology-stack-choices)
3. [Phase 1: Core Infrastructure](#3-phase-1-core-infrastructure)
4. [Phase 2: Rigid Body Dynamics](#4-phase-2-rigid-body-dynamics)
5. [Phase 3: Collision Detection](#5-phase-3-collision-detection)
6. [Phase 4: Constraint Solving](#6-phase-4-constraint-solving)
7. [Phase 5: Rendering](#7-phase-5-rendering)
8. [Phase 6: Advanced Physics](#8-phase-6-advanced-physics)
9. [Phase 7: ML Integration](#9-phase-7-ml-integration)
10. [Algorithm Reference](#10-algorithm-reference)
11. [Resources & Papers](#11-resources--papers)

---

## 1. Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                         Scene API                           │
├─────────────────────────────────────────────────────────────┤
│                        Simulator                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │  Rigid  │ │   Soft  │ │  Fluid  │ │  Cloth  │  Solvers  │
│  │  Body   │ │  Body   │ │  (SPH)  │ │  (PBD)  │           │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘           │
│       └──────────┬┴──────────┬┴──────────┘                 │
│                  │  Coupler  │                              │
├──────────────────┴───────────┴──────────────────────────────┤
│                    GPU Compute Backend                      │
├─────────────────────────────────────────────────────────────┤
│              Renderer (Rasterizer / Ray Tracer)             │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **GPU-first** - All physics runs on GPU, not CPU
2. **Data-oriented** - Structure of Arrays (SoA), not Array of Structures (AoS)
3. **Batched** - Support N parallel environments from day one
4. **Modular** - Solvers are independent, coupled via interfaces

---

## 2. Technology Stack Choices

### Option A: Taichi (Recommended for Starting)

```python
# Pros: Easy Python syntax, compiles to CUDA/Vulkan/Metal
# Cons: Less control, dependency on Taichi project

import taichi as ti
ti.init(arch=ti.gpu)

@ti.kernel
def update_physics():
    for i in positions:
        velocities[i] += gravity * dt
        positions[i] += velocities[i] * dt
```

### Option B: CUDA C++ (Maximum Performance)

```cpp
// Pros: Maximum control and performance
// Cons: Harder to write, NVIDIA-only (without extra work)

__global__ void update_physics(float3* pos, float3* vel, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    vel[i] += make_float3(0, -9.81f, 0) * dt;
    pos[i] += vel[i] * dt;
}
```

### Option C: Vulkan Compute (Cross-Platform)

```glsl
// Pros: Works everywhere (AMD, Intel, NVIDIA, mobile)
// Cons: Verbose, harder debugging

layout(local_size_x = 256) in;
void main() {
    uint i = gl_GlobalInvocationID.x;
    velocities[i] += gravity * dt;
    positions[i] += velocities[i] * dt;
}
```

### Option D: WebGPU/WGSL (Browser + Native)

```wgsl
// Pros: Runs in browser and native, modern API
// Cons: Newer ecosystem, some limitations

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    velocities[i] += gravity * dt;
    positions[i] += velocities[i] * dt;
}
```

### Recommendation

| Goal | Choice |
|------|--------|
| Learn & prototype fast | Taichi (Python) |
| Production robotics engine | CUDA C++ |
| Cross-platform game engine | Vulkan Compute |
| Web-based simulation | WebGPU |

---

## 3. Phase 1: Core Infrastructure

### 3.1 GPU Memory Management

```
Week 1-2: Build foundation

Tasks:
├── GPU buffer allocation/deallocation
├── Structure of Arrays (SoA) data layout
├── Batch dimension support (n_envs)
└── Zero-copy interop with ML framework
```

**Data Layout (Critical for Performance):**

```python
# BAD: Array of Structures (AoS) - poor cache/memory access
class Particle:
    position: vec3
    velocity: vec3
    mass: float
particles = [Particle() for _ in range(N)]

# GOOD: Structure of Arrays (SoA) - coalesced memory access
positions = ti.Vector.field(3, float, shape=(N, n_envs))
velocities = ti.Vector.field(3, float, shape=(N, n_envs))
masses = ti.field(float, shape=(N, n_envs))
```

### 3.2 Time Stepping Framework

```python
class Simulator:
    def step(self, dt):
        for _ in range(self.substeps):
            substep_dt = dt / self.substeps
            self.pre_step()
            for solver in self.solvers:
                solver.substep(substep_dt)
            self.couple_solvers()
            self.post_step()
```

### 3.3 Scene Graph

```
Tasks:
├── Entity management (add/remove bodies)
├── Hierarchical transforms (parent-child)
├── Material system
└── Geometry representations (mesh, primitive, SDF)
```

---

## 4. Phase 2: Rigid Body Dynamics

### 4.1 Single Rigid Body

**Algorithm: Semi-Implicit Euler Integration**

```python
# Simplest stable integrator
@ti.kernel
def integrate(dt: float):
    for i in range(n_bodies):
        # Update velocity from forces
        acceleration = forces[i] / masses[i]
        velocities[i] += acceleration * dt
        angular_vel[i] += torques[i] @ inv_inertia[i] * dt

        # Update position from velocity (semi-implicit: use NEW velocity)
        positions[i] += velocities[i] * dt
        orientations[i] = integrate_quaternion(orientations[i], angular_vel[i], dt)
```

**Key Concepts:**
- Linear momentum: `p = m * v`
- Angular momentum: `L = I * ω`
- Quaternion for rotation (avoid gimbal lock)

### 4.2 Articulated Bodies (Robots)

**Algorithm: Articulated Body Algorithm (ABA) / Featherstone**

```
For robot arms, legs, etc. with joints:

Forward Kinematics: joint angles → link positions
Inverse Dynamics: accelerations → required torques
Forward Dynamics: torques → accelerations (what we need for simulation)
```

**Implementation Steps:**

1. **Represent robot as tree of links**
   ```
   Link 0 (base)
     └── Link 1 (shoulder)
           └── Link 2 (elbow)
                 └── Link 3 (wrist)
   ```

2. **Forward pass** - Compute velocities and forces down the tree
3. **Backward pass** - Compute articulated inertias up the tree
4. **Forward pass** - Compute accelerations down the tree

**Key Paper:** Roy Featherstone - "Rigid Body Dynamics Algorithms" (2008)

### 4.3 Integration Methods Comparison

| Method | Stability | Accuracy | Speed | Use Case |
|--------|-----------|----------|-------|----------|
| Explicit Euler | Poor | O(dt) | Fast | Never use |
| Semi-Implicit Euler | Good | O(dt) | Fast | Real-time games |
| Velocity Verlet | Good | O(dt²) | Fast | Particles, molecules |
| RK4 | Good | O(dt⁴) | Slow | Accurate trajectories |
| Implicit Euler | Excellent | O(dt) | Slow | Stiff systems |

**Recommendation:** Start with Semi-Implicit Euler, add Implicit for stiff constraints later.

---

## 5. Phase 3: Collision Detection

### 5.1 Pipeline Overview

```
Collision Detection Pipeline:

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Broad Phase │ --> │ Narrow Phase │ --> │   Contact    │
│   O(n log n) │     │   O(k)       │     │  Generation  │
│  AABB, Grid  │     │  GJK, SAT    │     │  Points,     │
│              │     │              │     │  Normals     │
└──────────────┘     └──────────────┘     └──────────────┘
      │                    │                    │
      ▼                    ▼                    ▼
 Candidate pairs      Actual contacts     Constraint data
 (maybe colliding)    (yes/no + depth)    (for solver)
```

### 5.2 Broad Phase Algorithms

**Option A: Spatial Hash Grid (Simple, Fast)**

```python
@ti.kernel
def build_spatial_hash():
    for i in range(n_bodies):
        cell = ti.floor(positions[i] / cell_size).cast(int)
        hash_val = hash_cell(cell)
        # Insert into hash table
        idx = ti.atomic_add(cell_counts[hash_val], 1)
        cell_bodies[hash_val, idx] = i

@ti.kernel
def find_candidates():
    for i in range(n_bodies):
        cell = get_cell(positions[i])
        # Check 27 neighboring cells (3x3x3)
        for dx, dy, dz in ti.static(ti.ndrange((-1,2), (-1,2), (-1,2))):
            neighbor = cell + ti.Vector([dx, dy, dz])
            for j in get_bodies_in_cell(neighbor):
                if i < j:  # Avoid duplicate pairs
                    add_candidate_pair(i, j)
```

**Option B: Sweep and Prune (Better for Coherent Motion)**

```python
# Sort bodies along one axis (e.g., X)
# Bodies only collide if their intervals overlap

# 1. Sort AABBs by min_x
sorted_bodies = sort_by_min_x(bodies)

# 2. Sweep through, track active intervals
active = []
for body in sorted_bodies:
    # Remove bodies whose max_x < current min_x
    active = [b for b in active if b.max_x >= body.min_x]

    # All remaining active bodies are candidates
    for other in active:
        if overlaps_yz(body, other):
            add_candidate_pair(body, other)

    active.append(body)
```

**Option C: BVH (Bounding Volume Hierarchy)**

Best for complex static geometry. Build tree once, query fast.

### 5.3 Narrow Phase Algorithms

**Algorithm A: GJK (Gilbert-Johnson-Keerthi) - Convex Shapes**

```
GJK answers: "Do two convex shapes overlap?"

Key insight: Two shapes overlap iff their Minkowski Difference contains the origin.

Algorithm:
1. Pick initial direction d
2. Get support point: s = support_A(d) - support_B(-d)
3. Build simplex (point → line → triangle → tetrahedron)
4. Check if simplex contains origin
5. If not, find new direction toward origin
6. Repeat until contains origin (collision) or can prove it won't (no collision)
```

```python
def gjk(shape_a, shape_b):
    d = initial_direction()
    simplex = [support(shape_a, shape_b, d)]
    d = -simplex[0]

    while True:
        point = support(shape_a, shape_b, d)
        if dot(point, d) < 0:
            return False  # No collision

        simplex.append(point)

        if contains_origin(simplex):
            return True  # Collision!

        d = next_direction(simplex)  # Also updates simplex
```

**Algorithm B: EPA (Expanding Polytope Algorithm) - Penetration Depth**

After GJK confirms collision, EPA finds penetration depth and normal.

```
EPA:
1. Start with GJK's final simplex (tetrahedron containing origin)
2. Find face closest to origin
3. Expand polytope in that direction
4. Repeat until closest face is on boundary
5. Return: penetration_depth, contact_normal
```

**Algorithm C: SAT (Separating Axis Theorem) - Boxes, Convex Polygons**

```
Two convex shapes DON'T collide iff there exists a separating axis.

For boxes: test 15 axes (3 face normals each + 9 edge cross products)
For polygons: test all face normals from both shapes
```

### 5.4 Contact Generation

```python
@ti.dataclass
class Contact:
    body_a: int
    body_b: int
    point: vec3        # Contact point in world space
    normal: vec3       # Points from A to B
    penetration: float # Overlap depth
    friction: float

# Generate 1-4 contact points per colliding pair
# For face-face contact: 4 points (box corner)
# For edge-edge: 1-2 points
# For vertex-face: 1 point
```

---

## 6. Phase 4: Constraint Solving

### 6.1 What Are Constraints?

```
Constraints enforce rules:
- Non-penetration: bodies can't overlap
- Joints: bodies stay connected
- Friction: resists sliding
- Motors: enforce velocity/position targets
```

### 6.2 Constraint Formulation

**Velocity-Level Constraint (Impulse-Based):**

```
J * v = 0   (constraint satisfied)
J * v = b   (with bias for position correction)

Where:
- J = Jacobian matrix (how constraint depends on velocities)
- v = velocity vector [v1, ω1, v2, ω2, ...]
- b = bias term (for Baumgarte stabilization)
```

**Example - Contact Constraint:**

```python
# Non-penetration: relative velocity along normal >= 0
# J = [n, r1×n, -n, -r2×n]  (for bodies 1 and 2)
#
# n = contact normal
# r1, r2 = vectors from body centers to contact point
```

### 6.3 Solving Methods

**Option A: Sequential Impulses (PGS) - Simple, Robust**

```python
def solve_constraints_pgs(contacts, iterations=10):
    for _ in range(iterations):
        for c in contacts:
            # Compute relative velocity at contact
            v_rel = get_relative_velocity(c)

            # Compute impulse magnitude
            lambda_ = -(dot(v_rel, c.normal) + c.bias) / c.effective_mass

            # Clamp (non-penetration: only push apart)
            old_lambda = c.accumulated_lambda
            c.accumulated_lambda = max(0, old_lambda + lambda_)
            lambda_ = c.accumulated_lambda - old_lambda

            # Apply impulse
            apply_impulse(c.body_a, +lambda_ * c.normal, c.point)
            apply_impulse(c.body_b, -lambda_ * c.normal, c.point)
```

**Pros:** Simple, handles inequalities naturally, GPU-friendly
**Cons:** Slow convergence, order-dependent

**Option B: Direct Solver (LCP) - Accurate**

```
Formulate as Linear Complementarity Problem:
M * λ + q = 0
λ ≥ 0, (M * λ + q) ≥ 0, λᵀ(M * λ + q) = 0

Solve with:
- Lemke's algorithm
- PATH solver
- Projected Gauss-Seidel
```

**Pros:** More accurate
**Cons:** Harder to implement, harder to parallelize

**Option C: XPBD (Extended Position Based Dynamics) - Stable**

```python
def solve_xpbd(constraints, dt, iterations=10):
    for _ in range(iterations):
        for c in constraints:
            # Compute constraint violation
            C = evaluate_constraint(c)

            # Compute correction
            grad_C = constraint_gradient(c)
            alpha_tilde = c.compliance / (dt * dt)
            delta_lambda = (-C - alpha_tilde * c.lambda_) / (
                inverse_mass_sum(c) + alpha_tilde
            )

            # Apply correction
            c.lambda_ += delta_lambda
            apply_position_correction(c, delta_lambda * grad_C)
```

**Pros:** Unconditionally stable, easy to add new constraints
**Cons:** Not physically accurate (compliance instead of stiffness)

### 6.4 Contact Islands

```
Optimization: Group interacting bodies into "islands"
Bodies in different islands can be solved in parallel

Algorithm:
1. Build contact graph (bodies = nodes, contacts = edges)
2. Find connected components (union-find)
3. Solve each island independently on separate GPU threads
```

### 6.5 Friction

**Coulomb Friction Model:**

```
|f_tangent| ≤ μ * |f_normal|

Where:
- f_tangent = friction force (resists sliding)
- f_normal = normal force (contact force)
- μ = friction coefficient
```

```python
def apply_friction(contact, normal_impulse):
    # Compute tangent velocity
    v_rel = get_relative_velocity(contact)
    v_tangent = v_rel - dot(v_rel, contact.normal) * contact.normal

    if length(v_tangent) < epsilon:
        return  # No sliding

    # Friction direction opposes sliding
    tangent = normalize(v_tangent)

    # Compute friction impulse (clamped by Coulomb limit)
    max_friction = contact.friction * abs(normal_impulse)
    friction_impulse = min(length(v_tangent) * effective_mass, max_friction)

    apply_impulse(contact.body_a, -friction_impulse * tangent, contact.point)
    apply_impulse(contact.body_b, +friction_impulse * tangent, contact.point)
```

---

## 7. Phase 5: Rendering

### 7.1 Rasterization Pipeline

```
Vertex Data → Vertex Shader → Primitive Assembly → Rasterization
           → Fragment Shader → Depth Test → Framebuffer

Implement with: OpenGL, Vulkan, WebGPU, or Metal
```

**Minimum Viable Renderer:**

```python
# 1. Transform vertices to clip space
gl_Position = projection * view * model * vertex_position

# 2. Interpolate attributes across triangle
# 3. Compute lighting in fragment shader
color = ambient + diffuse * max(0, dot(normal, light_dir)) + specular

# 4. Output to framebuffer
```

### 7.2 Shadow Mapping

```
Two-pass rendering:

Pass 1 (Light's view):
- Render scene from light's perspective
- Store depth in shadow map texture

Pass 2 (Camera's view):
- For each pixel, transform to light space
- Compare depth with shadow map
- If pixel depth > shadow map depth: in shadow
```

### 7.3 Ray Tracing (Optional)

**Path Tracing Algorithm:**

```python
def trace_path(ray, depth=0):
    if depth > max_depth:
        return background_color

    hit = intersect_scene(ray)
    if not hit:
        return background_color

    # Direct lighting
    direct = sample_lights(hit)

    # Indirect lighting (recursive)
    new_direction = sample_hemisphere(hit.normal)
    new_ray = Ray(hit.point, new_direction)
    indirect = trace_path(new_ray, depth + 1)

    return hit.emission + hit.brdf * (direct + indirect)
```

**For GPU:** Use compute shaders, BVH for acceleration.

### 7.4 Batch Rendering

```python
# Render all environments in parallel

# Instead of:
for env in range(n_envs):
    render(env)  # Sequential, slow

# Do:
# Single draw call with instancing
glDrawArraysInstanced(GL_TRIANGLES, 0, n_vertices, n_envs)

# Or compute shader:
@compute_shader
def render_all_envs(env_id: int, pixel_id: int):
    # Each thread handles one pixel in one environment
    ...
```

---

## 8. Phase 6: Advanced Physics

### 8.1 Soft Bodies - FEM (Finite Element Method)

```
Discretize continuous material into elements (tetrahedra).

For each element:
1. Compute deformation gradient F
2. Compute strain ε = 0.5 * (FᵀF - I)
3. Compute stress σ using material model (Neo-Hookean, etc.)
4. Compute forces from stress

Material Models:
- Linear elastic: σ = λ*tr(ε)*I + 2μ*ε
- Neo-Hookean: handles large deformation
- Corotational: rotation-invariant linear
```

### 8.2 Cloth - Position Based Dynamics (PBD)

```python
def simulate_cloth_pbd(dt):
    # 1. Predict positions
    for i in particles:
        velocities[i] += gravity * dt
        predicted[i] = positions[i] + velocities[i] * dt

    # 2. Solve constraints (stretch, bend)
    for _ in range(iterations):
        for constraint in constraints:
            # Distance constraint: |p1 - p2| = rest_length
            delta = predicted[c.p2] - predicted[c.p1]
            distance = length(delta)
            correction = (distance - c.rest_length) / distance * delta

            predicted[c.p1] += 0.5 * correction * c.p1_inv_mass
            predicted[c.p2] -= 0.5 * correction * c.p2_inv_mass

    # 3. Update velocities and positions
    for i in particles:
        velocities[i] = (predicted[i] - positions[i]) / dt
        positions[i] = predicted[i]
```

### 8.3 Fluids - SPH (Smoothed Particle Hydrodynamics)

```python
def simulate_sph(dt):
    # 1. Find neighbors (spatial hash)
    neighbors = find_neighbors(particles, h)  # h = smoothing radius

    # 2. Compute density
    for i in particles:
        density[i] = sum(mass[j] * W(pos[i] - pos[j], h) for j in neighbors[i])

    # 3. Compute pressure (equation of state)
    for i in particles:
        pressure[i] = k * (density[i] - rest_density)

    # 4. Compute forces
    for i in particles:
        # Pressure force
        f_pressure = -sum(
            mass[j] * (pressure[i] + pressure[j]) / (2 * density[j])
            * grad_W(pos[i] - pos[j], h)
            for j in neighbors[i]
        )

        # Viscosity force
        f_viscosity = mu * sum(
            mass[j] * (vel[j] - vel[i]) / density[j]
            * laplacian_W(pos[i] - pos[j], h)
            for j in neighbors[i]
        )

        forces[i] = f_pressure + f_viscosity + gravity

    # 5. Integrate
    vel[i] += forces[i] / density[i] * dt
    pos[i] += vel[i] * dt
```

### 8.4 MPM (Material Point Method)

```
Hybrid approach: particles carry mass, grid computes forces

Each step:
1. Particle → Grid: Transfer mass and momentum
2. Grid: Compute forces, update velocities
3. Grid → Particle: Transfer velocity back
4. Particle: Update positions

Good for: sand, snow, mud, fracture
```

---

## 9. Phase 7: ML Integration

### 9.1 Zero-Copy Data Sharing

```python
# Taichi example
import taichi as ti
import torch

ti.init(arch=ti.cuda)

# Taichi field on GPU
positions = ti.Vector.field(3, float, shape=N)

# Get as PyTorch tensor (zero-copy!)
positions_torch = positions.to_torch(device='cuda')

# Modify in PyTorch
positions_torch += neural_network(observations)

# Changes reflected in Taichi field
```

### 9.2 Differentiable Simulation

```python
# Forward pass: run simulation, record operations
# Backward pass: reverse simulation, compute gradients

class DifferentiableStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, action):
        next_state = physics_step(state, action)
        ctx.save_for_backward(state, action, next_state)
        return next_state

    @staticmethod
    def backward(ctx, grad_output):
        state, action, next_state = ctx.saved_tensors

        # Compute Jacobians: ∂next_state/∂state, ∂next_state/∂action
        # Use finite differences or analytical derivatives

        grad_state = jacobian_state.T @ grad_output
        grad_action = jacobian_action.T @ grad_output
        return grad_state, grad_action
```

### 9.3 Batch Environment Interface

```python
class BatchedPhysicsEnv:
    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.scene = create_scene(n_envs=n_envs)

    def reset(self, env_ids=None):
        """Reset specified environments (or all)"""
        if env_ids is None:
            env_ids = range(self.n_envs)
        for i in env_ids:
            self.scene.reset_env(i)
        return self.get_observations()

    def step(self, actions):
        """
        actions: tensor of shape (n_envs, action_dim)
        returns: obs, rewards, dones, infos
        """
        self.scene.apply_actions(actions)
        self.scene.step()
        return (
            self.get_observations(),  # (n_envs, obs_dim)
            self.compute_rewards(),   # (n_envs,)
            self.check_termination(), # (n_envs,)
            {}
        )
```

---

## 10. Algorithm Reference

### Quick Reference Table

| Component | Algorithm | Complexity | When to Use |
|-----------|-----------|------------|-------------|
| **Integration** | Semi-Implicit Euler | O(n) | Default choice |
| | Implicit Euler | O(n³) | Stiff systems |
| | Verlet | O(n) | Particles |
| **Broad Phase** | Spatial Hash | O(n) | Uniform distribution |
| | Sweep & Prune | O(n log n) | Coherent motion |
| | BVH | O(n log n) build, O(log n) query | Static geometry |
| **Narrow Phase** | GJK + EPA | O(iterations) | Convex shapes |
| | SAT | O(faces) | Boxes, polytopes |
| | SDF | O(1) query | Complex shapes |
| **Constraints** | PGS (Sequential Impulse) | O(n * iterations) | Real-time, simple |
| | Direct LCP | O(n³) | Accurate |
| | XPBD | O(n * iterations) | Stable, soft |
| **Soft Body** | FEM | O(elements) | Accurate deformation |
| | PBD | O(constraints * iters) | Fast, stable |
| **Fluid** | SPH | O(n * neighbors) | Splashy liquids |
| | FLIP/PIC | O(grid + particles) | Large volumes |
| | Eulerian | O(grid³) | Smoke, gas |

### Implementation Priority

```
Phase 1 (Foundation):     GPU buffers, time stepping
Phase 2 (Basic Physics):  Single rigid body, gravity, ground plane
Phase 3 (Collision):      Spatial hash + sphere-sphere collision
Phase 4 (Constraints):    Sequential impulses for contacts
Phase 5 (Rendering):      Basic OpenGL/Vulkan rasterization
Phase 6 (Articulated):    Robot joints, Featherstone ABA
Phase 7 (Advanced):       GJK/EPA, friction, soft bodies
Phase 8 (ML):             PyTorch integration, batching
```

---

## 11. Resources & Papers

### Essential Reading

**Rigid Body Dynamics:**
- "Rigid Body Dynamics Algorithms" - Roy Featherstone (2008) - **The bible for articulated bodies**
- "Real-Time Collision Detection" - Christer Ericson (2004) - **Essential for collision**
- "Game Physics Engine Development" - Ian Millington (2010)

**Constraint Solving:**
- "Iterative Dynamics with Temporal Coherence" - Erin Catto (GDC 2005) - **Sequential impulses**
- "Detailed Rigid Body Simulation with Extended Position Based Dynamics" - Müller et al. (2020)

**Collision Detection:**
- "A Fast Procedure for Computing the Distance Between Complex Objects" - Gilbert et al. (1988) - **GJK**
- "Implementing GJK" - Casey Muratori (YouTube) - **Best GJK tutorial**

**Soft Bodies & Fluids:**
- "Position Based Dynamics" - Müller et al. (2007)
- "Smoothed Particle Hydrodynamics" - Monaghan (1992)
- "A Material Point Method for Snow Simulation" - Stomakhin et al. (2013)

### Code References

| Project | Language | Good For |
|---------|----------|----------|
| Box2D | C++ | 2D physics reference |
| Bullet | C++ | Full 3D engine |
| MuJoCo | C | Robotics, contact |
| Taichi | Python | GPU kernels |
| XPBD (paper code) | C++ | Constraint solving |
| Rapier | Rust | Modern, clean design |

### Video Tutorials

- "Constraints Derivation for Rigid Body Simulation" - Erin Catto (GDC)
- "Continuous Collision" - Erin Catto (GDC)
- "Physics for Game Programmers" - Squirrel Eiserloh (GDC)
- "Implementing GJK" - Casey Muratori (YouTube)
- "Two-Bit Coding" (YouTube) - Physics engine series

---

## Implementation Checklist

```
[ ] Phase 1: Core
    [ ] GPU buffer management
    [ ] SoA data layout
    [ ] Time stepping loop
    [ ] Basic scene graph

[ ] Phase 2: Rigid Bodies
    [ ] Single body integration
    [ ] Quaternion rotation
    [ ] Force/torque accumulation
    [ ] Ground plane

[ ] Phase 3: Collision
    [ ] AABB computation
    [ ] Spatial hash broad phase
    [ ] Sphere-sphere narrow phase
    [ ] GJK for convex shapes
    [ ] EPA for penetration depth

[ ] Phase 4: Constraints
    [ ] Contact constraint formulation
    [ ] Sequential impulse solver
    [ ] Friction (Coulomb model)
    [ ] Position correction (Baumgarte)

[ ] Phase 5: Rendering
    [ ] Basic rasterization
    [ ] Camera controls
    [ ] Lighting
    [ ] Shadow mapping

[ ] Phase 6: Articulated Bodies
    [ ] Joint types (revolute, prismatic)
    [ ] Forward kinematics
    [ ] Featherstone ABA
    [ ] Joint limits and motors

[ ] Phase 7: Advanced
    [ ] Contact islands
    [ ] Soft bodies (PBD or FEM)
    [ ] Fluids (SPH)
    [ ] Continuous collision detection

[ ] Phase 8: ML Integration
    [ ] Batch simulation (n_envs)
    [ ] Zero-copy PyTorch interop
    [ ] Gym-compatible API
    [ ] Differentiable physics (optional)
```

---

## Quick Start Recommendation

**Week 1-2:** Get Taichi working, render a falling cube with gravity

**Week 3-4:** Add collision detection (spatial hash + spheres)

**Week 5-6:** Add constraint solver (sequential impulses)

**Week 7-8:** Add robot arm (3 revolute joints)

**Then iterate:** Better collision (GJK), friction, more joints, ML integration

Start simple. A bouncing ball with ground collision is a great first milestone!
