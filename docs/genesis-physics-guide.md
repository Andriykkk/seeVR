# Genesis Physics Engine Guide

A comprehensive guide to understanding how physics work in the Genesis robotics simulation framework.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Time Stepping](#time-stepping)
4. [Rigid Body Dynamics](#rigid-body-dynamics)
5. [Collision Detection](#collision-detection)
6. [Constraint Solving](#constraint-solving)
7. [Multi-Solver System](#multi-solver-system)
8. [Materials and Properties](#materials-and-properties)
9. [Configuration Reference](#configuration-reference)
10. [Code Examples](#code-examples)

---

## Architecture Overview

Genesis is a **multi-solver physics engine** with a modular, layered architecture:

```
Scene (top-level wrapper)
  └── Simulator (manages solvers + coupling)
        ├── RigidSolver (rigid bodies)
        ├── MPMSolver (deformable bodies)
        ├── SPHSolver (fluids)
        ├── PBDSolver (cloth/particles)
        ├── FEMSolver (elastic bodies)
        ├── SFSolver (smoke/flow)
        └── Coupler (inter-solver interactions)
```

### Key Files

| Component | File Path |
|-----------|-----------|
| Scene | `Genesis/genesis/engine/scene.py` |
| Simulator | `Genesis/genesis/engine/simulator.py` |
| Rigid Solver | `Genesis/genesis/engine/solvers/rigid/rigid_solver.py` |
| Collision | `Genesis/genesis/engine/solvers/rigid/collider/collider.py` |
| Constraints | `Genesis/genesis/engine/solvers/rigid/constraint/solver.py` |
| Options | `Genesis/genesis/options/solvers.py` |
| Materials | `Genesis/genesis/engine/materials/rigid.py` |

---

## Core Components

### Scene

The `Scene` class is the top-level container that wraps everything:

```python
# Genesis/genesis/engine/scene.py

scene = gs.Scene(
    sim_options=gs.SimOptions(...),
    rigid_options=gs.RigidOptions(...),
    viewer_options=gs.ViewerOptions(...)
)
```

### Simulator

The `Simulator` manages the physics loop and coordinates multiple solvers:

```python
# Genesis/genesis/engine/simulator.py

# Main simulation loop
simulator.step()  # Advances simulation by dt
```

**Step hierarchy:**
```
step() → main step (dt)
  └── substep() → sub-step (dt/substeps)
        ├── process_input()
        ├── substep_pre_coupling()   # Each solver's internal dynamics
        ├── _coupler.couple()        # Inter-solver interactions
        └── substep_post_coupling()  # Finalization
```

---

## Time Stepping

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 0.01s | Main simulation timestep |
| `substeps` | 1 | Number of sub-steps per main step |
| `gravity` | (0, 0, -9.81) | Global gravity vector |

### How It Works

```python
# Effective sub-step duration
substep_dt = dt / substeps

# Current simulation time
cur_t = cur_substep_global * substep_dt
```

**Why substeps matter:**
- More substeps = better stability for stiff systems
- Trade-off between accuracy and performance
- Rule of thumb: increase substeps for fast-moving objects or stiff constraints

### Example

```python
scene = gs.Scene(
    sim_options=gs.SimOptions(
        dt=0.01,        # 10ms main step (100 Hz)
        substeps=4,     # 4 sub-steps = 2.5ms each (400 Hz internal)
        gravity=(0, 0, -9.81)
    )
)
```

---

## Rigid Body Dynamics

Genesis uses **Articulated Body Dynamics (ABD)** for rigid body simulation.

### Key File
`Genesis/genesis/engine/solvers/rigid/abd/forward_dynamics.py`

### Integration Schemes

| Integrator | Description | Use Case |
|------------|-------------|----------|
| `Euler` | Simple explicit Euler | Fast, less stable |
| `implicitfast` | Implicit integration (MuJoCo-compatible) | Stable, accurate |
| `approximate_implicitfast` | Faster approximation (default) | Good balance |

### Integration Flow

```python
# Basic integration (simplified)
vel_new = vel_old + acc * dt           # Velocity update
pos_new = pos_old + vel_new * dt       # Position update

# Implicit methods add damping correction
vel_new = vel_old + acc * dt * (1 - damping_factor)
```

### Forward Dynamics Pipeline

1. **Forward Kinematics** - Update link positions from joint angles
2. **Mass Matrix Computation** - Calculate inertia properties
3. **Force Accumulation** - External, actuation, constraint, bias forces
4. **Acceleration Computation** - Solve for accelerations
5. **Integration** - Update velocities and positions

---

## Collision Detection

### Key File
`Genesis/genesis/engine/solvers/rigid/collider/collider.py`

### Three-Phase Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Broad Phase   │ --> │  Narrow Phase   │ --> │ Contact Generation│
│  (AABB overlap) │     │ (GJK/MPR/SDF)   │     │  (up to 4 pts)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Phase Details

**1. Broad Phase**
- Sweep-and-prune algorithm using AABBs
- Quickly eliminates non-colliding pairs
- O(n log n) complexity

**2. Narrow Phase**

| Algorithm | Description | When Used |
|-----------|-------------|-----------|
| **GJK** | Gilbert-Johnson-Keerthi | Convex-convex, required for gradients |
| **MPR** | Minkowski Portal Refinement | Alternative convex collision |
| **SDF** | Signed Distance Field | Non-convex geometry |
| **Terrain** | Specialized | Ground plane collisions |

**3. Contact Generation**
- Generates up to 4 contact constraints per pair
- Computes contact points, normals, penetration depth

### Configuration

```python
gs.RigidOptions(
    enable_collision=True,
    enable_self_collision=True,
    enable_adjacent_collision=False,  # Parent-child pairs
    max_collision_pairs=150,
    use_gjk_collision=False,          # True required for gradients
    box_box_detection=True            # Optimized box-box
)
```

---

## Constraint Solving

### Key File
`Genesis/genesis/engine/solvers/rigid/constraint/solver.py`

### Solver Types

| Solver | Description |
|--------|-------------|
| `Newton` | Newton's method (default) |
| `CG` | Conjugate Gradient |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `iterations` | 50 | Main solver iterations |
| `tolerance` | 1e-6 | Convergence threshold |
| `ls_iterations` | 50 | Line search iterations |
| `ls_tolerance` | 1e-2 | Line search tolerance |
| `constraint_timeconst` | 0.01 | Constraint stiffness (smaller = stiffer) |
| `noslip_iterations` | 0 | Friction post-processing |

### Constraint Time Constant

The `constraint_timeconst` parameter is critical:
- Controls how quickly constraints are enforced
- **Smaller values** = stiffer, more accurate constraints
- **Must be >= 2 * substep_dt** for stability
- If constraints feel "soft", reduce this value (and increase substeps if needed)

### Example

```python
gs.RigidOptions(
    constraint_solver=gs.constraint_solver.Newton,
    iterations=100,           # More iterations for complex scenes
    tolerance=1e-8,           # Tighter tolerance
    constraint_timeconst=0.005  # Stiffer constraints
)
```

---

## Multi-Solver System

Genesis can run multiple physics solvers simultaneously for different material types.

### Available Solvers

| Solver | Material Type | Method |
|--------|---------------|--------|
| **RigidSolver** | Rigid bodies | Articulated Body Dynamics |
| **MPMSolver** | Deformable (sand, snow, elastic) | Material Point Method |
| **SPHSolver** | Fluids | Smoothed Particle Hydrodynamics |
| **PBDSolver** | Cloth, particles | Position-Based Dynamics |
| **FEMSolver** | Elastic bodies | Finite Element Method |
| **SFSolver** | Smoke, gas | Smoke Flow |

### Coupling Systems

Couplers manage interactions between different solvers:

**1. LegacyCoupler** - Simple boolean flags
```python
# Default: all couplings enabled
rigid_mpm=True, rigid_sph=True, rigid_pbd=True, rigid_fem=True
```

**2. SAPCoupler** - Semi-Analytic Primal (Drake-compatible)
```python
coupler_options=gs.CouplerOptions(
    coupler_type='sap',
    n_sap_iterations=5,
    contact_d_hat=0.001,      # Contact distance threshold
    contact_friction_mu=0.5,
    contact_resistance=1e9    # Contact stiffness
)
```

**3. IPCCoupler** - Incremental Potential Contact
- Best for cloth/FEM interactions
- Two-way coupling with rigid bodies
- Advanced self-contact detection

---

## Materials and Properties

### Key File
`Genesis/genesis/engine/materials/rigid.py`

### Rigid Material Properties

| Property | Default | Range | Description |
|----------|---------|-------|-------------|
| `friction` | None | [0.01, 5.0] | Coefficient of friction |
| `rho` | 200.0 | - | Density (kg/m^3) for mass calculation |
| `coup_friction` | 0.1 | - | Friction during solver coupling |
| `coup_softness` | 0.002 | - | Softness of coupling contacts |
| `coup_restitution` | 0.0 | [0, 1] | Bounciness (0=inelastic, 1=elastic) |
| `gravity_compensation` | 0.0 | [0, 1] | Factor to cancel gravity |

### Example

```python
# Create a bouncy rubber material
rubber = gs.materials.Rigid(
    friction=0.8,
    rho=1100.0,              # Rubber density
    coup_restitution=0.7     # Bouncy
)

# Create a heavy metal material
metal = gs.materials.Rigid(
    friction=0.3,
    rho=7800.0,              # Steel density
    coup_restitution=0.1     # Low bounce
)

# Apply to entity
scene.add_entity(gs.morphs.Sphere(radius=0.1), material=rubber)
```

---

## Configuration Reference

### SimOptions (Global)

```python
gs.SimOptions(
    dt=0.01,                    # Main timestep (seconds)
    substeps=1,                 # Sub-steps per main step
    substeps_local=1,           # For differentiable mode checkpointing
    gravity=(0, 0, -9.81),      # Gravity vector
    floor_height=0.0,           # Ground plane height
    requires_grad=False         # Enable differentiable simulation
)
```

### RigidOptions (Rigid Body Solver)

```python
gs.RigidOptions(
    # Collision
    enable_collision=True,
    enable_self_collision=True,
    enable_joint_limit=True,
    enable_adjacent_collision=False,
    enable_neutral_collision=False,
    max_collision_pairs=150,
    use_gjk_collision=False,
    box_box_detection=True,

    # Integration
    integrator=gs.integrator.approximate_implicitfast,

    # Constraint Solver
    constraint_solver=gs.constraint_solver.Newton,
    iterations=50,
    tolerance=1e-6,
    ls_iterations=50,
    ls_tolerance=1e-2,
    constraint_timeconst=0.01,
    noslip_iterations=0,
    noslip_tolerance=1e-6
)
```

### Performance Options

```python
gs.RigidOptions(
    # Hibernation (sleeping bodies)
    enable_hibernation=False,
    hibernation_vel_thresh=1e-3,
    hibernation_acc_thresh=1e-2,

    # Contact Islands (group connected bodies)
    enable_contact_island=False
)
```

---

## Code Examples

### Basic Scene Setup

```python
import genesis as gs

# Initialize Genesis
gs.init(backend=gs.cpu)  # or gs.gpu

# Create scene with physics options
scene = gs.Scene(
    sim_options=gs.SimOptions(
        dt=0.01,
        substeps=2,
        gravity=(0, 0, -9.81)
    ),
    rigid_options=gs.RigidOptions(
        enable_collision=True,
        integrator=gs.integrator.approximate_implicitfast
    )
)

# Add ground plane
scene.add_entity(gs.morphs.Plane())

# Add a falling box
box = scene.add_entity(
    gs.morphs.Box(size=(0.1, 0.1, 0.1)),
    material=gs.materials.Rigid(friction=0.5, rho=500),
    morph=gs.morphs.Morph(pos=(0, 0, 1))  # Start 1m high
)

# Build the scene
scene.build()

# Run simulation
for i in range(1000):
    scene.step()
    if i % 100 == 0:
        pos = box.get_pos()
        print(f"Step {i}: box at z={pos[2]:.3f}")
```

### Robot Simulation

```python
import genesis as gs

gs.init()

scene = gs.Scene(
    sim_options=gs.SimOptions(dt=0.005, substeps=4),
    rigid_options=gs.RigidOptions(
        enable_collision=True,
        enable_self_collision=True,
        constraint_timeconst=0.005
    )
)

# Add ground
scene.add_entity(gs.morphs.Plane())

# Load robot from URDF
robot = scene.add_entity(
    gs.morphs.URDF(file='path/to/robot.urdf'),
    material=gs.materials.Rigid(friction=0.8)
)

scene.build()

# Control loop
for step in range(10000):
    # Get current state
    q = robot.get_qpos()      # Joint positions
    dq = robot.get_qvel()     # Joint velocities

    # Compute control (e.g., PD controller)
    q_target = [0.0] * len(q)
    kp, kd = 100.0, 10.0
    tau = kp * (q_target - q) - kd * dq

    # Apply control
    robot.control_dofs_force(tau)

    # Step simulation
    scene.step()
```

### Differentiable Simulation

```python
import genesis as gs
import torch

gs.init(backend=gs.gpu)

scene = gs.Scene(
    sim_options=gs.SimOptions(
        dt=0.01,
        substeps=2,
        requires_grad=True  # Enable gradients!
    ),
    rigid_options=gs.RigidOptions(
        use_gjk_collision=True  # Required for gradients
    )
)

# ... add entities ...

scene.build()

# Forward pass
initial_vel = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
entity.set_vel(initial_vel)

for _ in range(100):
    scene.step()

final_pos = entity.get_pos()

# Backward pass - compute gradients
loss = (final_pos - target_pos).norm()
loss.backward()

# Now initial_vel.grad contains d(loss)/d(initial_vel)
print(f"Gradient: {initial_vel.grad}")
```

### Multi-Material Scene

```python
import genesis as gs

gs.init()

scene = gs.Scene(
    sim_options=gs.SimOptions(dt=0.01, substeps=4),
    mpm_options=gs.MPMOptions(grid_density=64),
    sph_options=gs.SPHOptions()
)

# Rigid container
container = scene.add_entity(
    gs.morphs.URDF(file='container.urdf'),
    material=gs.materials.Rigid(friction=0.3)
)

# Deformable sand (MPM)
sand = scene.add_entity(
    gs.morphs.Box(size=(0.2, 0.2, 0.1)),
    material=gs.materials.MPM.Sand()
)

# Water (SPH)
water = scene.add_entity(
    gs.morphs.Box(size=(0.1, 0.1, 0.1)),
    material=gs.materials.SPH.Liquid()
)

scene.build()

# Solvers automatically coupled
for _ in range(1000):
    scene.step()
```

---

## Performance Tips

1. **Reduce `max_collision_pairs`** if you know your scene has few collisions
2. **Enable hibernation** for scenes with objects that come to rest
3. **Use `box_box_detection=True`** for scenes with many boxes
4. **Increase `substeps`** instead of reducing `dt` for stability
5. **Use `approximate_implicitfast`** integrator (default) for best performance
6. **Disable `enable_self_collision`** if not needed
7. **Use contact islands** for scenes with many disconnected groups

---

## Troubleshooting

| Problem | Possible Cause | Solution |
|---------|----------------|----------|
| Objects pass through each other | Too large dt, not enough substeps | Increase substeps or decrease dt |
| Constraints feel "soft" | constraint_timeconst too large | Reduce constraint_timeconst |
| Simulation explodes | Numerical instability | Use implicit integrator, increase substeps |
| Slow performance | Too many collision pairs | Reduce max_collision_pairs, enable hibernation |
| Jittery contacts | Not enough solver iterations | Increase iterations |
| Gradients are NaN | Using MPR collision | Switch to use_gjk_collision=True |

---

## Further Reading

- Main solver implementation: `Genesis/genesis/engine/solvers/rigid/`
- Options definitions: `Genesis/genesis/options/solvers.py`
- Material definitions: `Genesis/genesis/engine/materials/`
- Example scripts: `Genesis/examples/`
