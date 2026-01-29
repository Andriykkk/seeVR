# Genesis Internal Data Structures

How Genesis stores vertices, triangles, physics state, collision data, and rendering information.

---

## Overview

Genesis stores everything in **flat Taichi fields** (GPU arrays) with a batch dimension `_B` for parallel environments. All data is structured for efficient GPU access with coalesced memory patterns.

---

## 1. Geometry Storage (Vertices & Triangles)

### Collision Geometry (per geom)

```python
# Stored as numpy arrays, then flattened into global GPU arrays
_init_verts: np.ndarray    # shape (n_verts, 3) - vertex positions in local frame
_init_faces: np.ndarray    # shape (n_faces, 3) - triangle indices (integer vertex IDs)
_init_normals: np.ndarray  # shape (n_faces, 3) - face normals
_init_edges: np.ndarray    # unique edges from trimesh
```

### Visual Geometry (separate from collision)

```python
_init_vverts: np.ndarray   # visual mesh vertices
_init_vfaces: np.ndarray   # visual mesh triangles
_init_vnormals: np.ndarray # visual normals
_uvs: np.ndarray           # UV texture coordinates
_color: vec4               # RGBA color
_surface: Surface          # surface/material info
```

### Global Flattened Arrays

All meshes from all entities are concatenated into global arrays:

```
Global Vertex Array:
[Entity0_Link0_Geom0_verts][Entity0_Link0_Geom1_verts][Entity0_Link1_verts][Entity1_verts]...

Global Face Array:
[Entity0_Link0_Geom0_faces][Entity0_Link0_Geom1_faces][Entity0_Link1_faces][Entity1_faces]...

Global SDF Cell Array:
[Entity0_Link0_Geom0_sdf][Entity0_Link0_Geom1_sdf][Entity0_Link1_sdf]...
```

Each geometry stores start/end indices to reference into these arrays:
```python
vert_start, vert_end    # indices into global vertex array
face_start, face_end    # indices into global face array
edge_start, edge_end    # indices into global edge array
```

---

## 2. Physics State (Taichi Fields)

### Link State - Rigid Body Transforms

```python
# Shape: (n_links, n_envs) - per link, per environment

# Position and Orientation
pos: ti.Vector.field(3, float)       # world position
quat: ti.Vector.field(4, float)      # orientation quaternion
i_pos: ti.Vector.field(3, float)     # inertial frame position

# Inertial Properties
cinr_inertial: ti.Matrix.field(3, 3) # composite inertia tensor
cinr_mass: ti.field(float)           # mass
cinr_pos: ti.Vector.field(3)         # center of mass position
cinr_quat: ti.Vector.field(4)        # inertia frame orientation

# Velocity
cdd_vel: ti.Vector.field(3)          # linear velocity
cdd_ang: ti.Vector.field(3)          # angular velocity

# Joint Transforms
j_pos: ti.Vector.field(3)            # joint position
j_quat: ti.Vector.field(4)           # joint orientation
j_vel: ti.Vector.field(3)            # joint linear velocity
j_ang: ti.Vector.field(3)            # joint angular velocity

# Accelerations
cacc_ang: ti.Vector.field(3)         # angular acceleration
cacc_lin: ti.Vector.field(3)         # linear acceleration

# Forces
cfrc_ang: ti.Vector.field(3)         # angular force
cfrc_vel: ti.Vector.field(3)         # linear force
cfrc_applied_ang: ti.Vector.field(3) # applied torque
cfrc_applied_vel: ti.Vector.field(3) # applied force
cfrc_coupling_ang/vel                # coupling forces (multi-solver)
contact_force: ti.Vector.field(3)    # contact force

# State
hibernated: ti.field(int)            # sleep state (0=awake, 1=sleeping)
mass_shift: ti.field(float)          # mass modification
root_COM: ti.Vector.field(3)         # root center of mass
```

### DOF State - Joint Values

```python
# Shape: (n_dofs, n_envs) - per degree of freedom, per environment

# Position and Velocity
pos: ti.field(float)           # joint positions (angles for revolute, distance for prismatic)
vel: ti.field(float)           # joint velocities
vel_prev: ti.field(float)      # previous velocity (for integration)
vel_next: ti.field(float)      # next velocity
acc: ti.field(float)           # joint accelerations

# Forces
force: ti.field(float)         # total joint force
qf_bias: ti.field(float)       # bias force (Coriolis, centrifugal)
qf_passive: ti.field(float)    # passive force (damping, springs)
qf_actuator: ti.field(float)   # actuator force
qf_applied: ti.field(float)    # externally applied force
qf_constraint: ti.field(float) # constraint force

# Control
ctrl_force: ti.field(float)    # control force input
ctrl_pos: ti.field(float)      # position control target
ctrl_vel: ti.field(float)      # velocity control target
ctrl_mode: ti.field(int)       # control mode enum

# Composite DOF (for articulated dynamics)
cdof_ang: ti.Vector.field(3)   # composite DOF angular
cdof_vel: ti.Vector.field(3)   # composite DOF velocity
cdofvel_ang/vel                # composite DOF velocities
cdofd_ang/vel                  # composite DOF derivatives

# Backward Pass (for differentiable simulation)
acc_bw: ti.field(float, shape=(2, n_dofs, _B))  # cached accelerations
pos_bw/quat_bw                 # cached positions for gradients
```

### Link Info - Static Properties

```python
# Shape: (n_links,) - static, shared across environments

parent_idx: ti.field(int)      # parent link index (-1 for root)
q_start, q_end: ti.field(int)  # indices into joint position array
dof_start, dof_end: ti.field(int)  # indices into DOF arrays
inertial_pos: ti.Vector.field(3)   # inertial frame offset
inertial_quat: ti.Vector.field(4)  # inertial frame rotation
inertial_mass: ti.field(float)     # link mass
is_fixed: ti.field(bool)           # fixed link flag
entity_idx: ti.field(int)          # parent entity ID
geom_start, geom_end: ti.field(int)  # collision geometry range
```

---

## 3. Collision Data

### Per-Geometry Info (Static)

```python
# Shape: (n_geoms,) - static properties

# Type and Parameters
type: ti.field(int)            # GEOM_TYPE enum (CAPSULE, BOX, MESH, TERRAIN, etc.)
data: ti.Vector.field(7)       # primitive parameters:
                               #   - Sphere: [radius, 0, 0, 0, 0, 0, 0]
                               #   - Box: [half_x, half_y, half_z, 0, 0, 0, 0]
                               #   - Capsule: [radius, half_length, 0, 0, 0, 0, 0]

# Local Transform
pos: ti.Vector.field(3)        # position in link frame
center: ti.Vector.field(3)     # center position
quat: ti.Vector.field(4)       # orientation in link frame

# Physics Properties
friction: ti.field(float)      # friction coefficient
sol_params: ti.Vector.field(7) # solver parameters:
                               # [timeconst, dampratio, dmin, dmax, width, mid, power]

# Topology (indices into global arrays)
vert_start, vert_end: ti.field(int)   # vertex range
face_start, face_end: ti.field(int)   # face range
edge_start, edge_end: ti.field(int)   # edge range
verts_state_start/end: ti.field(int)  # vertex state range

# Collision Filtering
is_convex: ti.field(bool)      # convex geometry flag
contype: ti.field(int)         # collision type bitmask
conaffinity: ti.field(int)     # collision affinity bitmask
needs_coup: ti.field(int)      # needs coupling flag

# Coupling Parameters
coup_friction: ti.field(float)
coup_softness: ti.field(float)
coup_restitution: ti.field(float)
```

### Per-Geometry State (Dynamic)

```python
# Shape: (n_geoms, n_envs) - per environment

pos: ti.Vector.field(3)        # world frame position
quat: ti.Vector.field(4)       # world frame quaternion
aabb_min: ti.Vector.field(3)   # axis-aligned bounding box min
aabb_max: ti.Vector.field(3)   # axis-aligned bounding box max
verts_updated: ti.field(bool)  # flag if vertices need updating
hibernated: ti.field(int)      # sleep state
```

### SDF (Signed Distance Field)

For complex non-convex meshes, precomputed 3D distance grids:

```python
# Per-Geometry SDF Info
T_mesh_to_sdf: ti.Matrix.field(4, 4, shape=(n_geoms,))  # transform to grid coords
sdf_res: ti.Vector.field(3, int, shape=(n_geoms,))     # grid resolution (e.g., 64x64x64)
sdf_cell_size: ti.field(float, shape=(n_geoms,))       # cell size in meters
sdf_max: ti.field(float, shape=(n_geoms,))             # max distance value
sdf_cell_start: ti.field(int, shape=(n_geoms,))        # start index in global SDF array

# Global Flattened SDF Data
geoms_sdf_val: ti.field(float, shape=(n_total_cells,))         # signed distance values
geoms_sdf_grad: ti.Vector.field(3, float, shape=(n_total_cells,))  # gradients
geoms_sdf_closest_vert: ti.field(int, shape=(n_total_cells,))  # closest vertex ID
```

**SDF Caching:** Large SDFs are precomputed and saved to `.gsd` (genesis signed distance) pickle files to avoid recomputation on load.

### Support Field (for GJK Convex Collision)

Precomputed support points in 180×180 directions for fast GJK:

```python
# Shape varies per geometry
support_cell_start: ti.field(int, shape=(n_geoms,))    # start index
support_v: ti.Vector.field(3, float, shape=(n_support_cells,))  # support vertices
support_vid: ti.field(int, shape=(n_support_cells,))   # support vertex IDs
support_res: int = 180  # resolution (180 directions)
```

### Contact Data

```python
# Shape: (max_contacts, n_envs) - per contact, per environment

# Geometry References
geom_a: ti.field(int)          # first geometry ID
geom_b: ti.field(int)          # second geometry ID
link_a: ti.field(int)          # first link ID
link_b: ti.field(int)          # second link ID

# Contact Geometry
normal: ti.Vector.field(3)     # contact normal (points from A to B)
pos: ti.Vector.field(3)        # contact point in world space
penetration: ti.field(float)   # penetration depth (overlap)

# Contact Physics
friction: ti.field(float)      # friction coefficient
sol_params: ti.Vector.field(7) # solver parameters
force: ti.Vector.field(3)      # computed contact force

# Tracking
n_contacts: ti.field(int, shape=(n_envs,))  # number of active contacts
```

### Broad-Phase Collision Data

```python
# Collision pair candidates
broad_collision_pairs: ti.Vector.field(2, int, shape=(max_pairs_broad, n_envs))

# AABB sorting buffers for sweep-and-prune
sort_buffer_min: ti.field(float)
sort_buffer_max: ti.field(float)
sort_buffer_idx: ti.field(int)
```

---

## 4. Constraint Solver Data

```python
# Jacobian Matrix
jac: ti.field(float, shape=(n_constraints, n_dofs, n_envs))

# Mass Matrix (for articulated bodies)
mass_mat: ti.field(float, shape=(n_dofs, n_dofs, n_envs))
mass_mat_L: ti.field(float)      # Cholesky factorization (lower triangular)
mass_mat_D_inv: ti.field(float)  # diagonal inverse
mass_mat_mask: ti.field(bool)    # active DOFs mask

# Constraint Values
efc_force: ti.field(float, shape=(n_constraints, n_envs))  # constraint forces (Lagrange multipliers)
efc_D: ti.field(float)           # constraint impedance
efc_b: ti.field(float)           # constraint bias (for Baumgarte stabilization)
efc_AR: ti.field(float)          # constraint compliance matrix

# Solution
qacc: ti.field(float, shape=(n_dofs, n_envs))           # computed accelerations
qfrc_constraint: ti.field(float, shape=(n_dofs, n_envs)) # constraint forces in joint space

# Intermediate Products
Ma: ti.field(float, shape=(n_dofs, n_envs))  # M @ a (mass matrix times acceleration)
Mgrad: ti.field(float)                        # gradient products

# Convergence
cost: ti.field(float)            # cost function value
gauss: ti.field(float)           # Gauss-Seidel residual
quad: ti.field(float)            # quadratic residual
```

---

## 5. Rendering Data

### Visual Geometry Info (Static)

```python
# Shape: (n_vgeoms,) - static references

pos: ti.Vector.field(3)        # local position relative to link
quat: ti.Vector.field(4)       # local orientation relative to link
link_idx: ti.field(int)        # parent link index
vvert_start, vvert_end: ti.field(int)  # vertex range in global visual array
vface_start, vface_end: ti.field(int)  # face range in global visual array
color: ti.Vector.field(4)      # RGBA color
```

### Visual Geometry State (Dynamic)

```python
# Shape: (n_vgeoms, n_envs) - per environment

pos: ti.Vector.field(3)        # world position
quat: ti.Vector.field(4)       # world orientation
```

### Transform Update Kernel

```python
@ti.kernel
def update_visual_transforms():
    for i in range(n_vgeoms):
        link = vgeom_link_idx[i]
        env = ti.static(range(n_envs))

        # World transform = link transform * local transform
        world_pos[i, env] = links_pos[link, env] + quat_rotate(
            links_quat[link, env],
            local_pos[i]
        )
        world_quat[i, env] = quat_mul(links_quat[link, env], local_quat[i])
```

### Rendering Backends

**Rasterizer (OpenGL/PyRender):**
- Reads visual geometry vertices/faces
- Applies transforms from physics state
- Uses GLSL shaders for lighting, shadows
- Outputs: RGB, Depth, Segmentation, Normal maps

**Ray Tracer (LuisaRender):**
- Same geometry, different pipeline
- Builds BVH acceleration structure
- Monte Carlo path tracing
- Built-in denoising

**Batch Renderer (Madrona):**
- Renders all environments in parallel
- GPU-native batch rendering
- Supports both rasterization and ray tracing modes

---

## 6. Global Configuration

```python
# Shape: varies

# Reference State
qpos0: ti.field(float, shape=(n_qs, n_envs))  # reference joint positions
links_T: ti.Matrix.field(4, 4, shape=(n_links,))  # link transformation matrices

# Awake Tracking (for hibernation optimization)
n_awake_dofs: ti.field(int)
n_awake_entities: ti.field(int)
n_awake_links: ti.field(int)
awake_dofs: ti.field(int, shape=(n_dofs,))
awake_entities: ti.field(int, shape=(n_entities,))
awake_links: ti.field(int, shape=(n_links,))

# Global Parameters
gravity: ti.Vector.field(3, float, shape=(n_envs,))  # per-env gravity
substep_dt: ti.field(float)     # timestep
iterations: ti.field(int)       # solver iterations
tolerance: ti.field(float)      # solver tolerance
noslip_iterations: ti.field(int)   # friction solver iterations
noslip_tolerance: ti.field(float)  # friction solver tolerance
```

---

## 7. Memory Layout Summary

```
Scene
├── Simulator
│   └── RigidSolver
│       │
│       ├── Static Info (shared across envs)
│       │   ├── links_info: (n_links,)
│       │   ├── dofs_info: (n_dofs,)
│       │   ├── geoms_info: (n_geoms,)
│       │   ├── verts: (n_total_verts, 3)      # all vertices concatenated
│       │   ├── faces: (n_total_faces, 3)      # all triangles concatenated
│       │   ├── edges: (n_total_edges, 2)      # all edges concatenated
│       │   ├── sdf_info: (n_geoms,)
│       │   └── sdf_data: (n_total_sdf_cells,) # all SDFs concatenated
│       │
│       └── Dynamic State (per environment)
│           ├── links_state: (n_links, n_envs)
│           ├── dofs_state: (n_dofs, n_envs)
│           ├── geoms_state: (n_geoms, n_envs)
│           ├── verts_state: (n_free_verts, n_envs)  # moving vertices
│           └── contacts: (max_contacts, n_envs)
│
└── Visualizer
    ├── Static
    │   ├── vgeoms_info: (n_vgeoms,)
    │   ├── vverts: (n_visual_verts, 3)
    │   └── vfaces: (n_visual_faces, 3)
    │
    └── Dynamic
        └── vgeoms_state: (n_vgeoms, n_envs)
```

---

## 8. Key Design Patterns

| Pattern | Description | Why |
|---------|-------------|-----|
| **Flat arrays** | All data in contiguous GPU arrays | Coalesced memory access, GPU-friendly |
| **Start/end indices** | Reference ranges in global arrays | Avoid per-entity allocation overhead |
| **Separate collision/visual** | Different meshes for physics vs rendering | Visual can be high-poly, collision low-poly |
| **SDF caching** | Save to `.gsd` files | Avoid expensive recomputation |
| **Support fields** | Precompute GJK support points | Fast convex collision queries |
| **Batch dimension last** | `(n_items, n_envs)` shape | Efficient parallel access pattern |
| **Fixed vs free verts** | Non-batched for static, batched for moving | Memory efficiency |
| **Hibernation tracking** | Skip sleeping bodies | Performance optimization |

---

## 9. Data Flow

```
Loading:
  URDF/Mesh file → Parse → numpy arrays → Taichi fields (GPU)
                                      ↓
                              SDF computation → .gsd cache

Simulation Step:
  1. Read joint controls (ctrl_force, ctrl_pos, ctrl_vel)
  2. Forward kinematics: dofs → link transforms
  3. Update geometry transforms: links → geoms
  4. Broad-phase collision: geoms → candidate pairs
  5. Narrow-phase collision: pairs → contacts
  6. Build constraints: contacts + joints → Jacobian
  7. Solve constraints: Jacobian + forces → accelerations
  8. Integrate: accelerations → velocities → positions
  9. Update visual transforms: links → vgeoms

Rendering:
  vgeoms_state → vertex transforms → rasterizer/raytracer → image
```

---

## 10. Accessing Data (User API)

```python
# Get link positions (all envs)
positions = entity.get_pos()  # shape: (n_envs, 3)

# Get joint positions
qpos = entity.get_qpos()  # shape: (n_envs, n_dofs)

# Get velocities
qvel = entity.get_qvel()  # shape: (n_envs, n_dofs)

# Set controls
entity.control_dofs_force(forces)  # shape: (n_envs, n_dofs)

# Get contact forces
contact_force = entity.get_contact_force()  # shape: (n_envs, 3)

# Direct field access (advanced)
solver = scene.sim.rigid_solver
positions_field = solver.links_state.pos  # Taichi field
positions_torch = positions_field.to_torch()  # Zero-copy to PyTorch
```
