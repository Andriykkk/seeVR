import taichi as ti

ti.init(arch=ti.gpu)

DEBUG = True

# --- Constants ---

FIXED_DT = 1.0 / 60
FRAME_TIME = 1.0 / 60

# --- Data ---

MAX_VERTICES = 100000
MAX_TRIANGLES = 100000
WIDTH, HEIGHT = 800, 600
MAX_BODIES = 1000
MAX_GEOMS = 2000
MAX_COLLISION_PAIRS = 10000

GEOM_SPHERE = 1
GEOM_BOX = 2
GEOM_PLANE = 3
GEOM_CAPSULE = 4
GEOM_MESH = 5
GEOM_SDF = 6

RigidBody = ti.types.struct(
    pos=ti.types.vector(3, ti.f32),       # Center of mass position
    quat=ti.types.vector(4, ti.f32),      # Orientation quaternion (w, x, y, z)
    vel=ti.types.vector(3, ti.f32),       # Linear velocity
    omega=ti.types.vector(3, ti.f32),     # Angular velocity
    mass=ti.f32,                          # Mass (0 = static/infinite mass)
    inv_mass=ti.f32,                      # 1/mass (0 for static)
    inertia=ti.types.vector(3, ti.f32),   # Diagonal inertia tensor (local space)
    inv_inertia=ti.types.vector(3, ti.f32), # 1/inertia
    # Render mesh mapping
    vert_start=ti.i32,                    # Start index in vertices array
    vert_count=ti.i32,                    # Number of vertices for this body
)

CollisionGeom = ti.types.struct(
    geom_type=ti.i32,                     # GEOM_SPHERE, GEOM_BOX, etc.
    body_idx=ti.i32,                      # Which rigid body owns this geom
    local_pos=ti.types.vector(3, ti.f32), # Offset from body center
    local_quat=ti.types.vector(4, ti.f32), # Rotation relative to body
    # Type-specific data (like Genesis vec7):
    # SPHERE: [radius, 0, 0, 0, 0, 0, 0]
    # BOX: [half_x, half_y, half_z, 0, 0, 0, 0]
    # CAPSULE: [radius, half_length, 0, 0, 0, 0, 0]
    # PLANE: [normal_x, normal_y, normal_z, 0, 0, 0, 0]
    # MESH: [data_start, data_count, hull_count, volume, error, 0, mesh_subtype]
    #   - mesh_subtype: MESH_SINGLE_HULL, MESH_DECOMPOSED
    # SDF: [grid_start, resolution_x, resolution_y, resolution_z, voxel_size, 0, 0]
    data=ti.types.vector(7, ti.f32),
    # Cached world-space transform (updated each frame)
    world_pos=ti.types.vector(3, ti.f32),
    world_quat=ti.types.vector(4, ti.f32),
    aabb_min=ti.types.vector(3, ti.f32),
    aabb_max=ti.types.vector(3, ti.f32),
)

@ti.data_oriented
class Data:
    def __init__(self):
        # --- Render: mesh geometry passed to ti_scene.mesh() ---
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)          # world-space vertex positions (recomputed each frame)
        self.original_vertices = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)  # local-space vertices (relative to body center, never changes)
        self.indices = ti.field(dtype=ti.i32, shape=MAX_TRIANGLES * 3)                # 3 indices per triangle into vertices
        self.vertex_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)     # per-vertex RGB color
        self.num_vertices = ti.field(dtype=ti.i32, shape=())                          # current vertex count (atomic counter)
        self.num_triangles = ti.field(dtype=ti.i32, shape=())                         # current triangle count (atomic counter)

        # --- Physics: rigid body state updated each simulation step ---
        self.bodies = RigidBody.field(shape=MAX_BODIES)                               # pos, quat, vel, inertia per body
        self.num_bodies = ti.field(dtype=ti.i32, shape=())                            # current body count (atomic counter)
        self.geoms = CollisionGeom.field(shape=MAX_GEOMS)                             # collision shapes attached to bodies
        self.num_geoms = ti.field(dtype=ti.i32, shape=())                             # current geom count (atomic counter)
        self.gravity = ti.Vector.field(3, dtype=ti.f32, shape=())                    # gravity vector (adjustable)

        # --- Collision: broad phase output ---
        self.collision_pairs = ti.Vector.field(2, dtype=ti.i32, shape=MAX_COLLISION_PAIRS)  # write: broad_phase | read: narrow_phase
        self.num_collision_pairs = ti.field(dtype=ti.i32, shape=())                         # write: broad_phase | read: narrow_phase, GUI

        # --- Debug: wireframe overlay built from indices each frame ---
        if DEBUG:
            self.wire_verts = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIANGLES * 6)   # 3 edges * 2 endpoints per triangle
            self.num_wire_verts = ti.field(dtype=ti.i32, shape=())                        # = num_triangles * 6
            self.aabb_verts = ti.Vector.field(3, dtype=ti.f32, shape=MAX_GEOMS * 24)      # 12 edges * 2 endpoints per geom
            self.num_aabb_verts = ti.field(dtype=ti.i32, shape=())                        # = num_geoms * 24

    @staticmethod
    def _dtype_size(dtype):
        """Bytes for a Taichi dtype, parsed from its name (e.g. f32→4, i64→8, u8→1)."""
        import re
        m = re.search(r'(\d+)', str(dtype))
        return int(m.group(1)) // 8 if m else 4

    @staticmethod
    def _elem_bytes(field):
        """Bytes per element, introspected from the Taichi field."""
        if hasattr(field, 'keys'):  # struct field — recurse into members
            return sum(Data._elem_bytes(getattr(field, k)) for k in field.keys)
        sz = Data._dtype_size(field.dtype)
        if hasattr(field, 'n'):     # vector field
            return field.n * sz
        return sz                   # scalar field

    def _iter_fields(self):
        """Yield (name, field, counter_field_or_None) for every Taichi field with shape."""
        import math
        for name in vars(self):
            if name.startswith('_'):
                continue
            field = getattr(self, name)
            if not hasattr(field, 'shape'):
                continue
            # skip scalar counters (shape=()) — they are used as counters, not data
            if not field.shape:
                continue
            # try to find a matching num_{name} counter
            counter = getattr(self, f"num_{name}", None)
            yield name, field, counter

    def gpu_memory(self):
        """Returns (allocated_bytes, used_bytes, per-field details)."""
        import math
        allocated = 0
        used = 0
        details = []
        for name, field, counter in self._iter_fields():
            eb = self._elem_bytes(field)
            max_count = math.prod(field.shape)
            a = max_count * eb
            u = counter[None] * eb if counter is not None else a
            allocated += a
            used += u
            details.append((name, a, u))
        return allocated, used, details

    def gpu_memory_str(self):
        allocated, used, details = self.gpu_memory()
        lines = [f"GPU: {used / 1048576:.1f} / {allocated / 1048576:.1f} MB"]
        for name, a, u in details:
            pct = u * 100 // a if a > 0 else 0
            lines.append(f"  {name:<16} {u / 1024:>7.0f} / {a / 1024:>7.0f} KB  ({pct}%)")
        return "\n".join(lines)


data = Data()
# data.gravity[None] = [0.0, -9.81, 0.0]
data.gravity[None] = [0.0, -0.0, 0.0]
