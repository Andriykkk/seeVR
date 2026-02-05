"""Shared scene data and constants for kernels.

Usage in main.py:
    import taichi as ti
    ti.init(arch=ti.gpu)

    from kernels.data import init_scene, scene, WIDTH, HEIGHT
    init_scene()  # Creates all Taichi fields

    from kernels.raytracing import raytrace  # Import kernels AFTER init_scene()
"""
import taichi as ti

# Constants
MAX_TRIANGLES = 500000
MAX_VERTICES = 500000
MAX_BVH_NODES = MAX_TRIANGLES * 2
WIDTH, HEIGHT = 800, 600

# Physics constants
MAX_BODIES = 1000
MAX_GEOMS = 2000  # Can have multiple geoms per body
MAX_COLLISION_PAIRS = 10000  # Maximum broad phase candidate pairs
MAX_CONTACTS = 10000  # Maximum narrow phase contacts

# Collision geometry types
GEOM_SPHERE = 1
GEOM_BOX = 2
GEOM_PLANE = 3
GEOM_CAPSULE = 4
GEOM_MESH = 5    # Triangle mesh (can be single hull, decomposed, or exact)
GEOM_SDF = 6     # Signed Distance Field

# GEOM_MESH sub-types (stored in data[6])
MESH_SINGLE_HULL = 0       # Single convex hull
MESH_DECOMPOSED = 1        # Multiple convex hulls (CoACD)

# BVH Node structure
# For internal nodes: left_first = left child, right_child = right child, tri_count = 0
# For leaf nodes: left_first = first prim index, right_child unused, tri_count > 0
BVHNode = ti.types.struct(
    aabb_min=ti.types.vector(3, ti.f32),
    aabb_max=ti.types.vector(3, ti.f32),
    left_first=ti.u32,
    right_child=ti.u32,
    tri_count=ti.u32,
    parent_idx=ti.u32,  # Parent node index (0xFFFFFFFF for root)
)

# Rigid Body structure (dynamic state, updated each frame)
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

# Collision Geometry structure (static info, set at load time)
# Each geom belongs to a body and defines a collision shape
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

# Contact structure (output of narrow phase)
Contact = ti.types.struct(
    point=ti.types.vector(3, ti.f32),     # Contact point in world space
    normal=ti.types.vector(3, ti.f32),    # Contact normal (from body_a to body_b)
    depth=ti.f32,                          # Penetration depth (positive = overlapping)
    body_a=ti.i32,                         # First body index
    body_b=ti.i32,                         # Second body index
    geom_a=ti.i32,                         # First geom index
    geom_b=ti.i32,                         # Second geom index
    # Solver state (set in PreStep, used during iterations)
    Pn=ti.f32,                             # Accumulated normal impulse
    bias=ti.f32,                           # Target relative velocity (computed once per frame)
    mass_normal=ti.f32,                    # Effective mass along normal (1/K)
)

# Global scene fields - initialized by init_scene()
vertices = None
original_vertices = None  # Local-space vertices for physics (relative to body center)
indices = None
vertex_colors = None
num_vertices = None
num_triangles = None
velocities = None
pixels = None
bvh_nodes = None
bvh_prim_indices = None
tri_centroids = None
num_bvh_nodes = None
traverse_stack = None
bvh_build_stack = None
morton_codes = None
scene_aabb_min = None
scene_aabb_max = None
# Radix sort temporaries
radix_histogram = None
radix_prefix_sum = None
morton_codes_temp = None
sort_indices = None
sort_indices_temp = None
bvh_aabb_flags = None  # Atomic flags for AABB propagation

# Physics fields
bodies = None
geoms = None
num_bodies = None
num_geoms = None
gravity = None  # Gravity vector (adjustable)
# Local vertices for collision geometry (convex hulls, stored in local space)
collision_verts = None
num_collision_verts = None
# Collision faces (triangle indices into collision_verts)
collision_faces = None
num_collision_faces = None
# Broad phase collision pairs (candidate pairs whose AABBs overlap)
collision_pairs = None  # [pair_idx, 0] = geom_a, [pair_idx, 1] = geom_b
num_collision_pairs = None
# Narrow phase contacts
contacts = None
num_contacts = None
# Debug rendering
debug_geom_verts = None
debug_geom_colors = None
debug_geom_indices = None
# Debug normal arrows (2 verts per face: center and center+normal)
debug_normal_verts = None
debug_normal_colors = None
# Debug contact points (point + normal arrow)
debug_contact_points = None
debug_contact_normals = None  # 2 verts per contact for normal line
# Debug solver forces (impulse arrows at contact points)
debug_force_verts = None  # 2 verts per contact for impulse arrow
debug_force_colors = None


def init_scene():
    """Initialize all Taichi fields. Call after ti.init()"""
    global vertices, original_vertices, indices, vertex_colors, num_vertices, num_triangles
    global velocities, pixels, bvh_nodes, bvh_prim_indices, tri_centroids
    global num_bvh_nodes, traverse_stack, bvh_build_stack
    global morton_codes, scene_aabb_min, scene_aabb_max
    global radix_histogram, radix_prefix_sum, morton_codes_temp
    global sort_indices, sort_indices_temp, bvh_aabb_flags
    global bodies, geoms, num_bodies, num_geoms, gravity
    global collision_verts, num_collision_verts
    global collision_faces, num_collision_faces
    global collision_pairs, num_collision_pairs
    global contacts, num_contacts
    global debug_geom_verts, debug_geom_colors, debug_geom_indices
    global debug_normal_verts, debug_normal_colors
    global debug_contact_points, debug_contact_normals
    global debug_force_verts, debug_force_colors

    # Geometry
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)
    original_vertices = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)  # Local-space
    indices = ti.field(dtype=ti.i32, shape=MAX_TRIANGLES * 3)
    vertex_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)
    num_vertices = ti.field(dtype=ti.i32, shape=())
    num_triangles = ti.field(dtype=ti.i32, shape=())
    velocities = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)

    # Pixel buffer
    pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

    # BVH
    bvh_nodes = BVHNode.field(shape=MAX_BVH_NODES)
    bvh_prim_indices = ti.field(dtype=ti.i32, shape=MAX_TRIANGLES)
    tri_centroids = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIANGLES)
    num_bvh_nodes = ti.field(dtype=ti.i32, shape=())
    traverse_stack = ti.field(dtype=ti.i32, shape=(WIDTH, HEIGHT, 64))
    bvh_build_stack = ti.field(dtype=ti.i32, shape=MAX_BVH_NODES)
    morton_codes = ti.field(dtype=ti.u32, shape=MAX_TRIANGLES)
    scene_aabb_min = ti.Vector.field(3, dtype=ti.f32, shape=())
    scene_aabb_max = ti.Vector.field(3, dtype=ti.f32, shape=())

    # Radix sort temporaries (256 buckets for 8-bit digits)
    radix_histogram = ti.field(dtype=ti.i32, shape=256)
    radix_prefix_sum = ti.field(dtype=ti.i32, shape=256)
    morton_codes_temp = ti.field(dtype=ti.u32, shape=MAX_TRIANGLES)
    sort_indices = ti.field(dtype=ti.i32, shape=MAX_TRIANGLES)
    sort_indices_temp = ti.field(dtype=ti.i32, shape=MAX_TRIANGLES)

    # Atomic flags for AABB propagation (one per internal node)
    bvh_aabb_flags = ti.field(dtype=ti.i32, shape=MAX_TRIANGLES)

    # Physics: Rigid bodies and collision geometry
    bodies = RigidBody.field(shape=MAX_BODIES)
    geoms = CollisionGeom.field(shape=MAX_GEOMS)
    num_bodies = ti.field(dtype=ti.i32, shape=())
    num_geoms = ti.field(dtype=ti.i32, shape=())
    gravity = ti.Vector.field(3, dtype=ti.f32, shape=())  # Adjustable gravity
    # Collision vertices for convex hulls (local space, never change)
    collision_verts = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)
    num_collision_verts = ti.field(dtype=ti.i32, shape=())
    # Collision faces (triangle indices into collision_verts)
    collision_faces = ti.Vector.field(3, dtype=ti.i32, shape=MAX_TRIANGLES)
    num_collision_faces = ti.field(dtype=ti.i32, shape=())

    # Broad phase collision pairs
    collision_pairs = ti.Vector.field(2, dtype=ti.i32, shape=MAX_COLLISION_PAIRS)
    num_collision_pairs = ti.field(dtype=ti.i32, shape=())

    # Narrow phase contacts
    contacts = Contact.field(shape=MAX_CONTACTS)
    num_contacts = ti.field(dtype=ti.i32, shape=())

    # Debug rendering: world-space collision geometry
    debug_geom_verts = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)
    debug_geom_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)
    debug_geom_indices = ti.field(dtype=ti.i32, shape=MAX_TRIANGLES * 3)
    # Debug normal arrows (2 verts per face for lines)
    debug_normal_verts = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIANGLES * 2)
    debug_normal_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIANGLES * 2)
    # Debug contact points and normals
    debug_contact_points = ti.Vector.field(3, dtype=ti.f32, shape=MAX_CONTACTS)
    debug_contact_normals = ti.Vector.field(3, dtype=ti.f32, shape=MAX_CONTACTS * 2)  # 2 verts per line
    # Debug solver forces (impulse arrows)
    debug_force_verts = ti.Vector.field(3, dtype=ti.f32, shape=MAX_CONTACTS * 2)  # 2 verts per line
    debug_force_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_CONTACTS * 2)

    # Initialize counts
    num_vertices[None] = 0
    num_triangles[None] = 0
    num_bodies[None] = 0
    num_geoms[None] = 0
    num_collision_verts[None] = 0
    num_collision_faces[None] = 0
    num_collision_pairs[None] = 0
    num_contacts[None] = 0
    gravity[None] = [0.0, -9.81, 0.0]  # Default Earth gravity
    # gravity[None] = [0.0, -2.0, 0.0]  # Default Earth gravity
