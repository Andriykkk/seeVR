Genesis Primitive Storage
Each geometry type stores only what it needs for fast collision:

Data Layout

# Genesis stores per-geom (from array_class.py):

geom_type: int              # SPHERE=1, CAPSULE=2, BOX=3, MESH=4...
geom_data: vec7             # Type-specific parameters (7 floats)
geom_pos: vec3              # Position (local to body)
geom_quat: vec4             # Orientation (local to body)
The magic: geom_data means different things per type:


# SPHERE (type=1)
geom_data = [radius, 0, 0, 0, 0, 0, 0]

# CAPSULE (type=2)  
geom_data = [radius, half_length, 0, 0, 0, 0, 0]

# BOX (type=3)
geom_data = [half_x, half_y, half_z, 0, 0, 0, 0]

# MESH/CONVEX (type=4)
geom_data = [vert_start, vert_end, face_start, face_end, 0, 0, 0]
Per-Primitive Storage
Sphere

# Only need:
center: vec3     # from geom_pos + body transform
radius: float    # from geom_data[0]

# Collision: 3 subtractions + 1 length + 1 compare
def sphere_sphere(a, b):
    d = length(b.center - a.center)
    return d < a.radius + b.radius
Capsule

# Stored as line segment + radius:
point_a: vec3         # start of line (computed from pos + quat)
point_b: vec3         # end of line (computed from pos + quat + half_length)
radius: float         # from geom_data[0]

# Or equivalently:
center: vec3
axis: vec3            # direction (from quaternion)
half_length: float    # from geom_data[1]
radius: float         # from geom_data[0]

     ╭───╮
    ╱     ╲
   │   A   │  ← point_a
   │   │   │
   │   │   │  ← radius from line
   │   │   │
   │   B   │  ← point_b
    ╲     ╱
     ╰───╯

Capsule = line segment + radius
Box

# Stored as:
center: vec3              # from geom_pos + body transform
orientation: mat3 or quat # from geom_quat + body orientation
half_extents: vec3        # from geom_data[0:3]

# Box axes (columns of rotation matrix):
axis_x: vec3
axis_y: vec3  
axis_z: vec3

    ┌─────────────┐
   ╱             ╱│
  ┌─────────────┐ │
  │      ↑      │ │   half_extents = (hx, hy, hz)
  │      │hy    │ │
  │   ───●───   │╱    center = ●
  │      hx     │
  └─────────────┘
Convex Hull

# Stored as indices into global arrays:
vert_start: int           # where vertices begin
vert_end: int             # where vertices end
face_start: int           # where faces begin
face_end: int             # where faces end

# Plus precomputed for GJK:
support_cache: [vec3...]  # support points in 180×180 directions
Collision Formulas
Sphere-Sphere

def sphere_sphere(a, b):
    # 3 subtractions
    dx = b.center.x - a.center.x
    dy = b.center.y - a.center.y
    dz = b.center.z - a.center.z
    
    # 3 multiplies + 2 adds (squared length, skip sqrt for speed)
    dist_sq = dx*dx + dy*dy + dz*dz
    
    # 1 add + 1 multiply + 1 compare
    sum_r = a.radius + b.radius
    if dist_sq < sum_r * sum_r:
        # Only compute sqrt if collision
        dist = sqrt(dist_sq)
        normal = (dx/dist, dy/dist, dz/dist)
        penetration = sum_r - dist
        return Contact(normal, penetration)
Total: ~10 operations (no collision case)

Sphere-Plane

def sphere_plane(sphere, plane):
    # plane: normal (nx,ny,nz) + distance d
    # dot product: 3 muls + 2 adds
    dist = dot(sphere.center, plane.normal) - plane.d
    
    # 1 compare
    if dist < sphere.radius:
        penetration = sphere.radius - dist
        return Contact(plane.normal, penetration)
Total: ~6 operations

Capsule-Capsule

def capsule_capsule(a, b):
    # Find closest points on two line segments
    # This is the most expensive part (~30 ops)
    closest_a, closest_b = closest_points_on_segments(
        a.point_a, a.point_b,
        b.point_a, b.point_b
    )
    
    # Then sphere-sphere between closest points
    d = length(closest_b - closest_a)
    if d < a.radius + b.radius:
        normal = (closest_b - closest_a) / d
        penetration = a.radius + b.radius - d
        return Contact(normal, penetration)
Total: ~50 operations

Box-Box (SAT)

def box_box(a, b):
    # Test 15 separating axes:
    # 3 face normals from box A
    # 3 face normals from box B
    # 9 edge-edge cross products
    
    axes = [
        a.axis_x, a.axis_y, a.axis_z,
        b.axis_x, b.axis_y, b.axis_z,
        cross(a.axis_x, b.axis_x),
        cross(a.axis_x, b.axis_y),
        cross(a.axis_x, b.axis_z),
        cross(a.axis_y, b.axis_x),
        cross(a.axis_y, b.axis_y),
        cross(a.axis_y, b.axis_z),
        cross(a.axis_z, b.axis_x),
        cross(a.axis_z, b.axis_y),
        cross(a.axis_z, b.axis_z),
    ]
    
    min_penetration = inf
    best_axis = None
    
    for axis in axes:
        # Project both boxes onto axis
        overlap = compute_overlap(a, b, axis)
        if overlap < 0:
            return None  # Separating axis found!
        if overlap < min_penetration:
            min_penetration = overlap
            best_axis = axis
    
    return Contact(best_axis, min_penetration)
Total: ~200-300 operations (early exit if separated)

Genesis Dispatch Table

# From Genesis collider - fast lookup by type pair

@ti.kernel
def narrow_phase():
    for pair_idx in range(n_pairs):
        geom_a = pairs[pair_idx].geom_a
        geom_b = pairs[pair_idx].geom_b
        
        type_a = geoms_info.type[geom_a]
        type_b = geoms_info.type[geom_b]
        
        # Dispatch to specialized function
        if type_a == SPHERE and type_b == SPHERE:
            contact = sphere_sphere(geom_a, geom_b)
        elif type_a == SPHERE and type_b == PLANE:
            contact = sphere_plane(geom_a, geom_b)
        elif type_a == SPHERE and type_b == BOX:
            contact = sphere_box(geom_a, geom_b)
        elif type_a == BOX and type_b == BOX:
            contact = box_box(geom_a, geom_b)
        elif type_a == CAPSULE and type_b == CAPSULE:
            contact = capsule_capsule(geom_a, geom_b)
        # ... etc for all pairs
        else:
            # Fallback: GJK for convex-convex
            contact = gjk_epa(geom_a, geom_b)
Precomputation at Load Time

def create_geom(type, params, local_pos, local_quat):
    geom = Geom()
    geom.type = type
    geom.pos = local_pos
    geom.quat = local_quat
    
    if type == SPHERE:
        geom.data[0] = params.radius
        
    elif type == CAPSULE:
        geom.data[0] = params.radius
        geom.data[1] = params.half_length
        
    elif type == BOX:
        geom.data[0] = params.half_x
        geom.data[1] = params.half_y
        geom.data[2] = params.half_z
        
    elif type == CONVEX:
        # Store vertices in global array
        geom.data[0] = global_verts_count  # vert_start
        global_verts.extend(params.vertices)
        geom.data[1] = global_verts_count  # vert_end
        
        # Precompute support directions (optional, for speed)
        geom.support_cache = precompute_support_field(params.vertices)
    
    return geom
Runtime Transform Update

@ti.kernel
def update_geom_transforms():
    for i in range(n_geoms):
        body_idx = geoms_info.body[i]
        
        # Combine body transform with local geom transform
        geoms_state.pos[i] = bodies_state.pos[body_idx] + \
            quat_rotate(bodies_state.quat[body_idx], geoms_info.pos[i])
        
        geoms_state.quat[i] = quat_mul(
            bodies_state.quat[body_idx], 
            geoms_info.quat[i]
        )
        
        # Update AABB for broad phase
        geoms_state.aabb_min[i], geoms_state.aabb_max[i] = \
            compute_aabb(i, geoms_state.pos[i], geoms_state.quat[i])
Summary
Type	Stored Data	Operations
Sphere	center + radius	~10
Capsule	line segment + radius	~50
Box	center + axes + half_extents	~200
Convex	vertex indices + support cache	~500+ (GJK)
Key optimizations:

Minimal data per primitive (vec7 fits everything)
Type dispatch avoids virtual functions
Precompute support points for GJK
Skip sqrt until confirmed collision
SAT early exit on first separating axis
ok it contain all values, but what about custom objects, complex hulls, links to their triangles to move, etc
Complex Objects Storage
Global Flat Arrays
Genesis concatenates ALL vertices/faces into global arrays:


Body 0 (robot arm):
  Link 0: Hull A (50 verts), Hull B (30 verts)
  Link 1: Hull C (40 verts)
  
Body 1 (box):
  Hull D (8 verts)

GLOBAL VERTEX ARRAY:
┌──────────────────────────────────────────────────────┐
│ Hull A verts │ Hull B verts │ Hull C verts │ Hull D │
│  0-49        │  50-79       │  80-119      │ 120-127│
└──────────────────────────────────────────────────────┘

GLOBAL FACE ARRAY:
┌──────────────────────────────────────────────────────┐
│ Hull A faces │ Hull B faces │ Hull C faces │ Hull D │
│  0-95        │  96-155      │  156-235     │ 236-247│
└──────────────────────────────────────────────────────┘
Per-Geom References

# Each geom stores indices into global arrays
geom_hull_a:
    type = CONVEX
    vert_start = 0
    vert_end = 50
    face_start = 0
    face_end = 96
    link_idx = 0      # which link it belongs to

geom_hull_b:
    type = CONVEX
    vert_start = 50
    vert_end = 80
    face_start = 96
    face_end = 156
    link_idx = 0

geom_hull_c:
    type = CONVEX
    vert_start = 80
    vert_end = 120
    face_start = 156
    face_end = 236
    link_idx = 1      # different link!
Data Structure Overview

┌─────────────────────────────────────────────────────────────┐
│                     STATIC (load once)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Global Vertices:  [v0, v1, v2, ... v10000]  (local space) │
│  Global Faces:     [f0, f1, f2, ... f5000]                 │
│  Global Edges:     [e0, e1, e2, ... e8000]                 │
│                                                             │
│  Per-Geom Info:                                             │
│    ├── type: int                                            │
│    ├── vert_start, vert_end: int                           │
│    ├── face_start, face_end: int                           │
│    ├── link_idx: int  (parent link)                        │
│    ├── local_pos: vec3                                      │
│    ├── local_quat: vec4                                     │
│    └── data: vec7 (primitive params or indices)            │
│                                                             │
│  Per-Link Info:                                             │
│    ├── parent_idx: int                                      │
│    ├── geom_start, geom_end: int                           │
│    └── body_idx: int                                        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    DYNAMIC (updated each frame)             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Per-Link State:  (n_links, n_envs)                        │
│    ├── pos: vec3                                            │
│    └── quat: vec4                                           │
│                                                             │
│  Per-Geom State:  (n_geoms, n_envs)                        │
│    ├── world_pos: vec3                                      │
│    ├── world_quat: vec4                                     │
│    ├── aabb_min: vec3                                       │
│    └── aabb_max: vec3                                       │
│                                                             │
│  Transformed Vertices (optional, for some collision):       │
│    └── world_verts: (n_free_verts, n_envs)                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
How Vertices Move
Option 1: Transform On-Demand (Genesis default)

# Vertices stored in LOCAL space (never change)
# Transform to world space only when needed

@ti.func
def get_world_vertex(geom_idx, local_vert_idx, env_idx):
    # Get geom's world transform
    geom_pos = geoms_state.pos[geom_idx, env_idx]
    geom_quat = geoms_state.quat[geom_idx, env_idx]
    
    # Get local vertex
    global_vert_idx = geoms_info.vert_start[geom_idx] + local_vert_idx
    local_vert = global_verts[global_vert_idx]
    
    # Transform to world
    world_vert = geom_pos + quat_rotate(geom_quat, local_vert)
    return world_vert
Pros: No memory for transformed verts
Cons: Recompute transform every query

Option 2: Pre-Transform All Vertices

# For frequently accessed geometry, transform all verts once per frame

@ti.kernel
def update_world_vertices():
    for geom_idx in range(n_geoms):
        if geoms_info.type[geom_idx] != CONVEX:
            continue
            
        for env_idx in range(n_envs):
            geom_pos = geoms_state.pos[geom_idx, env_idx]
            geom_quat = geoms_state.quat[geom_idx, env_idx]
            
            for i in range(geoms_info.vert_start[geom_idx], 
                          geoms_info.vert_end[geom_idx]):
                local_v = global_verts_local[i]
                world_verts[i, env_idx] = geom_pos + quat_rotate(geom_quat, local_v)
Pros: Transform once, use many times
Cons: Memory for all transformed verts × n_envs

Link Hierarchy (for robots)

Robot arm example:

  Body
    └── Link 0 (base)
          ├── Geom 0 (box)
          └── Link 1 (shoulder)
                ├── Geom 1 (capsule)
                └── Link 2 (elbow)
                      ├── Geom 2 (capsule)
                      └── Link 3 (wrist)
                            └── Geom 3 (gripper hulls)

# Per-link info
links_info:
    link_0: parent=-1,  geom_start=0, geom_end=1
    link_1: parent=0,   geom_start=1, geom_end=2
    link_2: parent=1,   geom_start=2, geom_end=3
    link_3: parent=2,   geom_start=3, geom_end=5  # 2 gripper hulls

# Per-geom info
geoms_info:
    geom_0: link=0, type=BOX
    geom_1: link=1, type=CAPSULE
    geom_2: link=2, type=CAPSULE
    geom_3: link=3, type=CONVEX, vert_start=0,   vert_end=50
    geom_4: link=3, type=CONVEX, vert_start=50,  vert_end=100
Transform Chain

@ti.kernel
def update_geom_world_transforms():
    for geom_idx in range(n_geoms):
        for env_idx in range(n_envs):
            # Get parent link
            link_idx = geoms_info.link_idx[geom_idx]
            
            # Link world transform (computed by forward kinematics)
            link_pos = links_state.pos[link_idx, env_idx]
            link_quat = links_state.quat[link_idx, env_idx]
            
            # Geom local transform (static)
            local_pos = geoms_info.local_pos[geom_idx]
            local_quat = geoms_info.local_quat[geom_idx]
            
            # Combine: world = link_transform × local_transform
            geoms_state.pos[geom_idx, env_idx] = \
                link_pos + quat_rotate(link_quat, local_pos)
            
            geoms_state.quat[geom_idx, env_idx] = \
                quat_mul(link_quat, local_quat)
GJK With Hull References

@ti.func
def gjk_support(geom_idx, direction, env_idx):
    """Find furthest point in direction for convex hull"""
    
    geom_pos = geoms_state.pos[geom_idx, env_idx]
    geom_quat = geoms_state.quat[geom_idx, env_idx]
    
    # Transform direction to local space
    local_dir = quat_rotate_inverse(geom_quat, direction)
    
    # Find furthest vertex
    best_dot = -1e10
    best_vert = vec3(0)
    
    for i in range(geoms_info.vert_start[geom_idx], 
                   geoms_info.vert_end[geom_idx]):
        v = global_verts[i]  # local space vertex
        d = dot(v, local_dir)
        if d > best_dot:
            best_dot = d
            best_vert = v
    
    # Transform result to world space
    return geom_pos + quat_rotate(geom_quat, best_vert)
Memory Layout for Your Engine

class PhysicsWorld:
    # Static (allocated once)
    global_vertices: np.ndarray      # (total_verts, 3) float32
    global_faces: np.ndarray         # (total_faces, 3) int32
    
    geoms_info: StructuredArray
        type: np.ndarray             # (n_geoms,) int
        vert_start: np.ndarray       # (n_geoms,) int
        vert_end: np.ndarray         # (n_geoms,) int
        face_start: np.ndarray       # (n_geoms,) int
        face_end: np.ndarray         # (n_geoms,) int
        link_idx: np.ndarray         # (n_geoms,) int
        local_pos: np.ndarray        # (n_geoms, 3) float
        local_quat: np.ndarray       # (n_geoms, 4) float
        data: np.ndarray             # (n_geoms, 7) float
    
    links_info: StructuredArray
        parent_idx: np.ndarray       # (n_links,) int
        geom_start: np.ndarray       # (n_links,) int
        geom_end: np.ndarray         # (n_links,) int
    
    # Dynamic (updated each frame)
    links_state: StructuredArray
        pos: np.ndarray              # (n_links, 3) float
        quat: np.ndarray             # (n_links, 4) float
        vel: np.ndarray              # (n_links, 3) float
        ang_vel: np.ndarray          # (n_links, 3) float
    
    geoms_state: StructuredArray
        world_pos: np.ndarray        # (n_geoms, 3) float
        world_quat: np.ndarray       # (n_geoms, 4) float
        aabb_min: np.ndarray         # (n_geoms, 3) float
        aabb_max: np.ndarray         # (n_geoms, 3) float
Summary
What	Where Stored	When Updated
Vertex positions (local)	Global array, static	Never (load time only)
Face indices	Global array, static	Never
Geom → vertex mapping	vert_start/end, static	Never
Geom → link mapping	link_idx, static	Never
Link world transform	links_state, dynamic	Every frame (FK)
Geom world transform	geoms_state, dynamic	Every frame
World vertices	Computed on-demand or cached	When needed
Key insight: Vertices never move in local space. Only transforms update. World positions computed when needed.