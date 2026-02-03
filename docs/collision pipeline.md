Genesis Collision Pipeline

                        BROAD PHASE
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   Spatial Hash      Sweep & Prune         BVH
   (uniform grid)    (sorted AABBs)     (for complex)
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                      Candidate pairs
                            │
                            ▼
                      NARROW PHASE
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
    Primitives          Convex              Non-convex
   (sphere, box)       (GJK+EPA)           (SDF query)
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                       Contact points
                            │
                            ▼
                    CONSTRAINT SOLVER
                       (Newton)
Genesis Geometry Types

# From Genesis source
GEOM_TYPE:
    PLANE = 0
    SPHERE = 1
    CAPSULE = 2
    BOX = 3
    MESH = 4      # convex hull
    SDF = 5       # signed distance field (non-convex)
    TERRAIN = 6
How Genesis Handles Each Case
1. Primitive vs Primitive (fastest)

# Sphere-sphere: direct formula
def sphere_sphere(a, b):
    d = b.pos - a.pos
    dist = length(d)
    if dist < a.radius + b.radius:
        return Contact(
            normal=d/dist,
            penetration=a.radius + b.radius - dist
        )

# Box-box: SAT (15 axes)
# Capsule-capsule: closest points on line segments
2. Convex vs Convex (GJK + EPA)

def convex_vs_convex(hull_a, hull_b):
    # GJK: Do they intersect?
    if gjk_intersect(hull_a, hull_b):
        # EPA: Find penetration depth & normal
        normal, depth, point = epa(hull_a, hull_b)
        return Contact(normal=normal, penetration=depth, point=point)
    return None
3. Non-Convex (SDF lookup)

def mesh_vs_convex(sdf_mesh, convex):
    contacts = []
    # Check convex vertices against SDF
    for vertex in convex.vertices:
        # Transform to SDF local space
        local_v = sdf_mesh.world_to_local(vertex)
        
        # O(1) lookup in precomputed SDF grid
        distance = sdf_mesh.sample(local_v)
        
        if distance < 0:  # inside
            gradient = sdf_mesh.sample_gradient(local_v)
            contacts.append(Contact(
                point=vertex,
                normal=gradient,
                penetration=-distance
            ))
    return contacts
Genesis Data Structures
Per Geom (collision shape)

# Stored in flat arrays
geom_type: int           # SPHERE, BOX, MESH, SDF...
geom_data: vec7          # type-specific (radius, size, etc.)
geom_pos: vec3           # local offset from body
geom_quat: vec4          # local rotation
vert_start, vert_end: int    # indices into global vertex array
face_start, face_end: int    # indices into global face array
For Convex Hulls

# Global arrays (all hulls concatenated)
all_vertices: [(x,y,z), ...]      # all hull vertices
all_faces: [(i,j,k), ...]         # all hull faces

# Per hull: just store start/end indices
hull.vert_start = 1000
hull.vert_end = 1050     # 50 vertices
For SDF (precomputed)

# 3D grid of signed distances
sdf_grid: float[64][64][64]      # distance at each cell
sdf_gradient: vec3[64][64][64]   # gradient at each cell
sdf_transform: mat4              # world → grid coords
For Your Engine (with BVH + convex)
Step 1: Body Structure

@dataclass
class RigidBody:
    # Physics (same for all shapes)
    position: vec3
    orientation: quat
    velocity: vec3
    angular_velocity: vec3
    mass: float
    inertia: mat3
    inv_inertia: mat3
    
    # Collision (list of convex shapes)
    geoms: list[ConvexHull]
    aabb: AABB  # bounding box of all geoms
Step 2: Convex Hull

@dataclass
class ConvexHull:
    vertices: np.ndarray      # (N, 3) local space
    faces: np.ndarray         # (M, 3) indices
    local_pos: vec3           # offset from body center
    local_rot: quat           # rotation from body
    
    # Precomputed for GJK
    support_points: dict      # direction → furthest vertex (optional)
    
    def support(self, direction):
        """Furthest point in direction (for GJK)"""
        # Transform direction to local space
        local_dir = inverse_rotate(self.local_rot, direction)
        
        # Find furthest vertex
        best = -inf
        best_v = None
        for v in self.vertices:
            d = dot(v, local_dir)
            if d > best:
                best = d
                best_v = v
        
        # Transform back to world
        return self.body.position + rotate(self.body.orientation, 
                   self.local_pos + rotate(self.local_rot, best_v))
Step 3: Collision Pipeline

def detect_collisions(bodies):
    contacts = []
    
    # BROAD PHASE: Your BVH
    candidate_pairs = bvh.query_all_pairs()  # body vs body AABBs
    
    # NARROW PHASE: Geom vs geom
    for body_a, body_b in candidate_pairs:
        for geom_a in body_a.geoms:
            for geom_b in body_b.geoms:
                # Quick AABB check
                if not aabb_overlap(geom_a.world_aabb, geom_b.world_aabb):
                    continue
                
                # GJK + EPA
                contact = gjk_epa(geom_a, geom_b)
                if contact:
                    contact.body_a = body_a
                    contact.body_b = body_b
                    contacts.append(contact)
    
    return contacts
Step 4: Full Frame

def physics_step(bodies, dt):
    # 1. Update AABBs
    for body in bodies:
        body.aabb = compute_world_aabb(body)
    
    # 2. Update BVH (or rebuild)
    bvh.update(bodies)
    
    # 3. Broad phase
    pairs = bvh.find_overlapping_pairs()
    
    # 4. Narrow phase
    contacts = []
    for a, b in pairs:
        contacts.extend(test_body_pair(a, b))
    
    # 5. Solve constraints (Newton)
    newton_solve(bodies, contacts, dt)
    
    # 6. Integrate
    for body in bodies:
        body.velocity += gravity * dt
        body.position += body.velocity * dt
        body.orientation = integrate_quat(body.orientation, body.angular_velocity, dt)
Optimization Tips (Genesis-style)
1. Support Point Caching

# Instead of checking all vertices every time
# Cache support points for common directions

class ConvexHull:
    def __init__(self, vertices):
        self.vertices = vertices
        # Precompute for 180x180 directions
        self.support_cache = precompute_support_field(vertices, resolution=180)
    
    def support(self, direction):
        # O(1) lookup instead of O(n) search
        return self.support_cache.lookup(direction)
2. Temporal Coherence

# Contacts from last frame are likely still valid
# Use as initial guess for GJK

def gjk_with_hint(hull_a, hull_b, previous_simplex=None):
    if previous_simplex:
        simplex = previous_simplex  # Start close to solution
    else:
        simplex = initial_simplex()
    # ... rest of GJK
3. Sleep / Hibernation

# Don't check sleeping bodies
if body.velocity < 0.001 and body.angular_velocity < 0.001:
    body.sleeping = True
    continue  # skip collision detection
Summary

Your setup:
  BVH (have) + Convex hulls (will add) + GJK/EPA (need)
  
Pipeline:
  BVH → candidate body pairs → geom vs geom AABB → GJK/EPA → contacts → Newton
  
Data:
  Body: physics props + list of ConvexHull geoms
  ConvexHull: vertices + faces + local transform + support function