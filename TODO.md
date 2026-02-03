[ ] physics
Component	What Genesis Uses	Realism
Shape representation	Convex + SDF	⭐⭐⭐ Exact (no approximation)
Collision detection	GJK + EPA	⭐⭐⭐ Mathematically exact
Constraint solver	Newton	⭐⭐⭐ Converges to exact
Integration	Implicit Euler	⭐⭐⭐ Stable + accurate

Physics Architecture with Newton Method - Step by Step
Key Insight: Two Separate Data Sets

┌─────────────────────────────────────────────────────────────────┐
│                    RENDERING (existing)                         │
│   data.vertices: 10,000 triangles (full mesh detail)           │
│   data.indices: triangle connectivity                          │
│   → Updated every frame by physics transforms                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    PHYSICS (new)                                │
│   collision_vertices: ~100 verts (simplified convex hulls)     │
│   Stored in LOCAL space, NEVER move                            │
│   → Transformed on-demand during collision queries             │
└─────────────────────────────────────────────────────────────────┘
Data Structures
Rigid Body (dynamic state, updated each frame):


body:
    pos: vec3          # Center of mass
    quat: vec4         # Orientation  
    vel: vec3          # Linear velocity
    omega: vec3        # Angular velocity
    mass: float        # (or inv_mass for static: 0)
    inertia: mat3      # In local space
Collision Geom (static info, set at load time):


geom:
    type: int          # SPHERE=1, BOX=2, CAPSULE=3, CONVEX=4
    body_idx: int      # Which rigid body owns this
    local_pos: vec3    # Offset from body center
    local_quat: vec4   # Rotation relative to body
    data: vec7         # SPHERE: [radius], BOX: [hx,hy,hz], CONVEX: [vert_start, vert_end]
Your Scene Mapped to Physics

Scene objects → Rigid bodies + Collision geoms:

Ground plane (STATIC):
    body_0: pos=(0,-0.25,0), mass=∞
    geom_0: type=BOX, data=[10, 0.25, 10], body=0

Red sphere:
    body_1: pos=(-2,3,0), vel=(0,0,0), mass=1.0
    geom_1: type=SPHERE, data=[0.5], body=1

Green sphere:  
    body_2: pos=(0,5,0), vel=(0,0,0), mass=1.0
    geom_2: type=SPHERE, data=[0.7], body=2

Orange box:
    body_3: pos=(-3,2,2), vel=(0,0,0), mass=1.0
    geom_3: type=BOX, data=[0.5, 0.5, 0.5], body=3
Physics Step Pipeline

Each Frame:
┌──────────────────────────────────────────────────────────────────┐
│ 1. APPLY FORCES                                                  │
│    for each body:                                                │
│        vel += gravity * dt        # (0, -9.8, 0) * dt            │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 2. PREDICT POSITIONS                                             │
│    for each body:                                                │
│        pos_predicted = pos + vel * dt                            │
│        quat_predicted = integrate_quat(quat, omega, dt)          │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 3. UPDATE GEOM WORLD TRANSFORMS                                  │
│    for each geom:                                                │
│        body = bodies[geom.body_idx]                              │
│        geom.world_pos = body.pos + rotate(body.quat, geom.local_pos)
│        geom.world_quat = body.quat * geom.local_quat             │
│        geom.aabb = compute_aabb(geom)  # for broad phase         │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 4. BROAD PHASE (BVH or spatial hash)                             │
│    Input: all geom AABBs                                         │
│    Output: candidate_pairs = [(geom_0, geom_1), (geom_1, geom_3)]│
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 5. NARROW PHASE (type dispatch)                                  │
│    for each (geom_a, geom_b) in candidate_pairs:                 │
│        if SPHERE-SPHERE: contact = sphere_sphere(a, b)           │
│        if SPHERE-BOX: contact = sphere_box(a, b)                 │
│        if BOX-BOX: contact = box_box_sat(a, b)                   │
│        if CONVEX-CONVEX: contact = gjk_epa(a, b)  ← uses support │
│                                                                  │
│    Output: contacts[] = [{point, normal, depth, body_a, body_b}] │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 6. NEWTON SOLVER (the core)                                      │
│                                                                  │
│    Build Jacobian J from all contacts                            │
│    Each contact row: [n, r_a×n, -n, -r_b×n]                      │
│                                                                  │
│    Newton iteration (3-5 iterations):                            │
│        c = J * v + b              # constraint violation         │
│        Δλ = solve(J M⁻¹ Jᵀ, -c)   # impulse update              │
│        λ = max(0, λ + Δλ)         # clamp (no pulling)          │
│        Δv = M⁻¹ * Jᵀ * Δλ         # velocity correction         │
│        v = v + Δv                                                │
│                                                                  │
│    Output: corrected velocities for all bodies                   │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 7. INTEGRATE                                                     │
│    for each body:                                                │
│        pos = pos + vel * dt                                      │
│        quat = normalize(quat + 0.5 * omega * quat * dt)          │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 8. UPDATE RENDER MESH                                            │
│    for each body:                                                │
│        for each vertex in body's mesh:                           │
│            data.vertices[i] = body.pos + rotate(body.quat, local_v)
│                                                                  │
│    (This updates your existing render data for ray tracing)      │
└──────────────────────────────────────────────────────────────────┘
How Convex Hull Collision Works
At Load Time (decompose complex mesh):


Dragon mesh (10k triangles)
           ↓
    Convex decomposition (V-HACD algorithm)
           ↓
    5 convex hulls, ~20 verts each

Store in global arrays:
    collision_verts = [hull0_v0, hull0_v1, ..., hull4_v19]  # LOCAL space
    
Create geoms:
    geom_dragon_hull0: type=CONVEX, vert_start=0, vert_end=20, body=dragon_body
    geom_dragon_hull1: type=CONVEX, vert_start=20, vert_end=45, body=dragon_body
    ...
At Runtime (GJK support query):


gjk_support(geom, direction):
    # 1. Transform direction to geom's local space
    local_dir = quat_rotate_inverse(geom.world_quat, direction)
    
    # 2. Find furthest vertex in that direction (linear scan)
    best_dot = -∞
    for i in range(geom.vert_start, geom.vert_end):
        v = collision_verts[i]           # LOCAL space, never changes!
        if dot(v, local_dir) > best_dot:
            best_dot = dot(v, local_dir)
            best_v = v
    
    # 3. Transform result back to world
    return geom.world_pos + quat_rotate(geom.world_quat, best_v)
GJK calls this support function ~10-20 times to find if shapes intersect, then EPA finds contact details.

Why Vertices "Never Move"

Collision vertex storage:
    collision_verts[i] = LOCAL position relative to geom center
    
Body moves from (0,5,0) to (0,3,0):
    - collision_verts[i] stays the same!
    - Only body.pos changes
    
When we need world position:
    world_v = body.pos + rotate(body.quat, collision_verts[i])
    
This is computed ONLY during collision queries, not stored
Memory Layout Summary

STATIC (never changes after load):
    collision_verts: [total_hull_verts, 3]     # all hulls concatenated
    geoms_info:
        type: [n_geoms]
        body_idx: [n_geoms]  
        vert_start: [n_geoms]
        vert_end: [n_geoms]
        local_pos: [n_geoms, 3]
        local_quat: [n_geoms, 4]
        data: [n_geoms, 7]

DYNAMIC (updated each frame):
    bodies_state:
        pos: [n_bodies, 3]
        quat: [n_bodies, 4]
        vel: [n_bodies, 3]
        omega: [n_bodies, 3]
    
    geoms_state:
        world_pos: [n_geoms, 3]      # computed from body transform
        world_quat: [n_geoms, 4]
        aabb_min: [n_geoms, 3]
        aabb_max: [n_geoms, 3]
This architecture separates:

What the shape is (static, in local space)
Where it is (dynamic, updated by physics)
How it looks (render mesh, updated from physics transforms)



[ ] remove object_starts for rastering, there already objects