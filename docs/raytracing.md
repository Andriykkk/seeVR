The most realistic is Path Tracing - rays bounce multiple times and handle various materials/geometry.

Realism Progression
Technique	Bounces	Features
Ray casting	0	Direct light only
Whitted ray tracing	2-3	Reflections, refractions
Path tracing	100+	Global illumination, soft shadows, caustics
Path Tracing (Most Realistic)
 
Multiple Geometry Types

@ti.func
def scene_intersect(ray_o, ray_d):
    closest_t = 1e10
    hit_normal = ti.Vector([0.0, 1.0, 0.0])
    hit_material = 0
    
    # Test spheres
    for i in range(num_spheres):
        t = ray_sphere(ray_o, ray_d, spheres[i])
        if 0 < t < closest_t:
            closest_t = t
            hit_normal = sphere_normal(ray_o + t*ray_d, spheres[i])
            hit_material = sphere_materials[i]
    
    # Test triangles (meshes)
    for i in range(num_triangles):
        t = ray_triangle(ray_o, ray_d, triangles[i])  # Möller–Trumbore
        if 0 < t < closest_t:
            closest_t = t
            hit_normal = triangle_normal(triangles[i])
            hit_material = triangle_materials[i]
    
    # Test boxes (AABB)
    for i in range(num_boxes):
        t = ray_box(ray_o, ray_d, boxes[i])  # Slab method
        if 0 < t < closest_t:
            ...
    
    # Test planes
    t = ray_plane(ray_o, ray_d, ground_plane)
    ...
    
    return closest_t < 1e10, closest_t, hit_normal, hit_material
Key Intersection Algorithms
Triangle (Möller–Trumbore):


@ti.func
def ray_triangle(o, d, v0, v1, v2):
    e1 = v1 - v0
    e2 = v2 - v0
    h = d.cross(e2)
    a = e1.dot(h)
    if abs(a) < 1e-8: return -1.0
    f = 1.0 / a
    s = o - v0
    u = f * s.dot(h)
    if u < 0 or u > 1: return -1.0
    q = s.cross(e1)
    v = f * d.dot(q)
    if v < 0 or u + v > 1: return -1.0
    return f * e2.dot(q)
Box (AABB slab method):


@ti.func
def ray_box(o, d, box_min, box_max):
    t_min = (box_min - o) / d
    t_max = (box_max - o) / d
    t1 = ti.min(t_min, t_max)
    t2 = ti.max(t_min, t_max)
    t_near = max(t1.x, t1.y, t1.z)
    t_far = min(t2.x, t2.y, t2.z)
    if t_near > t_far: return -1.0
    return t_near
Performance: BVH Acceleration
With many objects, use Bounding Volume Hierarchy:


Instead of testing 1000 triangles:
    Build tree of bounding boxes
    Test ~10-20 boxes to find candidate triangles
    Test only those triangles
    
O(N) → O(log N) per ray
Summary
For most realistic rendering:

Many samples per pixel (100-1000) for noise reduction