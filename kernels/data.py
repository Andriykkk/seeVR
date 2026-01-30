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

# BVH Node structure
BVHNode = ti.types.struct(
    aabb_min=ti.types.vector(3, ti.f32),
    aabb_max=ti.types.vector(3, ti.f32),
    left_first=ti.u32,
    tri_count=ti.u32,
)

# Global scene fields - initialized by init_scene()
vertices = None
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


def init_scene():
    """Initialize all Taichi fields. Call after ti.init()"""
    global vertices, indices, vertex_colors, num_vertices, num_triangles
    global velocities, pixels, bvh_nodes, bvh_prim_indices, tri_centroids
    global num_bvh_nodes, traverse_stack, bvh_build_stack

    # Geometry
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)
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

    # Initialize counts
    num_vertices[None] = 0
    num_triangles[None] = 0
