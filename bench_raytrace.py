import taichi as ti
import time
import math

ti.init(arch=ti.gpu)

MAX_TRIANGLES = 500000
MAX_VERTICES = 500000
MAX_BVH_NODES = MAX_TRIANGLES * 2
WIDTH, HEIGHT = 800, 600

BVHNode = ti.types.struct(
    aabb_min=ti.types.vector(3, ti.f32),
    aabb_max=ti.types.vector(3, ti.f32),
    left_first=ti.u32,
    tri_count=ti.u32,
)

vertices = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)
indices = ti.field(dtype=ti.i32, shape=MAX_TRIANGLES * 3)
num_triangles = ti.field(dtype=ti.i32, shape=())
bvh_nodes = BVHNode.field(shape=MAX_BVH_NODES)
bvh_prim_indices = ti.field(dtype=ti.i32, shape=MAX_TRIANGLES)
tri_centroids = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIANGLES)
hit_count = ti.field(dtype=ti.i32, shape=())


@ti.func
def ray_triangle_intersect(ray_o, ray_d, v0, v1, v2):
    e1 = v1 - v0
    e2 = v2 - v0
    h = ray_d.cross(e2)
    a = e1.dot(h)
    t = -1.0
    if ti.abs(a) > 1e-8:
        f = 1.0 / a
        s = ray_o - v0
        u = f * s.dot(h)
        if 0.0 <= u <= 1.0:
            q = s.cross(e1)
            v = f * ray_d.dot(q)
            if v >= 0.0 and u + v <= 1.0:
                t = f * e2.dot(q)
                if t < 0.001:
                    t = -1.0
    return t


@ti.func
def intersect_aabb(ray_o, ray_d, bmin, bmax, closest_t):
    inv_d = 1.0 / ray_d
    tx1 = (bmin[0] - ray_o[0]) * inv_d[0]
    tx2 = (bmax[0] - ray_o[0]) * inv_d[0]
    tmin = ti.min(tx1, tx2)
    tmax = ti.max(tx1, tx2)
    ty1 = (bmin[1] - ray_o[1]) * inv_d[1]
    ty2 = (bmax[1] - ray_o[1]) * inv_d[1]
    tmin = ti.max(tmin, ti.min(ty1, ty2))
    tmax = ti.min(tmax, ti.max(ty1, ty2))
    tz1 = (bmin[2] - ray_o[2]) * inv_d[2]
    tz2 = (bmax[2] - ray_o[2]) * inv_d[2]
    tmin = ti.max(tmin, ti.min(tz1, tz2))
    tmax = ti.min(tmax, ti.max(tz1, tz2))
    return tmax >= tmin and tmin < closest_t and tmax > 0.0


@ti.kernel
def bench_brute(cam_x: ti.f32, cam_y: ti.f32, cam_z: ti.f32,
                dir_x: ti.f32, dir_y: ti.f32, dir_z: ti.f32):
    cam_pos = ti.Vector([cam_x, cam_y, cam_z])
    cam_dir = ti.Vector([dir_x, dir_y, dir_z])
    cam_right = ti.Vector([1.0, 0.0, 0.0])
    cam_up = ti.Vector([0.0, 1.0, 0.0])

    fov_scale = 0.414  # tan(45/2)
    aspect = float(WIDTH) / float(HEIGHT)

    for i, j in ti.ndrange(WIDTH, HEIGHT):
        u = (2.0 * (float(i) + 0.5) / float(WIDTH) - 1.0) * aspect * fov_scale
        v = (2.0 * (float(j) + 0.5) / float(HEIGHT) - 1.0) * fov_scale
        ray_d = (cam_dir + u * cam_right + v * cam_up).normalized()

        closest_t = ti.f32(1e10)
        for k in range(num_triangles[None]):
            idx = k * 3
            v0 = vertices[indices[idx]]
            v1 = vertices[indices[idx + 1]]
            v2 = vertices[indices[idx + 2]]
            t = ray_triangle_intersect(cam_pos, ray_d, v0, v1, v2)
            if 0.001 < t < closest_t:
                closest_t = t
                ti.atomic_add(hit_count[None], 1)


@ti.kernel
def bench_bvh(cam_x: ti.f32, cam_y: ti.f32, cam_z: ti.f32,
              dir_x: ti.f32, dir_y: ti.f32, dir_z: ti.f32):
    cam_pos = ti.Vector([cam_x, cam_y, cam_z])
    cam_dir = ti.Vector([dir_x, dir_y, dir_z])
    cam_right = ti.Vector([1.0, 0.0, 0.0])
    cam_up = ti.Vector([0.0, 1.0, 0.0])

    fov_scale = 0.414
    aspect = float(WIDTH) / float(HEIGHT)

    for i, j in ti.ndrange(WIDTH, HEIGHT):
        u = (2.0 * (float(i) + 0.5) / float(WIDTH) - 1.0) * aspect * fov_scale
        v = (2.0 * (float(j) + 0.5) / float(HEIGHT) - 1.0) * fov_scale
        ray_d = (cam_dir + u * cam_right + v * cam_up).normalized()

        closest_t = ti.f32(1e10)
        stack = ti.Matrix([[0] * 32], dt=ti.i32)
        stack[0, 0] = 0
        stack_ptr = 1

        for _iter in range(64):
            if stack_ptr <= 0:
                break
            stack_ptr -= 1
            node_idx = stack[0, stack_ptr]
            node = bvh_nodes[node_idx]

            if not intersect_aabb(cam_pos, ray_d, node.aabb_min, node.aabb_max, closest_t):
                continue

            tri_count = ti.cast(node.tri_count, ti.i32)
            left_first = ti.cast(node.left_first, ti.i32)

            if tri_count > 0:
                for k in range(tri_count):
                    tri_idx = bvh_prim_indices[left_first + k]
                    idx = tri_idx * 3
                    v0 = vertices[indices[idx]]
                    v1 = vertices[indices[idx + 1]]
                    v2 = vertices[indices[idx + 2]]
                    t = ray_triangle_intersect(cam_pos, ray_d, v0, v1, v2)
                    if 0.001 < t < closest_t:
                        closest_t = t
                        ti.atomic_add(hit_count[None], 1)
            else:
                stack[0, stack_ptr] = left_first
                stack_ptr += 1
                stack[0, stack_ptr] = left_first + 1
                stack_ptr += 1


def load_obj(filename):
    import numpy as np
    raw_verts = []
    faces = []

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                raw_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                idxs = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])

    verts = np.array(raw_verts, dtype=np.float32)
    # Center and scale
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    verts -= center
    extent = verts.max(axis=0) - verts.min(axis=0)
    scale = 5.0 / extent.max()
    verts *= scale

    return verts, faces


def build_bvh_cpu(n, verts_np):
    import numpy as np

    centroids = tri_centroids.to_numpy()[:n]
    prim_indices = list(range(n))

    bvh_nodes[0].left_first = 0
    bvh_nodes[0].tri_count = n

    stack = [(0, 0, n)]
    nodes_used = 1

    while stack:
        node_idx, first, count = stack.pop()

        aabb_min = np.array([1e30, 1e30, 1e30])
        aabb_max = np.array([-1e30, -1e30, -1e30])
        for i in range(first, first + count):
            tri_idx = prim_indices[i]
            idx = tri_idx * 3
            for vi in range(3):
                v = vertices[indices[idx + vi]]
                aabb_min = np.minimum(aabb_min, [v[0], v[1], v[2]])
                aabb_max = np.maximum(aabb_max, [v[0], v[1], v[2]])

        bvh_nodes[node_idx].aabb_min = aabb_min.tolist()
        bvh_nodes[node_idx].aabb_max = aabb_max.tolist()

        if count <= 10:
            bvh_nodes[node_idx].left_first = first
            bvh_nodes[node_idx].tri_count = count
            for i in range(count):
                bvh_prim_indices[first + i] = prim_indices[first + i]
            continue

        extent = aabb_max - aabb_min
        axis = int(np.argmax(extent))
        split_pos = aabb_min[axis] + extent[axis] * 0.5

        i, j = first, first + count - 1
        while i <= j:
            if centroids[prim_indices[i]][axis] < split_pos:
                i += 1
            else:
                prim_indices[i], prim_indices[j] = prim_indices[j], prim_indices[i]
                j -= 1

        left_count = i - first
        if left_count == 0 or left_count == count:
            bvh_nodes[node_idx].left_first = first
            bvh_nodes[node_idx].tri_count = count
            for i in range(count):
                bvh_prim_indices[first + i] = prim_indices[first + i]
            continue

        left_idx = nodes_used
        nodes_used += 2

        bvh_nodes[node_idx].left_first = left_idx
        bvh_nodes[node_idx].tri_count = 0

        stack.append((left_idx + 1, i, count - left_count))
        stack.append((left_idx, first, left_count))

    print(f"BVH built: {nodes_used} nodes")


@ti.kernel
def init_centroids(n: ti.i32):
    for i in range(n):
        idx = i * 3
        v0 = vertices[indices[idx]]
        v1 = vertices[indices[idx + 1]]
        v2 = vertices[indices[idx + 2]]
        tri_centroids[i] = (v0 + v1 + v2) / 3.0
        bvh_prim_indices[i] = i


def main():
    print("Loading dragon...")
    verts, faces = load_obj("./models/dragon_smallest.obj")

    n_verts = len(verts)
    n_tris = len(faces)
    print(f"Triangles: {n_tris}, Vertices: {n_verts}")

    # Upload to GPU
    for i, v in enumerate(verts):
        vertices[i] = v
    for i, f in enumerate(faces):
        indices[i * 3] = f[0]
        indices[i * 3 + 1] = f[1]
        indices[i * 3 + 2] = f[2]
    num_triangles[None] = n_tris

    # Build BVH
    init_centroids(n_tris)
    ti.sync()
    build_bvh_cpu(n_tris, verts)

    # Camera setup
    cam_x, cam_y, cam_z = 0.0, 0.0, 10.0
    dir_x, dir_y, dir_z = 0.0, 0.0, -1.0

    rays_per_frame = WIDTH * HEIGHT

    # Warmup
    print("\nWarmup...")
    bench_brute(cam_x, cam_y, cam_z, dir_x, dir_y, dir_z)
    bench_bvh(cam_x, cam_y, cam_z, dir_x, dir_y, dir_z)
    ti.sync()

    # Benchmark brute force
    print("\n--- Brute Force ---")
    hit_count[None] = 0
    n_frames = 10
    t0 = time.perf_counter()
    for _ in range(n_frames):
        bench_brute(cam_x, cam_y, cam_z, dir_x, dir_y, dir_z)
    ti.sync()
    t1 = time.perf_counter()
    dt = t1 - t0
    fps = n_frames / dt
    mrays = (rays_per_frame * n_frames) / dt / 1e6
    print(f"Time: {dt:.3f}s for {n_frames} frames")
    print(f"FPS: {fps:.1f}")
    print(f"Mrays/s: {mrays:.2f}")

    # Benchmark BVH
    print("\n--- BVH ---")
    hit_count[None] = 0
    t0 = time.perf_counter()
    for _ in range(n_frames):
        bench_bvh(cam_x, cam_y, cam_z, dir_x, dir_y, dir_z)
    ti.sync()
    t1 = time.perf_counter()
    dt = t1 - t0
    fps = n_frames / dt
    mrays = (rays_per_frame * n_frames) / dt / 1e6
    print(f"Time: {dt:.3f}s for {n_frames} frames")
    print(f"FPS: {fps:.1f}")
    print(f"Mrays/s: {mrays:.2f}")


if __name__ == '__main__':
    main()
