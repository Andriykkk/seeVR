import taichi as ti
import kernels.data as data
from kernels.utils import intersect_aabb, ray_triangle_intersect, get_triangle_normal
from benchmark import benchmark, is_enabled_benchmark

@ti.func
def rand_pcg(seed: ti.u32) -> ti.u32:
    state = seed * ti.u32(747796405) + ti.u32(2891336453)
    word = ((state >> ((state >> 28) + 4)) ^ state) * ti.u32(277803737)
    return (word >> 22) ^ word


@ti.func
def rand_float(seed: ti.u32) -> ti.f32:
    return ti.cast(rand_pcg(seed), ti.f32) / 4294967295.0


@ti.func
def sample_hemisphere_cosine(normal, seed1: ti.u32, seed2: ti.u32):
    r1 = rand_float(seed1)
    r2 = rand_float(seed2)

    phi = 2.0 * 3.14159 * r1
    cos_theta = ti.sqrt(r2)
    sin_theta = ti.sqrt(1.0 - r2)

    up = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(normal[1]) > 0.99:
        up = ti.Vector([1.0, 0.0, 0.0])
    tangent = up.cross(normal).normalized()
    bitangent = normal.cross(tangent)

    dir_local = ti.Vector([
        sin_theta * ti.cos(phi),
        cos_theta,
        sin_theta * ti.sin(phi)
    ])
    return (tangent * dir_local[0] + normal * dir_local[1] + bitangent * dir_local[2]).normalized()


@ti.func
def sky_color(ray_d, intensity: ti.f32):
    t = 0.5 * (ray_d[1] + 1.0)
    base = (1.0 - t) * ti.Vector([1.0, 1.0, 1.0]) + t * ti.Vector([0.5, 0.7, 1.0])
    return base * intensity


@ti.func
def trace_bvh(ray_o, ray_d, px: ti.i32, py: ti.i32):
    """Trace ray through BVH with local stack"""
    closest_t = ti.f32(1e10)
    hit_normal = ti.Vector([0.0, 1.0, 0.0])
    hit_color = ti.Vector([0.0, 0.0, 0.0])
    hit = False

    stack = ti.Matrix([[0] * 32], dt=ti.i32)
    stack[0, 0] = 0
    stack_ptr = 1

    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = stack[0, stack_ptr]
        node = data.bvh_nodes[node_idx]

        if not intersect_aabb(ray_o, ray_d, node.aabb_min, node.aabb_max, closest_t):
            continue

        tri_count = ti.cast(node.tri_count, ti.i32)
        left_first = ti.cast(node.left_first, ti.i32)

        if tri_count > 0:
            for i in range(tri_count):
                tri_idx = data.bvh_prim_indices[left_first + i]
                idx = tri_idx * 3
                v0 = data.vertices[data.indices[idx]]
                v1 = data.vertices[data.indices[idx + 1]]
                v2 = data.vertices[data.indices[idx + 2]]

                t = ray_triangle_intersect(ray_o, ray_d, v0, v1, v2)
                if 0.001 < t < closest_t:
                    closest_t = t
                    hit_normal = get_triangle_normal(v0, v1, v2)
                    if hit_normal.dot(ray_d) > 0:
                        hit_normal = -hit_normal
                    hit_color = data.vertex_colors[data.indices[idx]]
                    hit = True
        else:
            right_child = ti.cast(node.right_child, ti.i32)
            stack[0, stack_ptr] = left_first
            stack_ptr += 1
            stack[0, stack_ptr] = right_child
            stack_ptr += 1

    return hit, closest_t, hit_normal, hit_color


@ti.kernel
def raytrace(cam_pos_x: ti.f32, cam_pos_y: ti.f32, cam_pos_z: ti.f32,
             cam_dir_x: ti.f32, cam_dir_y: ti.f32, cam_dir_z: ti.f32,
             cam_right_x: ti.f32, cam_right_y: ti.f32, cam_right_z: ti.f32,
             cam_up_x: ti.f32, cam_up_y: ti.f32, cam_up_z: ti.f32,
             frame: ti.i32, max_bounces: ti.i32, samples_per_pixel: ti.i32,
             sky_intensity: ti.f32):
    cam_pos = ti.Vector([cam_pos_x, cam_pos_y, cam_pos_z])
    cam_dir = ti.Vector([cam_dir_x, cam_dir_y, cam_dir_z])
    cam_right = ti.Vector([cam_right_x, cam_right_y, cam_right_z])
    cam_up = ti.Vector([cam_up_x, cam_up_y, cam_up_z])

    fov = 45.0
    aspect = ti.cast(data.WIDTH, ti.f32) / ti.cast(data.HEIGHT, ti.f32)
    fov_scale = ti.tan(fov * 0.5 * 3.14159 / 180.0)

    for i, j in data.pixels:
        pixel_color = ti.Vector([0.0, 0.0, 0.0])

        for sample in range(samples_per_pixel):
            seed = ti.cast(i + j * data.WIDTH + frame * data.WIDTH * data.HEIGHT + sample * 12345, ti.u32)

            jitter_x = rand_float(seed) - 0.5
            jitter_y = rand_float(seed + 1) - 0.5

            u = (2.0 * (ti.cast(i, ti.f32) + 0.5 + jitter_x) / ti.cast(data.WIDTH, ti.f32) - 1.0) * aspect * fov_scale
            v = (2.0 * (ti.cast(j, ti.f32) + 0.5 + jitter_y) / ti.cast(data.HEIGHT, ti.f32) - 1.0) * fov_scale

            ray_pos = cam_pos
            ray_dir = (cam_dir + u * cam_right + v * cam_up).normalized()

            throughput = ti.Vector([1.0, 1.0, 1.0])
            color = ti.Vector([0.0, 0.0, 0.0])

            for bounce in range(max_bounces):
                hit, t, normal, albedo = trace_bvh(ray_pos, ray_dir, i, j)

                if not hit:
                    color += throughput * sky_color(ray_dir, sky_intensity)
                    break

                hit_point = ray_pos + ray_dir * t
                throughput *= albedo

                if bounce > 1:
                    p = ti.max(throughput[0], ti.max(throughput[1], throughput[2]))
                    if rand_float(seed + ti.cast(bounce, ti.u32) * 7) > p:
                        break
                    throughput /= p

                seed = rand_pcg(seed)
                ray_dir = sample_hemisphere_cosine(normal, seed, seed + 1)
                ray_pos = hit_point + normal * 0.001

            pixel_color += color

        pixel_color /= ti.cast(samples_per_pixel, ti.f32)
        data.pixels[i, j] = pixel_color

@benchmark
def run_raytrace(cam, frame, settings):
    """Wrapper to call raytrace kernel"""
    raytrace(
        cam.pos[0], cam.pos[1], cam.pos[2],
        cam.direction[0], cam.direction[1], cam.direction[2],
        cam.right[0], cam.right[1], cam.right[2],
        cam.up[0], cam.up[1], cam.up[2],
        frame,
        settings.max_bounces,
        settings.samples_per_pixel,
        settings.sky_intensity
    )

    if is_enabled_benchmark():
        ti.sync()