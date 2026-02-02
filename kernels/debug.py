import taichi as ti
import kernels.data as data
from kernels.utils import intersect_aabb


@ti.func
def debug_trace_bvh(ray_o, ray_d, px: ti.i32, py: ti.i32):
    """Debug: trace BVH and return info about nodes visited"""
    closest_t = ti.f32(1e10)
    hit_node_idx = -1
    hit_depth = 0

    stack_ptr = 0
    data.traverse_stack[px, py, 0] = 0
    stack_ptr = 1
    depth = 0

    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = data.traverse_stack[px, py, stack_ptr]

        node_aabb_min = data.bvh_nodes[node_idx].aabb_min
        node_aabb_max = data.bvh_nodes[node_idx].aabb_max
        tri_count = ti.cast(data.bvh_nodes[node_idx].tri_count, ti.i32)
        left_first = ti.cast(data.bvh_nodes[node_idx].left_first, ti.i32)

        if intersect_aabb(ray_o, ray_d, node_aabb_min, node_aabb_max, closest_t):
            if tri_count > 0:
                center = (node_aabb_min + node_aabb_max) * 0.5
                t_approx = (center - ray_o).dot(ray_d.normalized())
                if t_approx > 0 and t_approx < closest_t:
                    closest_t = t_approx
                    hit_node_idx = node_idx
                    hit_depth = depth
            else:
                right_child = ti.cast(data.bvh_nodes[node_idx].right_child, ti.i32)
                data.traverse_stack[px, py, stack_ptr] = left_first
                stack_ptr += 1
                data.traverse_stack[px, py, stack_ptr] = right_child
                stack_ptr += 1
                depth += 1

    return hit_node_idx, hit_depth


@ti.kernel
def debug_render_bvh(cam_pos_x: ti.f32, cam_pos_y: ti.f32, cam_pos_z: ti.f32,
                     cam_dir_x: ti.f32, cam_dir_y: ti.f32, cam_dir_z: ti.f32,
                     cam_right_x: ti.f32, cam_right_y: ti.f32, cam_right_z: ti.f32,
                     cam_up_x: ti.f32, cam_up_y: ti.f32, cam_up_z: ti.f32):
    """Debug: render BVH leaf nodes with colors based on node index"""
    cam_pos = ti.Vector([cam_pos_x, cam_pos_y, cam_pos_z])
    cam_dir = ti.Vector([cam_dir_x, cam_dir_y, cam_dir_z])
    cam_right = ti.Vector([cam_right_x, cam_right_y, cam_right_z])
    cam_up = ti.Vector([cam_up_x, cam_up_y, cam_up_z])

    fov = 45.0
    aspect = ti.cast(data.WIDTH, ti.f32) / ti.cast(data.HEIGHT, ti.f32)
    fov_scale = ti.tan(fov * 0.5 * 3.14159 / 180.0)

    for i, j in data.pixels:
        u = (2.0 * (ti.cast(i, ti.f32) + 0.5) / ti.cast(data.WIDTH, ti.f32) - 1.0) * aspect * fov_scale
        v = (2.0 * (ti.cast(j, ti.f32) + 0.5) / ti.cast(data.HEIGHT, ti.f32) - 1.0) * fov_scale

        ray_dir = (cam_dir + u * cam_right + v * cam_up).normalized()

        node_idx, depth = debug_trace_bvh(cam_pos, ray_dir, i, j)

        if node_idx >= 0:
            r = ti.cast((node_idx * 73) % 256, ti.f32) / 255.0
            g = ti.cast((node_idx * 137) % 256, ti.f32) / 255.0
            b = ti.cast((node_idx * 199) % 256, ti.f32) / 255.0
            data.pixels[i, j] = ti.Vector([r, g, b])
        else:
            t = 0.5 * (ray_dir[1] + 1.0)
            data.pixels[i, j] = (1.0 - t) * ti.Vector([1.0, 1.0, 1.0]) + t * ti.Vector([0.5, 0.7, 1.0])


def run_debug_bvh(cam):
    """Wrapper to call debug render kernel"""
    debug_render_bvh(
        cam.pos[0], cam.pos[1], cam.pos[2],
        cam.direction[0], cam.direction[1], cam.direction[2],
        cam.right[0], cam.right[1], cam.right[2],
        cam.up[0], cam.up[1], cam.up[2]
    )
