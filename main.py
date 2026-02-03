import taichi as ti
import numpy as np
import math
import argparse
import time
from benchmark import benchmark, is_enabled_benchmark

ti.init(arch=ti.gpu)

# Initialize shared data fields (must be after ti.init)
import kernels.data as data
data.init_scene()

# Initialize radix sort (must be after ti.init and before using LBVH)
from kernels.radix_sort import init_radix_sort
init_radix_sort()

# Import kernels (must be after init_scene)
from kernels.bvh import build_lbvh
from kernels.raytracing import run_raytrace
from kernels.debug import run_debug_bvh

# Constants from data module
WIDTH, HEIGHT = data.WIDTH, data.HEIGHT

# Path tracing settings (can be changed at runtime via GUI)
class Settings:
    def __init__(self):
        self.max_bounces = 2
        self.samples_per_pixel = 2
        self.sky_intensity = 1.0
        self.debug_bvh = False

settings = Settings()


# GPU kernels for batch data copying
@ti.kernel
def _copy_vertices_batch(verts: ti.types.ndarray(dtype=ti.f32, ndim=2), offset: ti.i32):
    for i in range(verts.shape[0]):
        data.vertices[offset + i] = ti.Vector([verts[i, 0], verts[i, 1], verts[i, 2]])


@ti.kernel
def _copy_velocities_batch(vel: ti.types.ndarray(dtype=ti.f32, ndim=1), offset: ti.i32, count: ti.i32):
    for i in range(count):
        data.velocities[offset + i] = ti.Vector([vel[0], vel[1], vel[2]])


@ti.kernel
def _copy_indices_batch(indices: ti.types.ndarray(dtype=ti.i32, ndim=1), offset: ti.i32):
    for i in range(indices.shape[0]):
        data.indices[offset + i] = indices[i]


@ti.kernel
def _copy_colors_batch(color: ti.types.ndarray(dtype=ti.f32, ndim=1), offset: ti.i32, count: ti.i32):
    for i in range(count):
        data.vertex_colors[offset + i] = ti.Vector([color[0], color[1], color[2]])


class Scene:
    """Interface for adding objects to the scene. Uses kernels/data.py for storage."""

    def __init__(self):
        self.object_starts = []  # List of (start_vert, num_verts) tuples

    @benchmark
    def add_sphere(self, center, radius, color=(1.0, 1.0, 1.0), velocity=(0, 0, 0), segments=16):
        """Add a UV sphere as triangles"""
        cx, cy, cz = center
        start_vertex = data.num_vertices[None]
        verts = []

        # Generate vertices
        for lat in range(segments + 1):
            theta = lat * math.pi / segments
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)

            for lon in range(segments + 1):
                phi = lon * 2 * math.pi / segments
                sin_phi = math.sin(phi)
                cos_phi = math.cos(phi)

                x = cx + radius * cos_phi * sin_theta
                y = cy + radius * cos_theta
                z = cz + radius * sin_phi * sin_theta
                verts.append((x, y, z))

        # Add vertices
        for v in verts:
            data.vertices[data.num_vertices[None]] = v
            data.velocities[data.num_vertices[None]] = velocity
            data.num_vertices[None] += 1

        # Generate triangles
        for lat in range(segments):
            for lon in range(segments):
                current = start_vertex + lat * (segments + 1) + lon
                next_row = current + segments + 1

                idx = data.num_triangles[None] * 3
                data.indices[idx] = current
                data.indices[idx + 1] = next_row
                data.indices[idx + 2] = current + 1
                data.num_triangles[None] += 1

                idx = data.num_triangles[None] * 3
                data.indices[idx] = current + 1
                data.indices[idx + 1] = next_row
                data.indices[idx + 2] = next_row + 1
                data.num_triangles[None] += 1

        # Set vertex colors
        for i in range(start_vertex, data.num_vertices[None]):
            data.vertex_colors[i] = color

        self.object_starts.append((start_vertex, data.num_vertices[None] - start_vertex))
        return len(self.object_starts) - 1

    @benchmark
    def add_box(self, center, size, color=(1.0, 1.0, 1.0), velocity=(0, 0, 0)):
        """Add a box as 12 triangles"""
        cx, cy, cz = center
        sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
        start_vertex = data.num_vertices[None]

        verts = [
            (cx - sx, cy - sy, cz - sz),
            (cx + sx, cy - sy, cz - sz),
            (cx + sx, cy + sy, cz - sz),
            (cx - sx, cy + sy, cz - sz),
            (cx - sx, cy - sy, cz + sz),
            (cx + sx, cy - sy, cz + sz),
            (cx + sx, cy + sy, cz + sz),
            (cx - sx, cy + sy, cz + sz),
        ]

        faces = [
            (0, 1, 2), (0, 2, 3),
            (5, 4, 7), (5, 7, 6),
            (4, 0, 3), (4, 3, 7),
            (1, 5, 6), (1, 6, 2),
            (3, 2, 6), (3, 6, 7),
            (4, 5, 1), (4, 1, 0),
        ]

        for v in verts:
            data.vertices[data.num_vertices[None]] = v
            data.velocities[data.num_vertices[None]] = velocity
            data.num_vertices[None] += 1

        for f in faces:
            idx = data.num_triangles[None] * 3
            data.indices[idx] = start_vertex + f[0]
            data.indices[idx + 1] = start_vertex + f[1]
            data.indices[idx + 2] = start_vertex + f[2]
            data.num_triangles[None] += 1

        # Set vertex colors
        for i in range(start_vertex, data.num_vertices[None]):
            data.vertex_colors[i] = color

        self.object_starts.append((start_vertex, data.num_vertices[None] - start_vertex))
        return len(self.object_starts) - 1

    @benchmark
    def add_plane(self, center, normal, size, color=(0.5, 0.5, 0.5)):
        """Add a plane as 2 triangles"""
        nx, ny, nz = normal
        if abs(nx) < 0.9:
            tangent = (1, 0, 0)
        else:
            tangent = (0, 1, 0)

        tx, ty, tz = tangent
        bx = ny * tz - nz * ty
        by = nz * tx - nx * tz
        bz = nx * ty - ny * tx
        length = math.sqrt(bx * bx + by * by + bz * bz)
        bx, by, bz = bx / length, by / length, bz / length

        t2x = ny * bz - nz * by
        t2y = nz * bx - nx * bz
        t2z = nx * by - ny * bx

        cx, cy, cz = center
        s = size / 2
        start_vertex = data.num_vertices[None]

        verts = [
            (cx - s * bx - s * t2x, cy - s * by - s * t2y, cz - s * bz - s * t2z),
            (cx + s * bx - s * t2x, cy + s * by - s * t2y, cz + s * bz - s * t2z),
            (cx + s * bx + s * t2x, cy + s * by + s * t2y, cz + s * bz + s * t2z),
            (cx - s * bx + s * t2x, cy - s * by + s * t2y, cz - s * bz + s * t2z),
        ]

        for v in verts:
            data.vertices[data.num_vertices[None]] = v
            data.velocities[data.num_vertices[None]] = (0, 0, 0)
            data.num_vertices[None] += 1

        idx = data.num_triangles[None] * 3
        data.indices[idx] = start_vertex
        data.indices[idx + 1] = start_vertex + 1
        data.indices[idx + 2] = start_vertex + 2
        data.num_triangles[None] += 1

        idx = data.num_triangles[None] * 3
        data.indices[idx] = start_vertex
        data.indices[idx + 1] = start_vertex + 2
        data.indices[idx + 2] = start_vertex + 3
        data.num_triangles[None] += 1

        # Set vertex colors
        for i in range(start_vertex, data.num_vertices[None]):
            data.vertex_colors[i] = color

    @benchmark
    def add_mesh_from_obj(self, filename, center=(0, 0, 0), size=1.0, rotation=(0, 0, 0), color=(1.0, 1.0, 1.0), velocity=(0, 0, 0)):
        """Load mesh from OBJ file, scaled to fit within given size

        Args:
            rotation: (rx, ry, rz) in degrees - applied in Y, X, Z order
        """
        raw_verts = []
        faces = []
        start_vertex = data.num_vertices[None]

        # Read raw vertices and faces
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'v':
                    raw_verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif parts[0] == 'f':
                    indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                    for i in range(1, len(indices) - 1):
                        faces.append((indices[0], indices[i], indices[i + 1]))

        if not raw_verts:
            return -1

        num_verts = len(raw_verts)
        num_faces = len(faces)

        # Convert to NumPy arrays for fast batch operations
        verts = np.array(raw_verts, dtype=np.float32)

        # Calculate bounding box (vectorized)
        min_coords = verts.min(axis=0)
        max_coords = verts.max(axis=0)
        mesh_center = (min_coords + max_coords) / 2
        max_extent = (max_coords - min_coords).max()

        # Scale to fit within size
        scale = size / max_extent if max_extent > 0 else 1.0

        # Center and scale (vectorized)
        verts = (verts - mesh_center) * scale

        # Build rotation matrix (Y * X * Z order)
        rx, ry, rz = np.radians(rotation)
        cos_x, sin_x = np.cos(rx), np.sin(rx)
        cos_y, sin_y = np.cos(ry), np.sin(ry)
        cos_z, sin_z = np.cos(rz), np.sin(rz)

        # Rotation matrices
        Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]], dtype=np.float32)
        Rx = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]], dtype=np.float32)
        Rz = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]], dtype=np.float32)
        R = Rz @ Rx @ Ry  # Combined rotation matrix

        # Apply rotation (vectorized matrix multiply) and translate
        verts = (verts @ R.T + np.array(center, dtype=np.float32)).astype(np.float32)

        # Convert faces to flat indices array
        face_indices = (np.array(faces, dtype=np.int32).flatten() + start_vertex).astype(np.int32)

        # Batch copy using GPU kernels
        _copy_vertices_batch(verts, start_vertex)
        _copy_velocities_batch(np.array(velocity, dtype=np.float32), start_vertex, num_verts)
        _copy_indices_batch(face_indices, data.num_triangles[None] * 3)
        _copy_colors_batch(np.array(color, dtype=np.float32), start_vertex, num_verts)

        # Update counts
        data.num_vertices[None] = start_vertex + num_verts
        data.num_triangles[None] = data.num_triangles[None] + num_faces

        self.object_starts.append((start_vertex, num_verts))
        print(f"Loaded {filename}: {num_verts} vertices, {num_faces} triangles")
        return len(self.object_starts) - 1

    def clear(self):
        """Clear all objects"""
        data.num_vertices[None] = 0
        data.num_triangles[None] = 0
        self.object_starts.clear()

scene = Scene()

@benchmark
def run_build_bvh():
    """Build BVH acceleration structure using GPU LBVH"""
    build_lbvh(data.num_triangles[None])
    if is_enabled_benchmark():
        ti.sync()

class Camera:
    """First-person camera with WASD + mouse look"""

    def __init__(self, position=(0, 5, 15), yaw=-90.0, pitch=-15.0):
        self.pos = list(position)
        self.yaw = yaw
        self.pitch = pitch
        self.speed = 10.0
        self.sensitivity = 0.5
        self._last_mouse = None

        # Computed vectors
        self.direction = [0, 0, -1]
        self.right = [1, 0, 0]
        self.up = [0, 1, 0]
        self._update_vectors()

    def _update_vectors(self):
        rad_yaw = math.radians(self.yaw)
        rad_pitch = math.radians(self.pitch)

        # Direction
        self.direction = [
            math.cos(rad_pitch) * math.cos(rad_yaw),
            math.sin(rad_pitch),
            math.cos(rad_pitch) * math.sin(rad_yaw)
        ]
        length = math.sqrt(sum(x * x for x in self.direction))
        self.direction = [x / length for x in self.direction]

        # Right (cross product with world up)
        world_up = [0, 1, 0]
        self.right = [
            self.direction[1] * world_up[2] - self.direction[2] * world_up[1],
            self.direction[2] * world_up[0] - self.direction[0] * world_up[2],
            self.direction[0] * world_up[1] - self.direction[1] * world_up[0]
        ]
        length = math.sqrt(sum(x * x for x in self.right))
        self.right = [x / length for x in self.right]

        # Up (cross product of right and direction)
        self.up = [
            self.right[1] * self.direction[2] - self.right[2] * self.direction[1],
            self.right[2] * self.direction[0] - self.right[0] * self.direction[2],
            self.right[0] * self.direction[1] - self.right[1] * self.direction[0]
        ]

    @benchmark
    def handle_input(self, window, dt):
        # Keyboard movement
        if window.is_pressed('w'):
            rad_yaw = math.radians(self.yaw)
            self.pos[0] += math.cos(rad_yaw) * self.speed * dt
            self.pos[2] += math.sin(rad_yaw) * self.speed * dt
        if window.is_pressed('s'):
            rad_yaw = math.radians(self.yaw)
            self.pos[0] -= math.cos(rad_yaw) * self.speed * dt
            self.pos[2] -= math.sin(rad_yaw) * self.speed * dt
        if window.is_pressed('a'):
            rad_yaw = math.radians(self.yaw - 90)
            self.pos[0] += math.cos(rad_yaw) * self.speed * dt
            self.pos[2] += math.sin(rad_yaw) * self.speed * dt
        if window.is_pressed('d'):
            rad_yaw = math.radians(self.yaw + 90)
            self.pos[0] += math.cos(rad_yaw) * self.speed * dt
            self.pos[2] += math.sin(rad_yaw) * self.speed * dt
        if window.is_pressed(ti.ui.SPACE):
            self.pos[1] += self.speed * dt
        if window.is_pressed(ti.ui.SHIFT):
            self.pos[1] -= self.speed * dt

        # Mouse look (right-click drag)
        curr_mouse = window.get_cursor_pos()
        if window.is_pressed(ti.ui.RMB) and self._last_mouse is not None:
            dx = (curr_mouse[0] - self._last_mouse[0]) * 200
            dy = (curr_mouse[1] - self._last_mouse[1]) * 200
            self.yaw += dx * self.sensitivity
            self.pitch = max(-89, min(89, self.pitch + dy * self.sensitivity))
        self._last_mouse = curr_mouse

        self._update_vectors()

    def apply_to_ti_camera(self, ti_camera):
        """Apply position/rotation to Taichi UI camera for rasterization"""
        ti_camera.position(self.pos[0], self.pos[1], self.pos[2])
        ti_camera.lookat(
            self.pos[0] + self.direction[0],
            self.pos[1] + self.direction[1],
            self.pos[2] + self.direction[2]
        )


@ti.kernel
def update_physics(dt: ti.f32):
    for i in range(data.num_vertices[None]):
        data.vertices[i] += data.velocities[i] * dt

# Benchmarked wrappers for kernels (ti.sync() needed for accurate GPU timing)
@benchmark
def run_physics(dt):
    update_physics(dt)
    if is_enabled_benchmark():
        ti.sync()

@benchmark
def run_rasterize(ti_scene, ti_camera, canvas, cam):
    cam.apply_to_ti_camera(ti_camera)
    ti_scene.set_camera(ti_camera)
    ti_scene.ambient_light((0.2, 0.2, 0.2))
    ti_scene.point_light(pos=(10, 10, 10), color=(1, 1, 1))
    ti_scene.mesh(
        data.vertices,
        indices=data.indices,
        per_vertex_color=data.vertex_colors,
        two_sided=True
    )
    canvas.scene(ti_scene)
    if is_enabled_benchmark():
        ti.sync()


def create_demo_scene():
    """Create demo scene - dragon inside a room"""
    room_size = 10
    wall_thickness = 0.5
    half = room_size / 2

    # Floor
    scene.add_box(
        center=(0, -wall_thickness / 2, 0),
        size=(room_size, wall_thickness, room_size),
        color=(0.4, 0.4, 0.4)
    )

    # Ceiling
    scene.add_box(
        center=(0, room_size - wall_thickness / 2, 0),
        size=(room_size, wall_thickness, room_size),
        color=(0.5, 0.5, 0.5)
    )

    # Back wall (far from camera)
    scene.add_box(
        center=(0, half, -half - wall_thickness / 2),
        size=(room_size, room_size, wall_thickness),
        color=(0.6, 0.6, 0.7)
    )

    # Left wall
    scene.add_box(
        center=(-half - wall_thickness / 2, half, 0),
        size=(wall_thickness, room_size, room_size),
        color=(0.7, 0.5, 0.5)
    )

    # Right wall
    scene.add_box(
        center=(half + wall_thickness / 2, half, 0),
        size=(wall_thickness, room_size, room_size),
        color=(0.5, 0.7, 0.5)
    )

    # Front wall is OPEN (no wall) so we can see inside

    # Dragon inside the room
    scene.add_mesh_from_obj(
        "./models/dragon_small.obj",
        center=(0, 2.5, 0),
        size=8.0,
        color=(0.8, 0.3, 0.3),
        rotation=(0, 180, 0)
    )


@benchmark
def render_frame(camera, frame, window, canvas, ti_scene, ti_camera, use_raytracing):
    dt = 1.0 / 60.0
    camera.handle_input(window, dt)
    run_physics(dt)

    rays = 0
    if use_raytracing:
        if settings.debug_bvh:
            run_debug_bvh(camera)
            rays = WIDTH * HEIGHT
        else:
            run_build_bvh()
            run_raytrace(camera, frame, settings)
            rays = WIDTH * HEIGHT * settings.samples_per_pixel * settings.max_bounces
        canvas.set_image(data.pixels)
    else:
        run_rasterize(ti_scene, ti_camera, canvas, camera)

    if is_enabled_benchmark():
        ti.sync()

    return rays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raytrace', action='store_true', help='Use ray tracing')
    args = parser.parse_args()

    use_raytracing = args.raytrace

    create_demo_scene()

    print(f"Scene: {data.num_vertices[None]} vertices, {data.num_triangles[None]} triangles")
    print(f"Rendering: {'Ray Tracing' if use_raytracing else 'Rasterization'}")

    window = ti.ui.Window("Taichi Scene", (WIDTH, HEIGHT), vsync=False)
    canvas = window.get_canvas()
    ti_scene = window.get_scene()
    ti_camera = ti.ui.Camera()

    camera = Camera(position=(0, 5, 15), yaw=-90, pitch=-15)
    frame = 0
    last_time = time.perf_counter()
    fps_smooth = 0.0
    mrays_smooth = 0.0
    ema_alpha = 0.1  # Smoothing factor (lower = smoother, higher = more responsive)

    while window.running:
        rays = render_frame(camera, frame, window, canvas, ti_scene, ti_camera, use_raytracing)

        now = time.perf_counter()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps = 1.0 / dt
            mrays = rays * fps / 1e6
            # Exponential moving average for smooth display
            fps_smooth = ema_alpha * fps + (1 - ema_alpha) * fps_smooth
            mrays_smooth = ema_alpha * mrays + (1 - ema_alpha) * mrays_smooth

        # GUI panel
        gui = window.get_gui()
        gui.begin("Settings", 0.02, 0.02, 0.3, 0.3)
        gui.text(f"FPS: {fps_smooth:.1f}  Mrays/s: {mrays_smooth:.1f}")
        settings.max_bounces = gui.slider_int("Bounces", settings.max_bounces, 1, 16)
        settings.samples_per_pixel = gui.slider_int("Samples", settings.samples_per_pixel, 1, 64)
        settings.sky_intensity = gui.slider_float("Sky", settings.sky_intensity, 0.0, 3.0)
        settings.debug_bvh = gui.checkbox("Debug BVH", settings.debug_bvh)
        gui.end()

        frame += 1
        window.show()


if __name__ == '__main__':
    main()
