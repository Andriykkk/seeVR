import taichi as ti
import random
import math
import argparse
from benchmark import benchmark, enable_benchmark, is_enabled_benchmark

ti.init(arch=ti.gpu)

# Constants
MAX_TRIANGLES = 500000
MAX_VERTICES = 500000
WIDTH, HEIGHT = 800, 600


class Scene:
    """Manages all scene objects as triangle meshes"""

    def __init__(self):
        # All geometry stored as flat arrays (works for both rasterization and ray tracing)
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)
        self.indices = ti.field(dtype=ti.i32, shape=MAX_TRIANGLES * 3)  # Every 3 = one triangle
        self.vertex_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)
        self.num_vertices = ti.field(dtype=ti.i32, shape=())
        self.num_triangles = ti.field(dtype=ti.i32, shape=())

        # For animation - store velocity per vertex
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)

        # Object tracking for physics (start_vertex, num_vertices, center)
        self.object_starts = []  # List of (start_vert, num_verts) tuples

        # Pixel buffer for ray tracing
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

        # Initialize counts
        self.num_vertices[None] = 0
        self.num_triangles[None] = 0

        self._vertex_count = 0
        self._triangle_count = 0

    @benchmark
    def add_sphere(self, center, radius, color=(1.0, 1.0, 1.0), velocity=(0, 0, 0), segments=16):
        """Add a UV sphere as triangles"""
        cx, cy, cz = center
        start_vertex = self._vertex_count
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
            self.vertices[self._vertex_count] = v
            self.velocities[self._vertex_count] = velocity
            self._vertex_count += 1

        # Generate triangles
        for lat in range(segments):
            for lon in range(segments):
                current = start_vertex + lat * (segments + 1) + lon
                next_row = current + segments + 1

                idx = self._triangle_count * 3
                self.indices[idx] = current
                self.indices[idx + 1] = next_row
                self.indices[idx + 2] = current + 1
                self._triangle_count += 1

                idx = self._triangle_count * 3
                self.indices[idx] = current + 1
                self.indices[idx + 1] = next_row
                self.indices[idx + 2] = next_row + 1
                self._triangle_count += 1

        # Set vertex colors
        for i in range(start_vertex, self._vertex_count):
            self.vertex_colors[i] = color

        self.num_vertices[None] = self._vertex_count
        self.num_triangles[None] = self._triangle_count

        self.object_starts.append((start_vertex, self._vertex_count - start_vertex))
        return len(self.object_starts) - 1

    @benchmark
    def add_box(self, center, size, color=(1.0, 1.0, 1.0), velocity=(0, 0, 0)):
        """Add a box as 12 triangles"""
        cx, cy, cz = center
        sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
        start_vertex = self._vertex_count

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
            self.vertices[self._vertex_count] = v
            self.velocities[self._vertex_count] = velocity
            self._vertex_count += 1

        for f in faces:
            idx = self._triangle_count * 3
            self.indices[idx] = start_vertex + f[0]
            self.indices[idx + 1] = start_vertex + f[1]
            self.indices[idx + 2] = start_vertex + f[2]
            self._triangle_count += 1

        # Set vertex colors
        for i in range(start_vertex, self._vertex_count):
            self.vertex_colors[i] = color

        self.num_vertices[None] = self._vertex_count
        self.num_triangles[None] = self._triangle_count

        self.object_starts.append((start_vertex, self._vertex_count - start_vertex))
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
        start_vertex = self._vertex_count

        verts = [
            (cx - s * bx - s * t2x, cy - s * by - s * t2y, cz - s * bz - s * t2z),
            (cx + s * bx - s * t2x, cy + s * by - s * t2y, cz + s * bz - s * t2z),
            (cx + s * bx + s * t2x, cy + s * by + s * t2y, cz + s * bz + s * t2z),
            (cx - s * bx + s * t2x, cy - s * by + s * t2y, cz - s * bz + s * t2z),
        ]

        for v in verts:
            self.vertices[self._vertex_count] = v
            self.velocities[self._vertex_count] = (0, 0, 0)
            self._vertex_count += 1

        idx = self._triangle_count * 3
        self.indices[idx] = start_vertex
        self.indices[idx + 1] = start_vertex + 1
        self.indices[idx + 2] = start_vertex + 2
        self._triangle_count += 1

        idx = self._triangle_count * 3
        self.indices[idx] = start_vertex
        self.indices[idx + 1] = start_vertex + 2
        self.indices[idx + 2] = start_vertex + 3
        self._triangle_count += 1

        # Set vertex colors
        for i in range(start_vertex, self._vertex_count):
            self.vertex_colors[i] = color

        self.num_vertices[None] = self._vertex_count
        self.num_triangles[None] = self._triangle_count

    @benchmark
    def add_mesh_from_obj(self, filename, center=(0, 0, 0), size=1.0, rotation=(0, 0, 0), color=(1.0, 1.0, 1.0), velocity=(0, 0, 0)):
        """Load mesh from OBJ file, scaled to fit within given size

        Args:
            rotation: (rx, ry, rz) in degrees - applied in Y, X, Z order
        """
        raw_verts = []
        faces = []
        start_vertex = self._vertex_count

        # Read raw vertices and faces
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'v':
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    raw_verts.append((x, y, z))
                elif parts[0] == 'f':
                    indices = []
                    for p in parts[1:]:
                        idx = int(p.split('/')[0]) - 1
                        indices.append(idx)
                    for i in range(1, len(indices) - 1):
                        faces.append((indices[0], indices[i], indices[i + 1]))

        if not raw_verts:
            return -1

        # Calculate bounding box
        min_x = min(v[0] for v in raw_verts)
        max_x = max(v[0] for v in raw_verts)
        min_y = min(v[1] for v in raw_verts)
        max_y = max(v[1] for v in raw_verts)
        min_z = min(v[2] for v in raw_verts)
        max_z = max(v[2] for v in raw_verts)

        # Mesh center and dimensions
        mesh_center = ((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
        max_extent = max(max_x - min_x, max_y - min_y, max_z - min_z)

        # Scale to fit within size
        scale = size / max_extent if max_extent > 0 else 1.0

        # Precompute rotation (Y * X * Z order)
        rx, ry, rz = math.radians(rotation[0]), math.radians(rotation[1]), math.radians(rotation[2])
        cos_x, sin_x = math.cos(rx), math.sin(rx)
        cos_y, sin_y = math.cos(ry), math.sin(ry)
        cos_z, sin_z = math.cos(rz), math.sin(rz)

        # Transform: center at origin, scale, rotate, move to target center
        for x, y, z in raw_verts:
            # Center and scale
            vx = (x - mesh_center[0]) * scale
            vy = (y - mesh_center[1]) * scale
            vz = (z - mesh_center[2]) * scale

            # Rotate Y
            tmp_x = cos_y * vx + sin_y * vz
            tmp_z = -sin_y * vx + cos_y * vz
            vx, vz = tmp_x, tmp_z

            # Rotate X
            tmp_y = cos_x * vy - sin_x * vz
            tmp_z = sin_x * vy + cos_x * vz
            vy, vz = tmp_y, tmp_z

            # Rotate Z
            tmp_x = cos_z * vx - sin_z * vy
            tmp_y = sin_z * vx + cos_z * vy
            vx, vy = tmp_x, tmp_y

            # Translate to center
            vx += center[0]
            vy += center[1]
            vz += center[2]

            self.vertices[self._vertex_count] = (vx, vy, vz)
            self.velocities[self._vertex_count] = velocity
            self._vertex_count += 1

        for f in faces:
            idx = self._triangle_count * 3
            self.indices[idx] = start_vertex + f[0]
            self.indices[idx + 1] = start_vertex + f[1]
            self.indices[idx + 2] = start_vertex + f[2]
            self._triangle_count += 1

        # Set vertex colors
        for i in range(start_vertex, self._vertex_count):
            self.vertex_colors[i] = color

        self.num_vertices[None] = self._vertex_count
        self.num_triangles[None] = self._triangle_count

        self.object_starts.append((start_vertex, len(raw_verts)))
        print(f"Loaded {filename}: {len(raw_verts)} vertices, {len(faces)} triangles")
        return len(self.object_starts) - 1

    def clear(self):
        """Clear all objects"""
        self._vertex_count = 0
        self._triangle_count = 0
        self.num_vertices[None] = 0
        self.num_triangles[None] = 0
        self.object_starts.clear()

scene = Scene()


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
    for i in range(scene.num_vertices[None]):
        scene.vertices[i] += scene.velocities[i] * dt


@ti.func
def ray_triangle_intersect(ray_o, ray_d, v0, v1, v2):
    """Möller–Trumbore algorithm"""
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
def get_triangle_normal(v0, v1, v2):
    e1 = v1 - v0
    e2 = v2 - v0
    return e1.cross(e2).normalized()


@ti.kernel
def raytrace(cam_pos_x: ti.f32, cam_pos_y: ti.f32, cam_pos_z: ti.f32,
             cam_dir_x: ti.f32, cam_dir_y: ti.f32, cam_dir_z: ti.f32,
             cam_right_x: ti.f32, cam_right_y: ti.f32, cam_right_z: ti.f32,
             cam_up_x: ti.f32, cam_up_y: ti.f32, cam_up_z: ti.f32):
    cam_pos = ti.Vector([cam_pos_x, cam_pos_y, cam_pos_z])
    cam_dir = ti.Vector([cam_dir_x, cam_dir_y, cam_dir_z])
    cam_right = ti.Vector([cam_right_x, cam_right_y, cam_right_z])
    cam_up = ti.Vector([cam_up_x, cam_up_y, cam_up_z])

    fov = 45.0
    aspect = ti.cast(WIDTH, ti.f32) / ti.cast(HEIGHT, ti.f32)
    fov_scale = ti.tan(fov * 0.5 * 3.14159 / 180.0)
    light_dir = ti.Vector([1.0, 1.0, 1.0]).normalized()

    for i, j in scene.pixels:
        u = (2.0 * (ti.cast(i, ti.f32) + 0.5) / ti.cast(WIDTH, ti.f32) - 1.0) * aspect * fov_scale
        v = (2.0 * (ti.cast(j, ti.f32) + 0.5) / ti.cast(HEIGHT, ti.f32) - 1.0) * fov_scale
        ray_dir = (cam_dir + u * cam_right + v * cam_up).normalized()

        closest_t = 1e10
        hit_color = ti.Vector([0.1, 0.1, 0.15])
        hit_normal = ti.Vector([0.0, 1.0, 0.0])

        # Test all triangles (using flat indices)
        for k in range(scene.num_triangles[None]):
            idx = k * 3
            v0 = scene.vertices[scene.indices[idx]]
            v1 = scene.vertices[scene.indices[idx + 1]]
            v2 = scene.vertices[scene.indices[idx + 2]]
            t = ray_triangle_intersect(cam_pos, ray_dir, v0, v1, v2)
            if 0.001 < t < closest_t:
                closest_t = t
                hit_normal = get_triangle_normal(v0, v1, v2)
                if hit_normal.dot(ray_dir) > 0:
                    hit_normal = -hit_normal
                hit_color = scene.vertex_colors[scene.indices[idx]]

        # Shading
        if closest_t < 1e9:
            diff = ti.max(hit_normal.dot(light_dir), 0.0)
            ambient = 0.2
            scene.pixels[i, j] = hit_color * (ambient + diff * 0.8)
        else:
            scene.pixels[i, j] = ti.Vector([0.1, 0.1, 0.15])


# Benchmarked wrappers for kernels (ti.sync() needed for accurate GPU timing)
@benchmark
def run_physics(dt):
    update_physics(dt)
    if is_enabled_benchmark():
        ti.sync()


@benchmark
def run_raytrace(cam):
    raytrace(
        cam.pos[0], cam.pos[1], cam.pos[2],
        cam.direction[0], cam.direction[1], cam.direction[2],
        cam.right[0], cam.right[1], cam.right[2],
        cam.up[0], cam.up[1], cam.up[2]
    )
    if is_enabled_benchmark():
        ti.sync()


@benchmark
def run_rasterize(ti_scene, ti_camera, canvas, cam):
    cam.apply_to_ti_camera(ti_camera)
    ti_scene.set_camera(ti_camera)
    ti_scene.ambient_light((0.2, 0.2, 0.2))
    ti_scene.point_light(pos=(10, 10, 10), color=(1, 1, 1))
    ti_scene.mesh(
        scene.vertices,
        indices=scene.indices,
        per_vertex_color=scene.vertex_colors,
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
        center=(0, room_size + wall_thickness / 1.5, 0),
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
        "./models/dragon_smallest.obj",
        center=(0, 2.5, 0),
        size=8.0,
        color=(0.8, 0.3, 0.3),
        rotation=(0, 180, 0)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raytrace', action='store_true', help='Use ray tracing')
    args = parser.parse_args()

    use_raytracing = args.raytrace

    create_demo_scene()

    print(f"Scene: {scene._vertex_count} vertices, {scene._triangle_count} triangles")
    print(f"Rendering: {'Ray Tracing' if use_raytracing else 'Rasterization'}")

    window = ti.ui.Window("Taichi Scene", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()
    ti_scene = ti.ui.Scene()
    ti_camera = ti.ui.Camera()

    camera = Camera(position=(0, 5, 15), yaw=-90, pitch=-15)

    while window.running:
        dt = 1.0 / 60.0

        camera.handle_input(window, dt)
        run_physics(dt)

        if use_raytracing:
            run_raytrace(camera)
            canvas.set_image(scene.pixels)
        else:
            run_rasterize(ti_scene, ti_camera, canvas, camera)

        window.show()


if __name__ == '__main__':
    main()
