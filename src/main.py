import taichi as ti
import math
import time
from benchmark import benchmark, is_enabled_benchmark

ti.init(arch=ti.gpu)

# --- Data ---

MAX_VERTICES = 100000
MAX_TRIANGLES = 100000
WIDTH, HEIGHT = 800, 600
MAX_BODIES = 1000

RigidBody = ti.types.struct(
    pos=ti.types.vector(3, ti.f32),       # Center of mass position
    quat=ti.types.vector(4, ti.f32),      # Orientation quaternion (w, x, y, z)
    vel=ti.types.vector(3, ti.f32),       # Linear velocity
    omega=ti.types.vector(3, ti.f32),     # Angular velocity
    mass=ti.f32,                          # Mass (0 = static/infinite mass)
    inv_mass=ti.f32,                      # 1/mass (0 for static)
    inertia=ti.types.vector(3, ti.f32),   # Diagonal inertia tensor (local space)
    inv_inertia=ti.types.vector(3, ti.f32), # 1/inertia
    # Render mesh mapping
    vert_start=ti.i32,                    # Start index in vertices array
    vert_count=ti.i32,                    # Number of vertices for this body
)


@ti.data_oriented
class Data:
    def __init__(self):
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)
        self.indices = ti.field(dtype=ti.i32, shape=MAX_TRIANGLES * 3)
        self.vertex_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)
        self.num_vertices = ti.field(dtype=ti.i32, shape=())
        self.num_triangles = ti.field(dtype=ti.i32, shape=())
        self.bodies = RigidBody.field(shape=MAX_BODIES)
        self.num_bodies = ti.field(dtype=ti.i32, shape=())


data = Data()

# --- Scene ---


@ti.data_oriented
class Scene:

    @ti.kernel
    def _add_box_gpu(self, center: ti.types.vector(3, ti.f32),
                     half_size: ti.types.vector(3, ti.f32),
                     color: ti.types.vector(3, ti.f32),
                     mass: ti.f32):
        vs = ti.atomic_add(data.num_vertices[None], 8)
        ts = ti.atomic_add(data.num_triangles[None], 12)
        bi = ti.atomic_add(data.num_bodies[None], 1)

        box_v = ti.static([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
        ])
        box_f = ti.static([
            [0, 1, 2], [0, 2, 3],
            [5, 4, 7], [5, 7, 6],
            [4, 0, 3], [4, 3, 7],
            [1, 5, 6], [1, 6, 2],
            [3, 2, 6], [3, 6, 7],
            [4, 5, 1], [4, 1, 0],
        ])

        for i in ti.static(range(8)):
            data.vertices[vs + i] = center + half_size * ti.Vector(box_v[i])
            data.vertex_colors[vs + i] = color

        for i in ti.static(range(12)):
            for j in ti.static(range(3)):
                data.indices[ts * 3 + i * 3 + j] = vs + box_f[i][j]

        # Rigid body
        data.bodies[bi].pos = center
        data.bodies[bi].quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
        data.bodies[bi].vel = ti.Vector([0.0, 0.0, 0.0])
        data.bodies[bi].omega = ti.Vector([0.0, 0.0, 0.0])
        data.bodies[bi].mass = mass
        data.bodies[bi].inv_mass = 1.0 / mass if mass > 0.0 else 0.0
        data.bodies[bi].vert_start = vs
        data.bodies[bi].vert_count = 8

        if mass > 0.0:
            wx = half_size[0] * 2.0
            wy = half_size[1] * 2.0
            wz = half_size[2] * 2.0
            ix = mass / 12.0 * (wy * wy + wz * wz)
            iy = mass / 12.0 * (wx * wx + wz * wz)
            iz = mass / 12.0 * (wx * wx + wy * wy)
            data.bodies[bi].inertia = ti.Vector([ix, iy, iz])
            data.bodies[bi].inv_inertia = ti.Vector([1.0 / ix, 1.0 / iy, 1.0 / iz])
        else:
            data.bodies[bi].inertia = ti.Vector([0.0, 0.0, 0.0])
            data.bodies[bi].inv_inertia = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _add_sphere_gpu(self, center: ti.types.vector(3, ti.f32), radius: ti.f32,
                        color: ti.types.vector(3, ti.f32), segments: ti.i32,
                        mass: ti.f32):
        num_v = (segments + 1) * (segments + 1)
        num_t = segments * segments * 2
        vs = ti.atomic_add(data.num_vertices[None], num_v)
        ts = ti.atomic_add(data.num_triangles[None], num_t)
        bi = ti.atomic_add(data.num_bodies[None], 1)

        pi = 3.14159265358979

        for i in range(segments + 1):
            lat = pi * ti.cast(i, ti.f32) / ti.cast(segments, ti.f32)
            sin_lat = ti.sin(lat)
            cos_lat = ti.cos(lat)
            for j in range(segments + 1):
                lon = 2.0 * pi * ti.cast(j, ti.f32) / ti.cast(segments, ti.f32)
                idx = vs + i * (segments + 1) + j
                data.vertices[idx] = center + radius * ti.Vector([
                    ti.cos(lon) * sin_lat, cos_lat, ti.sin(lon) * sin_lat
                ])
                data.vertex_colors[idx] = color

        for i in range(segments):
            for j in range(segments):
                cur = vs + i * (segments + 1) + j
                nxt = cur + segments + 1
                t = ts * 3 + (i * segments + j) * 6
                # Triangle 1
                data.indices[t + 0] = cur
                data.indices[t + 1] = nxt
                data.indices[t + 2] = cur + 1
                # Triangle 2
                data.indices[t + 3] = cur + 1
                data.indices[t + 4] = nxt
                data.indices[t + 5] = nxt + 1

        # Rigid body
        data.bodies[bi].pos = center
        data.bodies[bi].quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
        data.bodies[bi].vel = ti.Vector([0.0, 0.0, 0.0])
        data.bodies[bi].omega = ti.Vector([0.0, 0.0, 0.0])
        data.bodies[bi].mass = mass
        data.bodies[bi].inv_mass = 1.0 / mass if mass > 0.0 else 0.0
        data.bodies[bi].vert_start = vs
        data.bodies[bi].vert_count = num_v

        if mass > 0.0:
            inertia = 0.4 * mass * radius * radius
            data.bodies[bi].inertia = ti.Vector([inertia, inertia, inertia])
            data.bodies[bi].inv_inertia = ti.Vector([1.0 / inertia, 1.0 / inertia, 1.0 / inertia])
        else:
            data.bodies[bi].inertia = ti.Vector([0.0, 0.0, 0.0])
            data.bodies[bi].inv_inertia = ti.Vector([0.0, 0.0, 0.0])

    @benchmark
    def add_box(self, center, half_size, color, mass=1.0):
        self._add_box_gpu(center, half_size, color, mass)
        if is_enabled_benchmark():
            ti.sync()

    @benchmark
    def add_sphere(self, center, radius, color, segments=16, mass=1.0):
        self._add_sphere_gpu(center, radius, color, segments, mass)
        if is_enabled_benchmark():
            ti.sync()


# --- Camera ---

class Camera:
    def __init__(self, position=(0, 5, 15), yaw=-90.0, pitch=-15.0):
        self.pos = list(position)
        self.yaw = yaw
        self.pitch = pitch
        self.speed = 10.0
        self.sensitivity = 0.5
        self._last_mouse = None
        self.direction = [0, 0, -1]

    def _update_vectors(self):
        ry = math.radians(self.yaw)
        rp = math.radians(self.pitch)
        self.direction = [
            math.cos(rp) * math.cos(ry),
            math.sin(rp),
            math.cos(rp) * math.sin(ry),
        ]

    def handle_input(self, window, dt):
        if window.is_pressed('w'):
            ry = math.radians(self.yaw)
            self.pos[0] += math.cos(ry) * self.speed * dt
            self.pos[2] += math.sin(ry) * self.speed * dt
        if window.is_pressed('s'):
            ry = math.radians(self.yaw)
            self.pos[0] -= math.cos(ry) * self.speed * dt
            self.pos[2] -= math.sin(ry) * self.speed * dt
        if window.is_pressed('a'):
            ry = math.radians(self.yaw - 90)
            self.pos[0] += math.cos(ry) * self.speed * dt
            self.pos[2] += math.sin(ry) * self.speed * dt
        if window.is_pressed('d'):
            ry = math.radians(self.yaw + 90)
            self.pos[0] += math.cos(ry) * self.speed * dt
            self.pos[2] += math.sin(ry) * self.speed * dt
        if window.is_pressed(ti.ui.SPACE):
            self.pos[1] += self.speed * dt
        if window.is_pressed(ti.ui.SHIFT):
            self.pos[1] -= self.speed * dt

        cur = window.get_cursor_pos()
        if window.is_pressed(ti.ui.RMB) and self._last_mouse is not None:
            dx = (cur[0] - self._last_mouse[0]) * 200
            dy = (cur[1] - self._last_mouse[1]) * 200
            self.yaw += dx * self.sensitivity
            self.pitch = max(-89, min(89, self.pitch + dy * self.sensitivity))
        self._last_mouse = cur
        self._update_vectors()

    def apply(self, ti_camera):
        ti_camera.position(*self.pos)
        ti_camera.lookat(
            self.pos[0] + self.direction[0],
            self.pos[1] + self.direction[1],
            self.pos[2] + self.direction[2],
        )


# --- Main ---

scene = Scene()


@benchmark
def create_demo_scene():
    # Ground (static)
    scene.add_box(
        ti.Vector([0.0, -0.25, 0.0]),
        ti.Vector([10.0, 0.25, 10.0]),
        ti.Vector([0.3, 0.3, 0.35]),
        mass=0.0,
    )

    # Some boxes
    scene.add_box(
        ti.Vector([0.0, 1.0, 0.0]),
        ti.Vector([0.5, 0.5, 0.5]),
        ti.Vector([0.8, 0.3, 0.3]),
    )
    scene.add_box(
        ti.Vector([2.0, 0.75, -1.0]),
        ti.Vector([0.75, 0.75, 0.75]),
        ti.Vector([0.3, 0.8, 0.3]),
    )

    # Some spheres
    scene.add_sphere(ti.Vector([-2.0, 1.5, 0.0]), 0.7, ti.Vector([0.2, 0.2, 0.9]), 16)
    scene.add_sphere(ti.Vector([0.0, 3.0, -2.0]), 0.5, ti.Vector([0.9, 0.9, 0.2]), 16)

    if is_enabled_benchmark():
        ti.sync()


@benchmark
def render_frame(camera, window, canvas, ti_scene, ti_camera, dt):
    camera.handle_input(window, dt)
    camera.apply(ti_camera)

    ti_scene.set_camera(ti_camera)
    ti_scene.ambient_light((0.4, 0.4, 0.4))
    ti_scene.point_light(pos=(10, 10, 10), color=(0.8, 0.8, 0.8))
    ti_scene.point_light(pos=(-10, 8, -10), color=(0.5, 0.5, 0.5))
    ti_scene.mesh(
        data.vertices,
        indices=data.indices,
        per_vertex_color=data.vertex_colors,
        two_sided=True,
    )
    canvas.scene(ti_scene)

    if is_enabled_benchmark():
        ti.sync()


FIXED_DT = 1.0 / 64


def main():
    create_demo_scene()
    print(f"Scene: {data.num_vertices[None]} vertices, {data.num_triangles[None]} triangles")

    window = ti.ui.Window("Simple Viewer", (WIDTH, HEIGHT), vsync=False)
    canvas = window.get_canvas()
    ti_scene = window.get_scene()
    ti_camera = ti.ui.Camera()
    camera = Camera(position=(0, 5, 15), yaw=-90, pitch=-15)

    frame = 0
    last_time = time.perf_counter()
    fps_smooth = 0.0

    while window.running:
        render_frame(camera, window, canvas, ti_scene, ti_camera, FIXED_DT)

        now = time.perf_counter()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps_smooth = 0.1 * (1.0 / dt) + 0.9 * fps_smooth

        gui = window.get_gui()
        gui.begin("Debug", 0.02, 0.02, 0.25, 0.15)
        gui.text(f"FPS: {fps_smooth:.1f}")
        gui.text(f"Frame: {frame}")
        gui.text(f"Verts: {data.num_vertices[None]}  Tris: {data.num_triangles[None]}  Bodies: {data.num_bodies[None]}")
        gui.end()

        frame += 1
        window.show()


if __name__ == '__main__':
    main()
