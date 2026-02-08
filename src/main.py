import taichi as ti
import math
import time
from benchmark import benchmark, is_enabled_benchmark
from data import data, GEOM_BOX, GEOM_SPHERE, MAX_VERTICES, MAX_TRIANGLES, MAX_BODIES, MAX_GEOMS, WIDTH, HEIGHT, FIXED_DT, FRAME_TIME, DEBUG
from physics import apply_gravity, integrate_bodies, compute_aabb
from utils import quat_rotate

# --- Scene ---

@ti.data_oriented
class Scene:
    def __init__(self, title="Simple Viewer", width=WIDTH, height=HEIGHT):
        self.window = ti.ui.Window(title, (width, height), vsync=False)
        self.canvas = self.window.get_canvas()
        self.ti_scene = self.window.get_scene()
        self.ti_camera = ti.ui.Camera()

    @ti.kernel
    def _add_box_gpu(self, center: ti.types.vector(3, ti.f32),
                     half_size: ti.types.vector(3, ti.f32),
                     color: ti.types.vector(3, ti.f32),
                     mass: ti.f32):
        vs = ti.atomic_add(data.num_vertices[None], 8)
        ts = ti.atomic_add(data.num_triangles[None], 12)
        bi = ti.atomic_add(data.num_bodies[None], 1)
        gi = ti.atomic_add(data.num_geoms[None], 1)

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
            local = half_size * ti.Vector(box_v[i])
            data.original_vertices[vs + i] = local
            data.vertices[vs + i] = center + local
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

        # Collision geom
        data.geoms[gi].geom_type = GEOM_BOX
        data.geoms[gi].body_idx = bi
        data.geoms[gi].local_pos = ti.Vector([0.0, 0.0, 0.0])
        data.geoms[gi].local_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
        data.geoms[gi].data = ti.Vector([half_size[0], half_size[1], half_size[2], 0.0, 0.0, 0.0, 0.0])
        data.geoms[gi].aabb_min, data.geoms[gi].aabb_max = compute_aabb(data.vertices, vs, vs + 8, center)

    @ti.kernel
    def _add_sphere_gpu(self, center: ti.types.vector(3, ti.f32), radius: ti.f32,
                        color: ti.types.vector(3, ti.f32), segments: ti.i32,
                        mass: ti.f32):
        num_v = (segments + 1) * (segments + 1)
        num_t = segments * segments * 2
        vs = ti.atomic_add(data.num_vertices[None], num_v)
        ts = ti.atomic_add(data.num_triangles[None], num_t)
        bi = ti.atomic_add(data.num_bodies[None], 1)
        gi = ti.atomic_add(data.num_geoms[None], 1)

        pi = 3.14159265358979

        for i in range(segments + 1):
            lat = pi * ti.cast(i, ti.f32) / ti.cast(segments, ti.f32)
            sin_lat = ti.sin(lat)
            cos_lat = ti.cos(lat)
            for j in range(segments + 1):
                lon = 2.0 * pi * ti.cast(j, ti.f32) / ti.cast(segments, ti.f32)
                idx = vs + i * (segments + 1) + j
                local = radius * ti.Vector([
                    ti.cos(lon) * sin_lat, cos_lat, ti.sin(lon) * sin_lat
                ])
                data.original_vertices[idx] = local
                data.vertices[idx] = center + local
                data.vertex_colors[idx] = color

        for i in range(segments):
            for j in range(segments):
                cur = vs + i * (segments + 1) + j
                nxt = cur + segments + 1
                t = ts * 3 + (i * segments + j) * 6
                if i == 0:
                    # Top pole: only fan triangle, duplicate it for slot 1
                    data.indices[t + 0] = cur
                    data.indices[t + 1] = nxt
                    data.indices[t + 2] = nxt + 1
                    data.indices[t + 3] = cur
                    data.indices[t + 4] = nxt
                    data.indices[t + 5] = nxt + 1
                elif i == segments - 1:
                    # Bottom pole: only fan triangle, duplicate it for slot 2
                    data.indices[t + 0] = cur
                    data.indices[t + 1] = nxt
                    data.indices[t + 2] = cur + 1
                    data.indices[t + 3] = cur
                    data.indices[t + 4] = nxt
                    data.indices[t + 5] = cur + 1
                else:
                    # Normal quad: two triangles
                    data.indices[t + 0] = cur
                    data.indices[t + 1] = nxt
                    data.indices[t + 2] = cur + 1
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

        # Collision geom
        data.geoms[gi].geom_type = GEOM_SPHERE
        data.geoms[gi].body_idx = bi
        data.geoms[gi].local_pos = ti.Vector([0.0, 0.0, 0.0])
        data.geoms[gi].local_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
        data.geoms[gi].data = ti.Vector([radius, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        data.geoms[gi].aabb_min, data.geoms[gi].aabb_max = compute_aabb(data.vertices, vs, vs + num_v, center)

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


@ti.kernel
def _build_wireframe():
    num_t = data.num_triangles[None]
    data.num_wire_verts[None] = num_t * 6
    for i in range(num_t):
        v0 = data.vertices[data.indices[i * 3 + 0]]
        v1 = data.vertices[data.indices[i * 3 + 1]]
        v2 = data.vertices[data.indices[i * 3 + 2]]
        base = i * 6
        data.wire_verts[base + 0] = v0
        data.wire_verts[base + 1] = v1
        data.wire_verts[base + 2] = v1
        data.wire_verts[base + 3] = v2
        data.wire_verts[base + 4] = v2
        data.wire_verts[base + 5] = v0

@ti.kernel
def update_render_vertices(num_bodies: ti.i32):
    """Transform render mesh vertices from local to world space.

    world_vertex = body.pos + quat_rotate(body.quat, local_vertex)
    """
    for b in range(num_bodies):
        body_pos = data.bodies[b].pos
        body_quat = data.bodies[b].quat
        start = data.bodies[b].vert_start
        count = data.bodies[b].vert_count

        for v in range(start, start + count):
            local_pos = data.original_vertices[v]
            data.vertices[v] = body_pos + quat_rotate(body_quat, local_pos)

@benchmark
def step(camera, scene, dt, physics_dt):
    camera.handle_input(scene.window, dt)
    # physics_dt = dt * time_scale (0 = paused, <1 = slow-mo, 1 = real-time)
    # TODO: integrate physics here with physics_dt

    apply_gravity(data.num_bodies[None], physics_dt)

    integrate_bodies(data.num_bodies[None], physics_dt)
    update_render_vertices(data.num_bodies[None])

    if is_enabled_benchmark():
        ti.sync()


@benchmark
def render(camera, scene, gui):
    camera.apply(scene.ti_camera)

    scene.ti_scene.set_camera(scene.ti_camera)
    scene.ti_scene.ambient_light((0.4, 0.4, 0.4))
    scene.ti_scene.point_light(pos=(10, 10, 10), color=(0.8, 0.8, 0.8))
    scene.ti_scene.point_light(pos=(-10, 8, -10), color=(0.5, 0.5, 0.5))
    scene.ti_scene.mesh(
        data.vertices,
        indices=data.indices,
        per_vertex_color=data.vertex_colors,
        two_sided=True,
    )
    if DEBUG and gui.show_wireframe:
        _build_wireframe()
        scene.ti_scene.lines(
            data.wire_verts,
            width=1.0,
            color=(0.0, 0.0, 0.0),
            vertex_count=data.num_wire_verts[None],
        )
    scene.canvas.scene(scene.ti_scene)

    if is_enabled_benchmark():
        ti.sync()

# --- GUI ---

class GUI:
    def __init__(self, scene):
        self.scene = scene
        self.frame = 0
        self.last_time = time.perf_counter()
        self.fps_smooth = 0.0
        self.time_scale = 1.0
        # Debug flags
        self.show_wireframe = False

    def update(self):
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.fps_smooth = 0.1 * (1.0 / dt) + 0.9 * self.fps_smooth

        if DEBUG:
            imgui = self.scene.window.get_gui()
            imgui.begin("Debug", 0.02, 0.02, 0.3, 0.3)
            imgui.text(f"FPS: {self.fps_smooth:.1f}")
            imgui.text(f"Frame: {self.frame}")
            imgui.text(f"Verts: {data.num_vertices[None]}  Tris: {data.num_triangles[None]}  Bodies: {data.num_bodies[None]}  Geoms: {data.num_geoms[None]}")
            allocated, used, _ = data.gpu_memory()
            imgui.text(f"GPU: {used / 1048576:.1f} / {allocated / 1048576:.1f} MB")
            self.time_scale = imgui.slider_float("Time Scale", self.time_scale, 0.0, 2.0)
            self.show_wireframe = imgui.checkbox("Wireframe", self.show_wireframe)
            imgui.end()

        self.frame += 1
        self.scene.window.show()


def main():
    create_demo_scene()
    print(f"Scene: {data.num_vertices[None]} vertices, {data.num_triangles[None]} triangles")

    gui = GUI(scene)
    camera = Camera(position=(0, 5, 15), yaw=-90, pitch=-15)

    while scene.window.running:
        step(camera, scene, FIXED_DT, FIXED_DT * gui.time_scale)
        render(camera, scene, gui)
        gui.update()


if __name__ == '__main__':
    main()
