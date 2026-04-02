import taichi as ti
import math
import time
import numpy as np
from benchmark import benchmark, is_enabled_benchmark
from data import data, GEOM_BOX, GEOM_SPHERE, GEOM_MESH, MAX_VERTICES, MAX_TRIANGLES, MAX_BODIES, MAX_GEOMS, WIDTH, HEIGHT, FIXED_DT, FRAME_TIME, DEBUG
from physics import apply_gravity, integrate_bodies, compute_aabb, get_world_aabb, update_geom_transforms, broad_phase, narrow_phase, perturb_contacts, solve_contacts, solve_contacts_penalty
from utils import quat_rotate
from raytracing import run_raytrace
from bvh import build_lbvh

USE_RAYTRACING = False

# --- Quickhull ---

def _face_normal(pts, f):
    n = np.cross(pts[f[1]] - pts[f[0]], pts[f[2]] - pts[f[0]])
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-10 else n

def _point_dist(pts, pi, f):
    return np.dot(pts[pi] - pts[f[0]], _face_normal(pts, f))

def _orient_face(pts, f, centroid):
    fc = (pts[f[0]] + pts[f[1]] + pts[f[2]]) / 3.0
    if np.dot(_face_normal(pts, f), fc - centroid) < 0:
        return [f[0], f[2], f[1]]
    return f

def quickhull_3d(verts):
    pts = np.asarray(verts, dtype=np.float64)
    n = len(pts)
    if n < 4:
        return pts.astype(np.float32)

    # Initial tetrahedron from extreme points
    ext = [np.argmin(pts[:,i//2]) if i%2==0 else np.argmax(pts[:,i//2]) for i in range(6)]
    p0, p1 = max(((ext[i], ext[j]) for i in range(6) for j in range(i+1,6)),
                  key=lambda p: np.linalg.norm(pts[p[0]] - pts[p[1]]))

    line = pts[p1] - pts[p0]
    line /= np.linalg.norm(line)
    p2 = max((i for i in range(n) if i not in (p0,p1)),
             key=lambda i: np.linalg.norm(pts[i] - pts[p0] - np.dot(pts[i]-pts[p0], line)*line))

    norm = np.cross(pts[p1]-pts[p0], pts[p2]-pts[p0])
    norm /= np.linalg.norm(norm)
    p3 = max((i for i in range(n) if i not in (p0,p1,p2)),
             key=lambda i: abs(np.dot(pts[i]-pts[p0], norm)))

    centroid = (pts[p0] + pts[p1] + pts[p2] + pts[p3]) / 4.0
    on_hull = {p0, p1, p2, p3}

    faces = [_orient_face(pts, f, centroid) for f in [[p0,p1,p2], [p0,p2,p3], [p0,p3,p1], [p1,p3,p2]]]
    outside = [[i for i in range(n) if i not in on_hull and _point_dist(pts, i, f) > 1e-10] for f in faces]

    while True:
        best = (-1, -1, 0)
        for fi, f in enumerate(faces):
            for pi in outside[fi]:
                d = _point_dist(pts, pi, f)
                if d > best[2]:
                    best = (fi, pi, d)
        if best[0] < 0:
            break

        new_pt = best[1]
        on_hull.add(new_pt)

        visible = [fi for fi, f in enumerate(faces) if _point_dist(pts, new_pt, f) > 1e-10]
        edges = {}
        for fi in visible:
            for i in range(3):
                e = (faces[fi][i], faces[fi][(i+1)%3])
                k = (min(e), max(e))
                edges[k] = edges.get(k, []) + [e]
        horizon = [es[0] for es in edges.values() if len(es) == 1]

        all_out = {pi for fi in visible for pi in outside[fi]} - {new_pt}

        for fi in sorted(visible, reverse=True):
            faces.pop(fi)
            outside.pop(fi)

        for e0, e1 in horizon:
            nf = _orient_face(pts, [e1, e0, new_pt], centroid)
            faces.append(nf)
            outside.append([pi for pi in all_out if pi not in on_hull and _point_dist(pts, pi, nf) > 1e-10])

    hull_idx = sorted(on_hull)
    return np.asarray(verts)[hull_idx].astype(np.float32)


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
            [0, 2, 1], [0, 3, 2],   # front  (-Z)
            [5, 7, 4], [5, 6, 7],   # back   (+Z)
            [4, 3, 0], [4, 7, 3],   # left   (-X)
            [1, 6, 5], [1, 2, 6],   # right  (+X)
            [3, 6, 2], [3, 7, 6],   # top    (+Y)
            [4, 1, 5], [4, 0, 1],   # bottom (-Y)
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
        data.geoms[gi].friction = 0.5
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
        data.geoms[gi].friction = 0.5
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

    @ti.kernel
    def _add_mesh_gpu(self, verts_np: ti.types.ndarray(dtype=ti.f32, ndim=2),
                      faces_np: ti.types.ndarray(dtype=ti.i32, ndim=2),
                      hull_verts: ti.types.ndarray(dtype=ti.f32, ndim=2),
                      num_v: ti.i32, num_t: ti.i32, num_hull: ti.i32,
                      center: ti.types.vector(3, ti.f32),
                      color: ti.types.vector(3, ti.f32),
                      mass: ti.f32):
        vs = ti.atomic_add(data.num_vertices[None], num_v)
        ts = ti.atomic_add(data.num_triangles[None], num_t)
        cs = ti.atomic_add(data.num_collision_verts[None], num_hull)
        bi = ti.atomic_add(data.num_bodies[None], 1)
        gi = ti.atomic_add(data.num_geoms[None], 1)

        for i in range(num_v):
            local = ti.Vector([verts_np[i, 0], verts_np[i, 1], verts_np[i, 2]])
            data.original_vertices[vs + i] = local
            data.vertices[vs + i] = center + local
            data.vertex_colors[vs + i] = color

        for i in range(num_t):
            data.indices[(ts + i) * 3 + 0] = vs + faces_np[i, 0]
            data.indices[(ts + i) * 3 + 1] = vs + faces_np[i, 1]
            data.indices[(ts + i) * 3 + 2] = vs + faces_np[i, 2]

        # Copy hull vertices to collision_verts
        for i in range(num_hull):
            data.collision_verts[cs + i] = ti.Vector([hull_verts[i, 0], hull_verts[i, 1], hull_verts[i, 2]])

        data.bodies[bi].pos = center
        data.bodies[bi].quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
        data.bodies[bi].vel = ti.Vector([0.0, 0.0, 0.0])
        data.bodies[bi].omega = ti.Vector([0.0, 0.0, 0.0])
        data.bodies[bi].mass = mass
        data.bodies[bi].inv_mass = 1.0 / mass if mass > 0.0 else 0.0
        data.bodies[bi].vert_start = vs
        data.bodies[bi].vert_count = num_v

        if mass > 0.0:
            # Rough inertia from AABB
            aabb_min, aabb_max = compute_aabb(data.vertices, vs, vs + num_v, center)
            sz = aabb_max - aabb_min
            ix = mass / 12.0 * (sz[1] * sz[1] + sz[2] * sz[2])
            iy = mass / 12.0 * (sz[0] * sz[0] + sz[2] * sz[2])
            iz = mass / 12.0 * (sz[0] * sz[0] + sz[1] * sz[1])
            data.bodies[bi].inertia = ti.Vector([ix, iy, iz])
            data.bodies[bi].inv_inertia = ti.Vector([1.0 / ix, 1.0 / iy, 1.0 / iz])
        else:
            data.bodies[bi].inertia = ti.Vector([0.0, 0.0, 0.0])
            data.bodies[bi].inv_inertia = ti.Vector([0.0, 0.0, 0.0])

        # Collision geom — GEOM_MESH with hull verts in collision_verts
        aabb_min, aabb_max = compute_aabb(data.vertices, vs, vs + num_v, center)
        data.geoms[gi].geom_type = GEOM_MESH
        data.geoms[gi].body_idx = bi
        data.geoms[gi].local_pos = ti.Vector([0.0, 0.0, 0.0])
        data.geoms[gi].local_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
        data.geoms[gi].data = ti.Vector([ti.cast(cs, ti.f32), ti.cast(num_hull, ti.f32), 0.0, 0.0, 0.0, 0.0, 0.0])
        data.geoms[gi].friction = 0.5
        data.geoms[gi].aabb_min = aabb_min
        data.geoms[gi].aabb_max = aabb_max

    def add_mesh(self, filename, center, color, mass=1.0, scale=1.0):
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
                    indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                    for i in range(1, len(indices) - 1):
                        faces.append([indices[0], indices[i], indices[i + 1]])

        if not raw_verts:
            return

        verts_np = np.array(raw_verts, dtype=np.float32)
        centroid = verts_np.mean(axis=0)
        verts_np -= centroid
        verts_np *= scale

        hull_verts = quickhull_3d(verts_np)

        faces_np = np.array(faces, dtype=np.int32)
        self._add_mesh_gpu(verts_np, faces_np, hull_verts, len(raw_verts), len(faces), len(hull_verts), center, color, mass)


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
        self.right = [1, 0, 0]
        self.up = [0, 1, 0]

    def _update_vectors(self):
        ry = math.radians(self.yaw)
        rp = math.radians(self.pitch)
        dx = math.cos(rp) * math.cos(ry)
        dy = math.sin(rp)
        dz = math.cos(rp) * math.sin(ry)
        self.direction = [dx, dy, dz]
        # right = normalize(direction x (0,1,0)) = (-dz, 0, dx)
        rx = -dz
        ry2 = 0.0
        rz = dx
        rl = math.sqrt(rx * rx + rz * rz)
        if rl > 1e-8:
            rx /= rl
            rz /= rl
        self.right = [rx, ry2, rz]
        # up = right x direction
        self.up = [
            ry2 * dz - rz * dy,
            rz * dx - rx * dz,
            rx * dy - ry2 * dx,
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

    # Mesh
    scene.add_mesh("../models/cylinder.obj", ti.Vector([0.0, 2.0, 2.0]), ti.Vector([0.9, 0.5, 0.2]), scale=0.2)

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
def _build_aabb_lines():
    num_g = data.num_geoms[None]
    data.num_aabb_verts[None] = num_g * 24
    for i in range(num_g):
        mn, mx = get_world_aabb(i)
        # 8 corners of the AABB box
        c0 = ti.Vector([mn[0], mn[1], mn[2]])
        c1 = ti.Vector([mx[0], mn[1], mn[2]])
        c2 = ti.Vector([mx[0], mx[1], mn[2]])
        c3 = ti.Vector([mn[0], mx[1], mn[2]])
        c4 = ti.Vector([mn[0], mn[1], mx[2]])
        c5 = ti.Vector([mx[0], mn[1], mx[2]])
        c6 = ti.Vector([mx[0], mx[1], mx[2]])
        c7 = ti.Vector([mn[0], mx[1], mx[2]])
        # 12 edges = 24 line endpoints
        base = i * 24
        # Bottom face
        data.aabb_verts[base + 0] = c0; data.aabb_verts[base + 1] = c1
        data.aabb_verts[base + 2] = c1; data.aabb_verts[base + 3] = c2
        data.aabb_verts[base + 4] = c2; data.aabb_verts[base + 5] = c3
        data.aabb_verts[base + 6] = c3; data.aabb_verts[base + 7] = c0
        # Top face
        data.aabb_verts[base + 8] = c4; data.aabb_verts[base + 9] = c5
        data.aabb_verts[base + 10] = c5; data.aabb_verts[base + 11] = c6
        data.aabb_verts[base + 12] = c6; data.aabb_verts[base + 13] = c7
        data.aabb_verts[base + 14] = c7; data.aabb_verts[base + 15] = c4
        # Vertical edges
        data.aabb_verts[base + 16] = c0; data.aabb_verts[base + 17] = c4
        data.aabb_verts[base + 18] = c1; data.aabb_verts[base + 19] = c5
        data.aabb_verts[base + 20] = c2; data.aabb_verts[base + 21] = c6
        data.aabb_verts[base + 22] = c3; data.aabb_verts[base + 23] = c7

@ti.kernel
def _build_contact_lines(cam_pos: ti.types.vector(3, ti.f32), cam_dir: ti.types.vector(3, ti.f32)):
    nc = data.num_contacts[None]
    data.num_contact_verts[None] = nc * 2
    near = 0.5  # distance from camera
    for i in range(nc):
        p = data.contacts[i].pos
        n = data.contacts[i].normal
        # Project to near-camera plane: offset along view direction
        to_point = (p - cam_pos).normalized()
        screen_p = cam_pos + to_point * near
        screen_n = cam_pos + (p + n * 0.3 - cam_pos).normalized() * near
        data.contact_verts[i * 2 + 0] = screen_p
        data.contact_verts[i * 2 + 1] = screen_n

@ti.kernel
def _build_hull_dots(num_geoms: ti.i32):
    count = 0
    for gi in range(num_geoms):
        if data.geoms[gi].geom_type == GEOM_MESH:
            start = ti.cast(data.geoms[gi].data[0], ti.i32)
            num_h = ti.cast(data.geoms[gi].data[1], ti.i32)
            pos = data.geoms[gi].world_pos
            q = data.geoms[gi].world_quat
            base = ti.atomic_add(count, num_h)
            for i in range(num_h):
                local_v = data.collision_verts[start + i]
                data.hull_debug_verts[base + i] = pos + quat_rotate(q, local_v)
    data.num_hull_debug_verts[None] = count

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
def build_bvh():
    """Build BVH acceleration structure using GPU LBVH"""
    build_lbvh(data.num_triangles[None])
    if is_enabled_benchmark():
        ti.sync()

@benchmark
def step(camera, scene, dt, physics_dt):
    camera.handle_input(scene.window, dt)
    # physics_dt = dt * time_scale (0 = paused, <1 = slow-mo, 1 = real-time)
    # TODO: integrate physics here with physics_dt

    # Update transforms for collision detection
    update_geom_transforms(data.num_geoms[None])

    # Collision detection
    broad_phase(data.num_geoms[None])
    narrow_phase(data.num_collision_pairs[None])
    perturb_contacts(data.num_contacts[None])

    # Apply gravity, then solve contacts (so solver cancels gravity at contacts)
    apply_gravity(data.num_bodies[None], physics_dt)
    solve_contacts(physics_dt)

    # Integrate with corrected velocities
    integrate_bodies(data.num_bodies[None], physics_dt)

    # Sync render vertices
    update_render_vertices(data.num_bodies[None])

    if is_enabled_benchmark():
        ti.sync()

@benchmark
def render(camera, scene, gui):
    camera.apply(scene.ti_camera)
    scene.ti_scene.set_camera(scene.ti_camera)

    if USE_RAYTRACING:
        build_bvh()
        run_raytrace(camera, gui.frame, gui)
        scene.canvas.set_image(data.pixels)
    else:
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
        if DEBUG and gui.show_aabb:
            _build_aabb_lines()
            scene.ti_scene.lines(
                data.aabb_verts,
                width=3.0,
                color=(0.0, 1.0, 0.0),
                vertex_count=data.num_aabb_verts[None],
            )
        if DEBUG and gui.show_hull:
            _build_hull_dots(data.num_geoms[None])
            nh = data.num_hull_debug_verts[None]
            if nh > 0:
                scene.ti_scene.particles(
                    data.hull_debug_verts,
                    radius=0.03,
                    color=(1.0, 1.0, 0.0),
                    index_count=nh,
                )
        if DEBUG and gui.show_contacts:
            cam = ti.Vector([camera.pos[0], camera.pos[1], camera.pos[2]])
            cam_d = ti.Vector([camera.direction[0], camera.direction[1], camera.direction[2]])
            _build_contact_lines(cam, cam_d)
            nc = data.num_contact_verts[None]
            if nc > 0:
                scene.ti_scene.lines(
                    data.contact_verts,
                    width=3.0,
                    color=(1.0, 0.0, 0.0),
                    vertex_count=nc,
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
        self.show_aabb = False
        self.show_contacts = False
        self.show_hull = False
        # Raytracing settings
        self.max_bounces = 4
        self.samples_per_pixel = 1
        self.sky_intensity = 1.0

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
            imgui.text(f"Collision pairs: {data.num_collision_pairs[None]}  Contacts: {data.num_contacts[None]}")
            allocated, used, _ = data.gpu_memory()
            imgui.text(f"GPU: {used / 1048576:.1f} / {allocated / 1048576:.1f} MB")
            self.time_scale = imgui.slider_float("Time Scale", self.time_scale, 0.0, 2.0)
            self.show_wireframe = imgui.checkbox("Wireframe", self.show_wireframe)
            self.show_aabb = imgui.checkbox("AABB", self.show_aabb)
            self.show_contacts = imgui.checkbox("Contacts", self.show_contacts)
            self.show_hull = imgui.checkbox("Hull", self.show_hull)
            if USE_RAYTRACING:
                self.max_bounces = int(imgui.slider_float("Max Bounces", self.max_bounces, 1, 16))
                self.samples_per_pixel = int(imgui.slider_float("Samples/Pixel", self.samples_per_pixel, 1, 32))
                self.sky_intensity = imgui.slider_float("Sky Intensity", self.sky_intensity, 0.0, 5.0)
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
