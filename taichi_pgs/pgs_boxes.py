import taichi as ti
import taichi.math as tm
import time

ti.init(arch=ti.cpu)

# --- Config ---
NUM_DYNAMIC = 6
NUM_BOXES = 1 + NUM_DYNAMIC
SUBSTEPS = 10
DT = 1.0 / (60.0 * SUBSTEPS)
GRAVITY = tm.vec3(0.0, -9.81, 0.0)
PGS_ITERS = 30
FRICTION = 0.5
BAUMGARTE = 0.2
SLOP = 0.005
MAX_CONTACTS = 256

# --- State ---
pos = ti.Vector.field(3, float, NUM_BOXES)
vel = ti.Vector.field(3, float, NUM_BOXES)
quat = ti.Vector.field(4, float, NUM_BOXES)  # x,y,z,w
omega = ti.Vector.field(3, float, NUM_BOXES)
half_size = ti.Vector.field(3, float, NUM_BOXES)
inv_mass = ti.field(float, NUM_BOXES)
inv_inertia = ti.Vector.field(3, float, NUM_BOXES)

# --- Contacts ---
num_contacts = ti.field(int, ())
c_a = ti.field(int, MAX_CONTACTS)
c_b = ti.field(int, MAX_CONTACTS)
c_n = ti.Vector.field(3, float, MAX_CONTACTS)
c_p = ti.Vector.field(3, float, MAX_CONTACTS)
c_pen = ti.field(float, MAX_CONTACTS)
c_ln = ti.field(float, MAX_CONTACTS)

# --- Rendering ---
mesh_verts = ti.Vector.field(3, float, NUM_BOXES * 8)
mesh_colors = ti.Vector.field(3, float, NUM_BOXES * 8)
mesh_indices = ti.field(int, NUM_BOXES * 36)
box_color = ti.Vector.field(3, float, NUM_BOXES)


# ======================== Helpers ========================

@ti.func
def quat_to_mat(q: tm.vec4) -> tm.mat3:
    x, y, z, w = q[0], q[1], q[2], q[3]
    return ti.Matrix([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


@ti.func
def quat_mul(a: tm.vec4, b: tm.vec4) -> tm.vec4:
    return tm.vec4(
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    )


@ti.func
def col3(M: tm.mat3, c: int) -> tm.vec3:
    return tm.vec3(M[0, c], M[1, c], M[2, c])


@ti.func
def apply_inv_I(idx: int, v: tm.vec3) -> tm.vec3:
    R = quat_to_mat(quat[idx])
    return R @ (inv_inertia[idx] * (R.transpose() @ v))


# ======================== SAT Collision ========================

@ti.func
def sat_test(a: int, b: int):
    """OBB-OBB SAT with 6 face axes + 9 edge-cross axes.
    Returns (overlap, ref_is_a, face_idx, pen, normal_a_to_b).
    For edge axes: ref_is_a = -1, face_idx encodes i*3+j."""
    Ra = quat_to_mat(quat[a])
    Rb = quat_to_mat(quat[b])
    ha = half_size[a]
    hb = half_size[b]
    d = pos[b] - pos[a]
    C = Ra.transpose() @ Rb
    t = Ra.transpose() @ d

    absC = ti.Matrix.zero(float, 3, 3)
    for r in range(3):
        for cc in range(3):
            absC[r, cc] = ti.abs(C[r, cc]) + 1e-6

    min_pen = 1e10
    ref_is_a = 1
    best_face = 0
    best_n = tm.vec3(0.0, 1.0, 0.0)
    sep = 0

    # A's 3 face normals
    for i in range(3):
        r_a = ha[i]
        r_b = hb[0] * absC[i, 0] + hb[1] * absC[i, 1] + hb[2] * absC[i, 2]
        pen = r_a + r_b - ti.abs(t[i])
        if pen < 0.0:
            sep = 1
        if pen < min_pen and sep == 0:
            min_pen = pen
            ref_is_a = 1
            best_face = i
            s = 1.0 if t[i] >= 0.0 else -1.0
            best_n = col3(Ra, i) * s

    # B's 3 face normals
    for i in range(3):
        tp = t[0] * C[0, i] + t[1] * C[1, i] + t[2] * C[2, i]
        r_a = ha[0] * absC[0, i] + ha[1] * absC[1, i] + ha[2] * absC[2, i]
        r_b = hb[i]
        pen = r_a + r_b - ti.abs(tp)
        if pen < 0.0:
            sep = 1
        if pen < min_pen and sep == 0:
            min_pen = pen
            ref_is_a = 0
            best_face = i
            s = 1.0 if tp >= 0.0 else -1.0
            best_n = col3(Rb, i) * s

    # 9 edge-cross axes: A_i x B_j
    for i in range(3):
        i0 = (i + 1) % 3
        i1 = (i + 2) % 3
        for j in range(3):
            j0 = (j + 1) % 3
            j1 = (j + 2) % 3
            tp = t[i1] * C[i0, j] - t[i0] * C[i1, j]
            r_a = ha[i0] * absC[i1, j] + ha[i1] * absC[i0, j]
            r_b = hb[j0] * absC[i, j1] + hb[j1] * absC[i, j0]
            pen = r_a + r_b - ti.abs(tp)
            if pen < 0.0:
                sep = 1
            if sep == 0:
                axis = col3(Ra, i).cross(col3(Rb, j))
                axis_len = axis.norm()
                if axis_len > 1e-4:
                    pen_norm = pen / axis_len
                    if pen_norm < min_pen:
                        min_pen = pen_norm
                        ref_is_a = -1  # edge-edge
                        best_face = i * 3 + j
                        s = 1.0 if d.dot(axis) >= 0.0 else -1.0
                        best_n = axis / axis_len * s

    return 1 - sep, ref_is_a, best_face, min_pen, best_n


@ti.func
def add_face_contacts(ref: int, inc: int, ref_face: int, normal: tm.vec3):
    """Project incident face vertices onto reference face. Normal: ref -> inc."""
    R_ref = quat_to_mat(quat[ref])
    R_inc = quat_to_mat(quat[inc])
    h_ref = half_size[ref]
    h_inc = half_size[inc]

    # Reference face
    ref_center = pos[ref] + normal * h_ref[ref_face]
    rt1 = (ref_face + 1) % 3
    rt2 = (ref_face + 2) % 3
    tan1 = col3(R_ref, rt1)
    tan2 = col3(R_ref, rt2)

    # Find incident face (most anti-aligned with normal)
    min_d = 1e10
    inc_ax = 0
    inc_sg = 1.0
    for k in range(3):
        ck = col3(R_inc, k)
        dp = ck.dot(normal)
        if dp < min_d:
            min_d = dp
            inc_ax = k
            inc_sg = 1.0
        if -dp < min_d:
            min_d = -dp
            inc_ax = k
            inc_sg = -1.0

    inc_n = col3(R_inc, inc_ax) * inc_sg
    inc_c = pos[inc] + inc_n * h_inc[inc_ax]
    i1 = (inc_ax + 1) % 3
    i2 = (inc_ax + 2) % 3
    it1 = col3(R_inc, i1)
    it2 = col3(R_inc, i2)

    # Project each incident vertex onto reference face
    for k in range(4):
        s1 = 1.0 if k & 1 else -1.0
        s2 = 1.0 if k & 2 else -1.0
        v = inc_c + it1 * (s1 * h_inc[i1]) + it2 * (s2 * h_inc[i2])

        d = (v - ref_center).dot(normal)
        if d < 0.01:
            rel = v - ref_center
            u = rel.dot(tan1)
            w = rel.dot(tan2)
            if ti.abs(u) <= h_ref[rt1] + 0.01 and ti.abs(w) <= h_ref[rt2] + 0.01:
                depth = ti.max(-d, 0.0)
                ci = ti.atomic_add(num_contacts[None], 1)
                if ci < MAX_CONTACTS:
                    c_a[ci] = ref
                    c_b[ci] = inc
                    c_n[ci] = normal
                    c_p[ci] = v - d * normal
                    c_pen[ci] = depth


@ti.func
def add_edge_contact(a: int, b: int, ei: int, ej: int, normal: tm.vec3, pen: float):
    """Single contact point for edge-edge case."""
    Ra = quat_to_mat(quat[a])
    Rb = quat_to_mat(quat[b])
    ha = half_size[a]
    hb = half_size[b]

    da = col3(Ra, ei)
    db = col3(Rb, ej)

    # Find edge base points (on the rim of each box facing the other)
    pa = pos[a]
    for k in range(3):
        if k != ei:
            ck = col3(Ra, k)
            s = 1.0 if normal.dot(ck) > 0.0 else -1.0
            pa += s * ha[k] * ck
    pb = pos[b]
    for k in range(3):
        if k != ej:
            ck = col3(Rb, k)
            s = -1.0 if normal.dot(ck) > 0.0 else 1.0
            pb += s * hb[k] * ck

    # Closest points on two lines
    r = pa - pb
    a12 = da.dot(db)
    b1 = r.dot(da)
    b2 = r.dot(db)
    denom = 1.0 - a12 * a12
    sa = 0.0
    tb = 0.0
    if ti.abs(denom) > 1e-8:
        sa = (a12 * b2 - b1) / denom
        tb = (b2 - a12 * b1) / denom
    sa = ti.max(-ha[ei], ti.min(sa, ha[ei]))
    tb = ti.max(-hb[ej], ti.min(tb, hb[ej]))

    pt_a = pa + sa * da
    pt_b = pb + tb * db
    contact_pt = (pt_a + pt_b) * 0.5

    ci = ti.atomic_add(num_contacts[None], 1)
    if ci < MAX_CONTACTS:
        c_a[ci] = a
        c_b[ci] = b
        c_n[ci] = normal
        c_p[ci] = contact_pt
        c_pen[ci] = pen


@ti.kernel
def detect_contacts():
    num_contacts[None] = 0
    for i in range(NUM_BOXES):
        for j in range(i + 1, NUM_BOXES):
            if inv_mass[i] > 0.0 or inv_mass[j] > 0.0:
                overlap, ref_is_a, face, pen, normal = sat_test(i, j)
                if overlap == 1:
                    if ref_is_a == 1:
                        add_face_contacts(i, j, face, normal)
                    elif ref_is_a == 0:
                        add_face_contacts(j, i, face, -normal)
                    else:
                        ei = face // 3
                        ej = face % 3
                        add_edge_contact(i, j, ei, ej, normal, pen)


# ======================== Solver ========================

@ti.kernel
def apply_gravity():
    for i in range(NUM_BOXES):
        if inv_mass[i] > 0.0:
            vel[i] += GRAVITY * DT


@ti.kernel
def clear_impulses():
    for i in range(MAX_CONTACTS):
        c_ln[i] = 0.0


@ti.kernel
def pgs_solve():
    for _ in range(1):  # single thread — critical for PGS convergence
        nc = num_contacts[None]
        for ci in range(nc):
            a = c_a[ci]
            b = c_b[ci]
            n = c_n[ci]
            p = c_p[ci]
            ra = p - pos[a]
            rb = p - pos[b]

            va = vel[a] + omega[a].cross(ra)
            vb = vel[b] + omega[b].cross(rb)
            vn = (vb - va).dot(n)

            ra_x_n = ra.cross(n)
            rb_x_n = rb.cross(n)
            k = (inv_mass[a] + inv_mass[b]
                 + ra_x_n.dot(apply_inv_I(a, ra_x_n))
                 + rb_x_n.dot(apply_inv_I(b, rb_x_n)))

            bias = BAUMGARTE * ti.max(c_pen[ci] - SLOP, 0.0) / DT
            d_ln = (-vn + bias) / k
            old = c_ln[ci]
            c_ln[ci] = ti.max(old + d_ln, 0.0)
            d_ln = c_ln[ci] - old

            imp = d_ln * n
            vel[a] -= inv_mass[a] * imp
            vel[b] += inv_mass[b] * imp
            omega[a] -= apply_inv_I(a, ra.cross(imp))
            omega[b] += apply_inv_I(b, rb.cross(imp))

            # Coulomb friction
            va2 = vel[a] + omega[a].cross(ra)
            vb2 = vel[b] + omega[b].cross(rb)
            dv2 = vb2 - va2
            vt = dv2 - dv2.dot(n) * n
            vt_len = vt.norm()
            if vt_len > 1e-6:
                td = vt / vt_len
                ra_x_t = ra.cross(td)
                rb_x_t = rb.cross(td)
                kt = (inv_mass[a] + inv_mass[b]
                      + ra_x_t.dot(apply_inv_I(a, ra_x_t))
                      + rb_x_t.dot(apply_inv_I(b, rb_x_t)))
                d_lt = ti.min(vt_len / kt, FRICTION * c_ln[ci])
                imp_t = d_lt * td
                vel[a] += inv_mass[a] * imp_t
                vel[b] -= inv_mass[b] * imp_t
                omega[a] += apply_inv_I(a, ra.cross(imp_t))
                omega[b] -= apply_inv_I(b, rb.cross(imp_t))


@ti.kernel
def integrate():
    for i in range(NUM_BOXES):
        if inv_mass[i] > 0.0:
            vel[i] *= 0.999
            omega[i] *= 0.998
            pos[i] += vel[i] * DT
            w_q = tm.vec4(omega[i][0], omega[i][1], omega[i][2], 0.0)
            q_dot = 0.5 * quat_mul(w_q, quat[i])
            q = quat[i] + q_dot * DT
            quat[i] = q / q.norm()


# ======================== Rendering ========================

@ti.kernel
def update_mesh():
    for bi in range(NUM_BOXES):
        h = half_size[bi]
        p = pos[bi]
        R = quat_to_mat(quat[bi])
        for ci in range(8):
            sx = 1.0 if ci & 1 else -1.0
            sy = 1.0 if ci & 2 else -1.0
            sz = 1.0 if ci & 4 else -1.0
            mesh_verts[bi * 8 + ci] = p + R @ tm.vec3(sx * h[0], sy * h[1], sz * h[2])
            mesh_colors[bi * 8 + ci] = box_color[bi]


def init_mesh_indices():
    faces = [[0, 4, 6, 2], [1, 3, 7, 5], [0, 1, 5, 4],
             [2, 6, 7, 3], [0, 2, 3, 1], [4, 5, 7, 6]]
    for bi in range(NUM_BOXES):
        for fi, face in enumerate(faces):
            idx = (bi * 12 + fi * 2) * 3
            v = [bi * 8 + f for f in face]
            mesh_indices[idx] = v[0]
            mesh_indices[idx + 1] = v[1]
            mesh_indices[idx + 2] = v[2]
            mesh_indices[idx + 3] = v[0]
            mesh_indices[idx + 4] = v[2]
            mesh_indices[idx + 5] = v[3]


# ======================== Scene ========================

def add_box(idx, p, hs, color, mass):
    pos[idx] = p
    half_size[idx] = hs
    vel[idx] = tm.vec3(0.0, 0.0, 0.0)
    omega[idx] = tm.vec3(0.0, 0.0, 0.0)
    quat[idx] = tm.vec4(0.0, 0.0, 0.0, 1.0)
    box_color[idx] = color
    if mass == 0.0:
        inv_mass[idx] = 0.0
        inv_inertia[idx] = tm.vec3(0.0, 0.0, 0.0)
    else:
        inv_mass[idx] = 1.0 / mass
        inv_inertia[idx] = tm.vec3(
            3.0 / (mass * (hs[1] ** 2 + hs[2] ** 2)),
            3.0 / (mass * (hs[0] ** 2 + hs[2] ** 2)),
            3.0 / (mass * (hs[0] ** 2 + hs[1] ** 2)),
        )


def init_scene():
    # Ground (static) — matches main.zig
    add_box(0, tm.vec3(0, -0.25, 0), tm.vec3(10, 0.25, 10),
            tm.vec3(0.3, 0.3, 0.35), 0.0)
    # Stacked boxes
    add_box(1, tm.vec3(0, 1, 0), tm.vec3(0.5, 0.5, 0.5),
            tm.vec3(0.85, 0.25, 0.25), 1.0)
    add_box(2, tm.vec3(0.02, 2.5, 0), tm.vec3(0.5, 0.5, 0.5),
            tm.vec3(0.25, 0.55, 0.85), 1.0)
    add_box(3, tm.vec3(-0.02, 4.0, 0), tm.vec3(0.5, 0.5, 0.5),
            tm.vec3(0.85, 0.65, 0.2), 1.0)
    add_box(4, tm.vec3(0.01, 5.5, 0), tm.vec3(0.5, 0.5, 0.5),
            tm.vec3(0.35, 0.8, 0.35), 1.0)
    # Side boxes
    add_box(5, tm.vec3(3, 1, 0), tm.vec3(0.75, 0.75, 0.75),
            tm.vec3(0.3, 0.8, 0.3), 1.0)
    add_box(6, tm.vec3(-3, 2, 1), tm.vec3(0.6, 0.6, 0.6),
            tm.vec3(0.8, 0.5, 0.2), 1.0)


# ======================== Main ========================

def main():
    init_scene()
    init_mesh_indices()

    window = ti.ui.Window("PGS Boxes 3D", (1024, 768))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    # Orbit camera state
    cam_dist = 12.0
    cam_yaw = 0.7
    cam_pitch = 0.5
    cam_target = [0.0, 1.5, 0.0]
    last_mouse = None
    mouse_down = False

    fc = 0
    lt = time.time()

    while window.running:
        # --- Mouse orbit camera ---
        import math
        cur_mouse = window.get_cursor_pos()

        if window.is_pressed(ti.ui.LMB):
            if last_mouse is not None and mouse_down:
                dx = cur_mouse[0] - last_mouse[0]
                dy = cur_mouse[1] - last_mouse[1]
                cam_yaw -= dx * 3.0
                cam_pitch += dy * 3.0
                cam_pitch = max(-1.4, min(1.4, cam_pitch))
            mouse_down = True
        else:
            mouse_down = False
        last_mouse = cur_mouse

        # Scroll zoom via keyboard ([ and ] or - and =)
        if window.is_pressed('w'):
            cam_dist = max(2.0, cam_dist - 0.1)
        if window.is_pressed('s'):
            cam_dist = min(30.0, cam_dist + 0.1)

        cx = cam_target[0] + cam_dist * math.cos(cam_pitch) * math.sin(cam_yaw)
        cy = cam_target[1] + cam_dist * math.sin(cam_pitch)
        cz = cam_target[2] + cam_dist * math.cos(cam_pitch) * math.cos(cam_yaw)
        camera.position(cx, cy, cz)
        camera.lookat(cam_target[0], cam_target[1], cam_target[2])
        camera.up(0.0, 1.0, 0.0)

        # --- Physics ---
        for _ in range(SUBSTEPS):
            apply_gravity()
            detect_contacts()
            clear_impulses()
            for _ in range(PGS_ITERS):
                pgs_solve()
            integrate()

        # --- Render ---
        update_mesh()
        scene.set_camera(camera)
        scene.ambient_light((0.3, 0.3, 0.3))
        scene.point_light(pos=(5.0, 10.0, 5.0), color=(1.0, 1.0, 1.0))
        scene.mesh(mesh_verts, mesh_indices, per_vertex_color=mesh_colors)
        canvas.scene(scene)
        window.show()

        fc += 1
        now = time.time()
        if now - lt >= 1.0:
            print(f"FPS: {fc / (now - lt):.1f}")
            fc = 0
            lt = now


if __name__ == "__main__":
    main()
