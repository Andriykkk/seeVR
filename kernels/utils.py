import taichi as ti


@ti.func
def intersect_aabb(ray_o, ray_d, bmin, bmax, closest_t):
    """Ray-AABB intersection test"""
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


@ti.func
def ray_triangle_intersect(ray_o, ray_d, v0, v1, v2):
    """Moller-Trumbore algorithm"""
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
