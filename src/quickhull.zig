const std = @import("std");
const Vec = @import("vec.zig").Vec;
const math = std.math;

const V3 = [3]f32;

fn sub(a: V3, b: V3) V3 {
    return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

fn cross(a: V3, b: V3) V3 {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

fn dot(a: V3, b: V3) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

fn length(a: V3) f32 {
    return @sqrt(dot(a, a));
}

fn normalize(a: V3) V3 {
    const l = length(a);
    if (l < 1e-10) return a;
    return .{ a[0] / l, a[1] / l, a[2] / l };
}

fn faceNormal(pts: []const V3, f: [3]u32) V3 {
    return normalize(cross(sub(pts[f[1]], pts[f[0]]), sub(pts[f[2]], pts[f[0]])));
}

fn pointDist(pts: []const V3, pi: u32, f: [3]u32) f32 {
    return dot(sub(pts[pi], pts[f[0]]), faceNormal(pts, f));
}

fn orientFace(pts: []const V3, f: [3]u32, centroid: V3) [3]u32 {
    const fc = V3{
        (pts[f[0]][0] + pts[f[1]][0] + pts[f[2]][0]) / 3.0,
        (pts[f[0]][1] + pts[f[1]][1] + pts[f[2]][1]) / 3.0,
        (pts[f[0]][2] + pts[f[1]][2] + pts[f[2]][2]) / 3.0,
    };
    if (dot(faceNormal(pts, f), sub(fc, centroid)) < 0) {
        return .{ f[0], f[2], f[1] };
    }
    return f;
}

/// Compute convex hull indices from a set of 3D points.
/// Returns a Vec of unique vertex indices that form the hull.
pub fn quickhull3d(pts: []const V3, alloc: std.mem.Allocator) !Vec(u32) {
    var result = Vec(u32).init(alloc);
    const n: u32 = @intCast(pts.len);

    if (n < 4) {
        for (0..n) |i| try result.push(@intCast(i));
        return result;
    }

    // Find 6 extreme points (min/max on each axis)
    var ext: [6]u32 = .{ 0, 0, 0, 0, 0, 0 };
    for (0..n) |i_| {
        const i: u32 = @intCast(i_);
        for (0..3) |a| {
            if (pts[i][a] < pts[ext[a * 2]][a]) ext[a * 2] = i;
            if (pts[i][a] > pts[ext[a * 2 + 1]][a]) ext[a * 2 + 1] = i;
        }
    }

    // Find most distant pair among extremes
    var p0: u32 = 0;
    var p1: u32 = 1;
    var best_dist: f32 = 0;
    for (0..6) |i| {
        for (i + 1..6) |j| {
            const d = length(sub(pts[ext[i]], pts[ext[j]]));
            if (d > best_dist) {
                best_dist = d;
                p0 = ext[i];
                p1 = ext[j];
            }
        }
    }

    // Find point most distant from line p0-p1
    const line_dir = normalize(sub(pts[p1], pts[p0]));
    var p2: u32 = 0;
    var best_d2: f32 = 0;
    for (0..n) |i_| {
        const i: u32 = @intCast(i_);
        if (i == p0 or i == p1) continue;
        const v = sub(pts[i], pts[p0]);
        const proj = V3{ v[0] - dot(v, line_dir) * line_dir[0], v[1] - dot(v, line_dir) * line_dir[1], v[2] - dot(v, line_dir) * line_dir[2] };
        const d = length(proj);
        if (d > best_d2) {
            best_d2 = d;
            p2 = i;
        }
    }

    // Find point most distant from plane p0-p1-p2
    const plane_n = normalize(cross(sub(pts[p1], pts[p0]), sub(pts[p2], pts[p0])));
    var p3: u32 = 0;
    var best_d3: f32 = 0;
    for (0..n) |i_| {
        const i: u32 = @intCast(i_);
        if (i == p0 or i == p1 or i == p2) continue;
        const d = @abs(dot(sub(pts[i], pts[p0]), plane_n));
        if (d > best_d3) {
            best_d3 = d;
            p3 = i;
        }
    }

    const centroid = V3{
        (pts[p0][0] + pts[p1][0] + pts[p2][0] + pts[p3][0]) / 4.0,
        (pts[p0][1] + pts[p1][1] + pts[p2][1] + pts[p3][1]) / 4.0,
        (pts[p0][2] + pts[p1][2] + pts[p2][2] + pts[p3][2]) / 4.0,
    };

    // Track which points are on hull
    const on_hull = try alloc.alloc(bool, n);
    defer alloc.free(on_hull);
    @memset(on_hull, false);
    on_hull[p0] = true;
    on_hull[p1] = true;
    on_hull[p2] = true;
    on_hull[p3] = true;

    // Initial tetrahedron faces
    var faces = Vec([3]u32).init(alloc);
    defer faces.deinit();
    try faces.push(orientFace(pts, .{ p0, p1, p2 }, centroid));
    try faces.push(orientFace(pts, .{ p0, p2, p3 }, centroid));
    try faces.push(orientFace(pts, .{ p0, p3, p1 }, centroid));
    try faces.push(orientFace(pts, .{ p1, p3, p2 }, centroid));

    // Outside sets: for each face, which points are in front of it
    var outside = Vec(Vec(u32)).init(alloc);
    defer {
        for (outside.items[0..outside.len]) |*o| o.deinit();
        outside.deinit();
    }
    for (0..4) |fi| {
        var outs = Vec(u32).init(alloc);
        for (0..n) |i_| {
            const i: u32 = @intCast(i_);
            if (!on_hull[i] and pointDist(pts, i, faces.items[fi]) > 1e-10) {
                try outs.push(i);
            }
        }
        try outside.push(outs);
    }

    // Main loop
    while (true) {
        // Find point furthest from any face
        var best_fi: i32 = -1;
        var best_pi: u32 = 0;
        var best_val: f32 = 0;
        for (0..faces.len) |fi| {
            for (outside.items[fi].items[0..outside.items[fi].len]) |pi| {
                const d = pointDist(pts, pi, faces.items[fi]);
                if (d > best_val) {
                    best_val = d;
                    best_fi = @intCast(fi);
                    best_pi = pi;
                }
            }
        }
        if (best_fi < 0) break;

        const new_pt = best_pi;
        on_hull[new_pt] = true;

        // Find visible faces
        var visible = Vec(u32).init(alloc);
        defer visible.deinit();
        for (0..faces.len) |fi| {
            if (pointDist(pts, new_pt, faces.items[fi]) > 1e-10) {
                try visible.push(@intCast(fi));
            }
        }

        // Find horizon edges (edges shared by exactly one visible face)
        const Edge = struct { a: u32, b: u32 };
        var horizon = Vec(Edge).init(alloc);
        defer horizon.deinit();

        for (visible.items[0..visible.len]) |fi| {
            const f = faces.items[fi];
            const edges = [3]Edge{
                .{ .a = f[0], .b = f[1] },
                .{ .a = f[1], .b = f[2] },
                .{ .a = f[2], .b = f[0] },
            };
            for (edges) |e| {
                const k_min = @min(e.a, e.b);
                const k_max = @max(e.a, e.b);
                // Check if this edge is shared with another visible face
                var shared = false;
                for (visible.items[0..visible.len]) |fi2| {
                    if (fi2 == fi) continue;
                    const f2 = faces.items[fi2];
                    const edges2 = [3][2]u32{
                        .{ f2[0], f2[1] },
                        .{ f2[1], f2[2] },
                        .{ f2[2], f2[0] },
                    };
                    for (edges2) |e2| {
                        if (@min(e2[0], e2[1]) == k_min and @max(e2[0], e2[1]) == k_max) {
                            shared = true;
                        }
                    }
                }
                if (!shared) try horizon.push(e);
            }
        }

        // Collect all outside points from visible faces
        var all_out = Vec(u32).init(alloc);
        defer all_out.deinit();
        for (visible.items[0..visible.len]) |fi| {
            for (outside.items[fi].items[0..outside.items[fi].len]) |pi| {
                if (pi != new_pt and !on_hull[pi]) try all_out.push(pi);
            }
        }

        // Remove visible faces (reverse order to keep indices valid)
        var vi: u32 = visible.len;
        while (vi > 0) {
            vi -= 1;
            const fi = visible.items[vi];
            // Swap-remove face and outside set
            outside.items[fi].deinit();
            if (fi < faces.len - 1) {
                faces.items[fi] = faces.items[faces.len - 1];
                outside.items[fi] = outside.items[outside.len - 1];
            }
            faces.len -= 1;
            outside.len -= 1;
        }

        // Add new faces from horizon edges to new_pt
        for (horizon.items[0..horizon.len]) |e| {
            const nf = orientFace(pts, .{ e.b, e.a, new_pt }, centroid);
            try faces.push(nf);

            var outs = Vec(u32).init(alloc);
            for (all_out.items[0..all_out.len]) |pi| {
                if (!on_hull[pi] and pointDist(pts, pi, nf) > 1e-10) {
                    try outs.push(pi);
                }
            }
            try outside.push(outs);
        }
    }

    // Collect unique hull vertex indices
    for (0..n) |i| {
        if (on_hull[i]) try result.push(@intCast(i));
    }

    return result;
}
