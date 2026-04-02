const std = @import("std");
const Vulkan = @import("vulkan.zig").Vulkan;
const Buffer = Vulkan.Buffer;


pub const MAX_VERTICES: u32 = 100_000;
pub const MAX_TRIANGLES: u32 = 100_000;
pub const MAX_BODIES: u32 = 1_000;
pub const MAX_GEOMS: u32 = 2_000;
pub const MAX_CONTACTS: u32 = 10_000;
pub const FIXED_DT: f32 = 1.0 / 60.0;

const VERTEX = Vulkan.USAGE_VERTEX;
const INDEX = Vulkan.USAGE_INDEX;
const STORAGE = Vulkan.USAGE_STORAGE;

pub const Data = struct {
    vk: *Vulkan,

    // Render: vertex/index data read by graphics pipeline, written by physics compute
    vertices: Buffer, // [MAX_VERTICES] float3 — world-space positions (updated each frame by compute)
    colors: Buffer, // [MAX_VERTICES] float3 — per-vertex RGB
    indices: Buffer, // [MAX_TRIANGLES*3] uint — triangle indices into vertices
    original_vertices: Buffer, // [MAX_VERTICES] float3 — local-space positions (never changes after upload)

    // Body: rigid body state, indexed by body ID
    body_pos: Buffer, // [MAX_BODIES] float3 — center of mass
    body_quat: Buffer, // [MAX_BODIES] float4 — orientation (w,x,y,z)
    body_vel: Buffer, // [MAX_BODIES] float3 — linear velocity
    body_omega: Buffer, // [MAX_BODIES] float3 — angular velocity
    body_inv_mass: Buffer, // [MAX_BODIES] float — 1/mass (0 = static)
    body_inertia: Buffer, // [MAX_BODIES] float3 — diagonal inertia tensor
    body_inv_inertia: Buffer, // [MAX_BODIES] float3 — 1/inertia
    body_vert_start: Buffer, // [MAX_BODIES] uint — first vertex index for this body
    body_vert_count: Buffer, // [MAX_BODIES] uint — number of vertices for this body

    // Geom: collision shapes attached to bodies, indexed by geom ID
    geom_type: Buffer, // [MAX_GEOMS] int — 1=sphere, 2=box, 5=mesh
    geom_body_idx: Buffer, // [MAX_GEOMS] uint — which body owns this geom
    geom_data: Buffer, // [MAX_GEOMS] float7 — type-specific (box: half_x/y/z, sphere: radius, ...)
    geom_friction: Buffer, // [MAX_GEOMS] float — Coulomb friction coefficient
    geom_world_pos: Buffer, // [MAX_GEOMS] float3 — cached world position (updated each frame)
    geom_world_quat: Buffer, // [MAX_GEOMS] float4 — cached world orientation
    geom_aabb_min: Buffer, // [MAX_GEOMS] float3 — axis-aligned bounding box min
    geom_aabb_max: Buffer, // [MAX_GEOMS] float3 — axis-aligned bounding box max

    // Contact: collision results from narrow phase, indexed by contact ID
    contact_pos: Buffer, // [MAX_CONTACTS] float3 — contact point in world space
    contact_normal: Buffer, // [MAX_CONTACTS] float3 — contact normal (A→B direction)
    contact_penetration: Buffer, // [MAX_CONTACTS] float — penetration depth
    contact_geom_a: Buffer, // [MAX_CONTACTS] uint — geom index A
    contact_geom_b: Buffer, // [MAX_CONTACTS] uint — geom index B

    // CPU staging
    s_vertices: []f32,
    s_colors: []f32,
    s_indices: []u32,
    s_orig_verts: []f32,
    s_body_pos: []f32,
    s_body_quat: []f32,
    s_body_vel: []f32,
    s_body_omega: []f32,
    s_body_inv_mass: []f32,
    s_body_inertia: []f32,
    s_body_inv_inertia: []f32,
    s_body_vert_start: []u32,
    s_body_vert_count: []u32,
    s_geom_type: []i32,
    s_geom_body_idx: []u32,
    s_geom_data: []f32,
    s_geom_friction: []f32,
    alloc: std.mem.Allocator,

    // Counters
    num_vertices: u32,
    num_triangles: u32,
    num_bodies: u32,
    num_geoms: u32,

    pub fn init(vk_ctx: *Vulkan, alloc: std.mem.Allocator) !Data {
        return Data{
            .vk = vk_ctx,
            .alloc = alloc,

            .vertices = try vk_ctx.createBuffer(MAX_VERTICES * @sizeOf([3]f32), VERTEX),
            .colors = try vk_ctx.createBuffer(MAX_VERTICES * @sizeOf([3]f32), VERTEX),
            .indices = try vk_ctx.createBuffer(MAX_TRIANGLES * 3 * @sizeOf(u32), INDEX),
            .original_vertices = try vk_ctx.createBuffer(MAX_VERTICES * @sizeOf([3]f32), STORAGE),
            .body_pos = try vk_ctx.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .body_quat = try vk_ctx.createBuffer(MAX_BODIES * @sizeOf([4]f32), STORAGE),
            .body_vel = try vk_ctx.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .body_omega = try vk_ctx.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .body_inv_mass = try vk_ctx.createBuffer(MAX_BODIES * @sizeOf(f32), STORAGE),
            .body_inertia = try vk_ctx.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .body_inv_inertia = try vk_ctx.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .body_vert_start = try vk_ctx.createBuffer(MAX_BODIES * @sizeOf(u32), STORAGE),
            .body_vert_count = try vk_ctx.createBuffer(MAX_BODIES * @sizeOf(u32), STORAGE),
            .geom_type = try vk_ctx.createBuffer(MAX_GEOMS * @sizeOf(i32), STORAGE),
            .geom_body_idx = try vk_ctx.createBuffer(MAX_GEOMS * @sizeOf(u32), STORAGE),
            .geom_data = try vk_ctx.createBuffer(MAX_GEOMS * @sizeOf([7]f32), STORAGE),
            .geom_friction = try vk_ctx.createBuffer(MAX_GEOMS * @sizeOf(f32), STORAGE),
            .geom_world_pos = try vk_ctx.createBuffer(MAX_GEOMS * @sizeOf([3]f32), STORAGE),
            .geom_world_quat = try vk_ctx.createBuffer(MAX_GEOMS * @sizeOf([4]f32), STORAGE),
            .geom_aabb_min = try vk_ctx.createBuffer(MAX_GEOMS * @sizeOf([3]f32), STORAGE),
            .geom_aabb_max = try vk_ctx.createBuffer(MAX_GEOMS * @sizeOf([3]f32), STORAGE),
            .contact_pos = try vk_ctx.createBuffer(MAX_CONTACTS * @sizeOf([3]f32), STORAGE),
            .contact_normal = try vk_ctx.createBuffer(MAX_CONTACTS * @sizeOf([3]f32), STORAGE),
            .contact_penetration = try vk_ctx.createBuffer(MAX_CONTACTS * @sizeOf(f32), STORAGE),
            .contact_geom_a = try vk_ctx.createBuffer(MAX_CONTACTS * @sizeOf(u32), STORAGE),
            .contact_geom_b = try vk_ctx.createBuffer(MAX_CONTACTS * @sizeOf(u32), STORAGE),

            .s_vertices = try alloc.alloc(f32, MAX_VERTICES * 3),
            .s_colors = try alloc.alloc(f32, MAX_VERTICES * 3),
            .s_indices = try alloc.alloc(u32, MAX_TRIANGLES * 3),
            .s_orig_verts = try alloc.alloc(f32, MAX_VERTICES * 3),
            .s_body_pos = try alloc.alloc(f32, MAX_BODIES * 3),
            .s_body_quat = try alloc.alloc(f32, MAX_BODIES * 4),
            .s_body_vel = try alloc.alloc(f32, MAX_BODIES * 3),
            .s_body_omega = try alloc.alloc(f32, MAX_BODIES * 3),
            .s_body_inv_mass = try alloc.alloc(f32, MAX_BODIES),
            .s_body_inertia = try alloc.alloc(f32, MAX_BODIES * 3),
            .s_body_inv_inertia = try alloc.alloc(f32, MAX_BODIES * 3),
            .s_body_vert_start = try alloc.alloc(u32, MAX_BODIES),
            .s_body_vert_count = try alloc.alloc(u32, MAX_BODIES),
            .s_geom_type = try alloc.alloc(i32, MAX_GEOMS),
            .s_geom_body_idx = try alloc.alloc(u32, MAX_GEOMS),
            .s_geom_data = try alloc.alloc(f32, MAX_GEOMS * 7),
            .s_geom_friction = try alloc.alloc(f32, MAX_GEOMS),

            .num_vertices = 0,
            .num_triangles = 0,
            .num_bodies = 0,
            .num_geoms = 0,
        };
    }

    // --- Scene construction ---

    pub fn addBox(self: *Data, center: [3]f32, half: [3]f32, color: [3]f32, mass: f32) !u32 {
        const vs = self.num_vertices;
        const ts = self.num_triangles;
        const bi = self.num_bodies;
        const gi = self.num_geoms;

        const box_v = [8][3]f32{
            .{ -half[0], -half[1], -half[2] }, .{ half[0], -half[1], -half[2] },
            .{ half[0], half[1], -half[2] },   .{ -half[0], half[1], -half[2] },
            .{ -half[0], -half[1], half[2] },  .{ half[0], -half[1], half[2] },
            .{ half[0], half[1], half[2] },     .{ -half[0], half[1], half[2] },
        };
        const box_f = [12][3]u32{
            .{ 0, 2, 1 }, .{ 0, 3, 2 }, .{ 5, 7, 4 }, .{ 5, 6, 7 },
            .{ 4, 3, 0 }, .{ 4, 7, 3 }, .{ 1, 6, 5 }, .{ 1, 2, 6 },
            .{ 3, 6, 2 }, .{ 3, 7, 6 }, .{ 4, 1, 5 }, .{ 4, 0, 1 },
        };

        for (0..8) |i| {
            const b = (vs + @as(u32, @intCast(i))) * 3;
            for (0..3) |j| {
                self.s_orig_verts[b + j] = box_v[i][j];
                self.s_vertices[b + j] = center[j] + box_v[i][j];
                self.s_colors[b + j] = color[j];
            }
        }
        for (0..12) |i| {
            const b = (ts + @as(u32, @intCast(i))) * 3;
            for (0..3) |j| self.s_indices[b + j] = vs + box_f[i][j];
        }
        self.num_vertices += 8;
        self.num_triangles += 12;

        const b3 = bi * 3;
        self.s_body_pos[b3..][0..3].* = center;
        self.s_body_quat[bi * 4 ..][0..4].* = .{ 1, 0, 0, 0 };
        self.s_body_vel[b3..][0..3].* = .{ 0, 0, 0 };
        self.s_body_omega[b3..][0..3].* = .{ 0, 0, 0 };
        self.s_body_inv_mass[bi] = if (mass > 0) 1.0 / mass else 0;
        self.s_body_vert_start[bi] = vs;
        self.s_body_vert_count[bi] = 8;
        if (mass > 0) {
            const wx = half[0] * 2;
            const wy = half[1] * 2;
            const wz = half[2] * 2;
            self.s_body_inertia[b3 + 0] = mass / 12.0 * (wy * wy + wz * wz);
            self.s_body_inertia[b3 + 1] = mass / 12.0 * (wx * wx + wz * wz);
            self.s_body_inertia[b3 + 2] = mass / 12.0 * (wx * wx + wy * wy);
            self.s_body_inv_inertia[b3 + 0] = 1.0 / self.s_body_inertia[b3 + 0];
            self.s_body_inv_inertia[b3 + 1] = 1.0 / self.s_body_inertia[b3 + 1];
            self.s_body_inv_inertia[b3 + 2] = 1.0 / self.s_body_inertia[b3 + 2];
        } else {
            self.s_body_inertia[b3..][0..3].* = .{ 0, 0, 0 };
            self.s_body_inv_inertia[b3..][0..3].* = .{ 0, 0, 0 };
        }
        self.num_bodies += 1;

        self.s_geom_type[gi] = 2;
        self.s_geom_body_idx[gi] = bi;
        self.s_geom_data[gi * 7 ..][0..7].* = .{ half[0], half[1], half[2], 0, 0, 0, 0 };
        self.s_geom_friction[gi] = 0.5;
        self.num_geoms += 1;

        return bi;
    }

    pub fn addSphere(self: *Data, center: [3]f32, radius: f32, color: [3]f32, segments: u32, mass: f32) !u32 {
        const vs = self.num_vertices;
        const ts = self.num_triangles;
        const bi = self.num_bodies;
        const gi = self.num_geoms;
        const num_v = (segments + 1) * (segments + 1);
        const num_t = segments * segments * 2;
        const pi = std.math.pi;

        // Vertices
        for (0..segments + 1) |i_| {
            const i: f32 = @floatFromInt(i_);
            const segs: f32 = @floatFromInt(segments);
            const lat = pi * i / segs;
            const sin_lat = @sin(lat);
            const cos_lat = @cos(lat);
            for (0..segments + 1) |j_| {
                const j: f32 = @floatFromInt(j_);
                const lon = 2.0 * pi * j / segs;
                const idx = (vs + @as(u32, @intCast(i_)) * (segments + 1) + @as(u32, @intCast(j_))) * 3;
                const local = [3]f32{
                    radius * @cos(lon) * sin_lat,
                    radius * cos_lat,
                    radius * @sin(lon) * sin_lat,
                };
                for (0..3) |k| {
                    self.s_orig_verts[idx + k] = local[k];
                    self.s_vertices[idx + k] = center[k] + local[k];
                    self.s_colors[idx + k] = color[k];
                }
            }
        }
        self.num_vertices += num_v;

        // Triangles
        for (0..segments) |i_| {
            for (0..segments) |j_| {
                const cur = vs + @as(u32, @intCast(i_)) * (segments + 1) + @as(u32, @intCast(j_));
                const nxt = cur + segments + 1;
                const t = (ts + @as(u32, @intCast(i_)) * segments + @as(u32, @intCast(j_))) * 6;
                if (i_ == 0) {
                    // Top pole
                    self.s_indices[t + 0] = cur;
                    self.s_indices[t + 1] = nxt;
                    self.s_indices[t + 2] = nxt + 1;
                    self.s_indices[t + 3] = cur;
                    self.s_indices[t + 4] = nxt;
                    self.s_indices[t + 5] = nxt + 1;
                } else if (i_ == segments - 1) {
                    // Bottom pole
                    self.s_indices[t + 0] = cur;
                    self.s_indices[t + 1] = nxt;
                    self.s_indices[t + 2] = cur + 1;
                    self.s_indices[t + 3] = cur;
                    self.s_indices[t + 4] = nxt;
                    self.s_indices[t + 5] = cur + 1;
                } else {
                    // Normal quad: two triangles
                    self.s_indices[t + 0] = cur;
                    self.s_indices[t + 1] = nxt;
                    self.s_indices[t + 2] = cur + 1;
                    self.s_indices[t + 3] = cur + 1;
                    self.s_indices[t + 4] = nxt;
                    self.s_indices[t + 5] = nxt + 1;
                }
            }
        }
        self.num_triangles += num_t;

        // Body
        const b3 = bi * 3;
        self.s_body_pos[b3..][0..3].* = center;
        self.s_body_quat[bi * 4 ..][0..4].* = .{ 1, 0, 0, 0 };
        self.s_body_vel[b3..][0..3].* = .{ 0, 0, 0 };
        self.s_body_omega[b3..][0..3].* = .{ 0, 0, 0 };
        self.s_body_inv_mass[bi] = if (mass > 0) 1.0 / mass else 0;
        self.s_body_vert_start[bi] = vs;
        self.s_body_vert_count[bi] = num_v;
        if (mass > 0) {
            const inertia = 0.4 * mass * radius * radius;
            self.s_body_inertia[b3..][0..3].* = .{ inertia, inertia, inertia };
            self.s_body_inv_inertia[b3..][0..3].* = .{ 1.0 / inertia, 1.0 / inertia, 1.0 / inertia };
        } else {
            self.s_body_inertia[b3..][0..3].* = .{ 0, 0, 0 };
            self.s_body_inv_inertia[b3..][0..3].* = .{ 0, 0, 0 };
        }
        self.num_bodies += 1;

        // Geom
        self.s_geom_type[gi] = 1; // SPHERE
        self.s_geom_body_idx[gi] = bi;
        self.s_geom_data[gi * 7 ..][0..7].* = .{ radius, 0, 0, 0, 0, 0, 0 };
        self.s_geom_friction[gi] = 0.5;
        self.num_geoms += 1;

        return bi;
    }

    /// Upload all staging data to GPU
    pub fn upload(self: *Data) !void {
        const v = self.vk;
        try v.uploadSlice(self.vertices, f32, self.s_vertices[0 .. self.num_vertices * 3]);
        try v.uploadSlice(self.colors, f32, self.s_colors[0 .. self.num_vertices * 3]);
        try v.uploadSlice(self.indices, u32, self.s_indices[0 .. self.num_triangles * 3]);
        try v.uploadSlice(self.original_vertices, f32, self.s_orig_verts[0 .. self.num_vertices * 3]);
        try v.uploadSlice(self.body_pos, f32, self.s_body_pos[0 .. self.num_bodies * 3]);
        try v.uploadSlice(self.body_quat, f32, self.s_body_quat[0 .. self.num_bodies * 4]);
        try v.uploadSlice(self.body_vel, f32, self.s_body_vel[0 .. self.num_bodies * 3]);
        try v.uploadSlice(self.body_omega, f32, self.s_body_omega[0 .. self.num_bodies * 3]);
        try v.uploadSlice(self.body_inv_mass, f32, self.s_body_inv_mass[0..self.num_bodies]);
        try v.uploadSlice(self.body_inertia, f32, self.s_body_inertia[0 .. self.num_bodies * 3]);
        try v.uploadSlice(self.body_inv_inertia, f32, self.s_body_inv_inertia[0 .. self.num_bodies * 3]);
        try v.uploadSlice(self.body_vert_start, u32, self.s_body_vert_start[0..self.num_bodies]);
        try v.uploadSlice(self.body_vert_count, u32, self.s_body_vert_count[0..self.num_bodies]);
        try v.uploadSlice(self.geom_type, i32, self.s_geom_type[0..self.num_geoms]);
        try v.uploadSlice(self.geom_body_idx, u32, self.s_geom_body_idx[0..self.num_geoms]);
        try v.uploadSlice(self.geom_data, f32, self.s_geom_data[0 .. self.num_geoms * 7]);
        try v.uploadSlice(self.geom_friction, f32, self.s_geom_friction[0..self.num_geoms]);
    }

    pub fn deinit(self: *Data) void {
        const bufs = &[_]*Buffer{
            &self.vertices, &self.colors, &self.indices, &self.original_vertices,
            &self.body_pos, &self.body_quat, &self.body_vel, &self.body_omega,
            &self.body_inv_mass, &self.body_inertia, &self.body_inv_inertia,
            &self.body_vert_start, &self.body_vert_count,
            &self.geom_type, &self.geom_body_idx, &self.geom_data, &self.geom_friction,
            &self.geom_world_pos, &self.geom_world_quat, &self.geom_aabb_min, &self.geom_aabb_max,
            &self.contact_pos, &self.contact_normal, &self.contact_penetration,
            &self.contact_geom_a, &self.contact_geom_b,
        };
        for (bufs) |b| self.vk.destroyBuffer(b.*);

        self.alloc.free(self.s_vertices);
        self.alloc.free(self.s_colors);
        self.alloc.free(self.s_indices);
        self.alloc.free(self.s_orig_verts);
        self.alloc.free(self.s_body_pos);
        self.alloc.free(self.s_body_quat);
        self.alloc.free(self.s_body_vel);
        self.alloc.free(self.s_body_omega);
        self.alloc.free(self.s_body_inv_mass);
        self.alloc.free(self.s_body_inertia);
        self.alloc.free(self.s_body_inv_inertia);
        self.alloc.free(self.s_body_vert_start);
        self.alloc.free(self.s_body_vert_count);
        self.alloc.free(self.s_geom_type);
        self.alloc.free(self.s_geom_body_idx);
        self.alloc.free(self.s_geom_data);
        self.alloc.free(self.s_geom_friction);
    }
};
