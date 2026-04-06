const std = @import("std");
const Vulkan = @import("vulkan.zig").Vulkan;
const Buffer = Vulkan.Buffer;

const VERTEX = Vulkan.USAGE_VERTEX;
const INDEX = Vulkan.USAGE_INDEX;
const STORAGE = Vulkan.USAGE_STORAGE;

pub const MAX_VERTICES: u32 = 10_000;
pub const MAX_TRIANGLES: u32 = 10_000;
pub const MAX_BODIES: u32 = 256;
pub const MAX_GEOMS: u32 = 256;
pub const MAX_CONTACTS: u32 = 4_000;
pub const MAX_COLLISION_PAIRS: u32 = 10_000;

pub const Data = struct {
    vk: *Vulkan,
    alloc: std.mem.Allocator,

    // ---- GPU Render buffers ----
    vertices: Buffer, // [MAX_VERTICES] float3 — world-space positions, updated each frame by compute
    colors: Buffer, // [MAX_VERTICES] float3 — per-vertex RGB color
    indices: Buffer, // [MAX_TRIANGLES*3] uint — triangle indices into vertices
    original_vertices: Buffer, // [MAX_VERTICES] float3 — local-space positions, never changes after upload

    // ---- GPU Body state (dynamics only, no collision shape data) ----
    body_pos: Buffer, // [MAX_BODIES] float3 — center of mass position
    body_quat: Buffer, // [MAX_BODIES] float4 — orientation quaternion (x,y,z,w)
    body_vel: Buffer, // [MAX_BODIES] float3 — linear velocity
    body_omega: Buffer, // [MAX_BODIES] float3 — angular velocity
    body_inv_mass: Buffer, // [MAX_BODIES] float — 1/mass (0 = static/fixed)
    body_inertia: Buffer, // [MAX_BODIES] float3 — diagonal inertia tensor
    body_inv_inertia: Buffer, // [MAX_BODIES] float3 — 1/inertia per axis
    body_half: Buffer, // [MAX_BODIES] float3 — half-extents, derived from first geom on upload
    body_vert_start: Buffer, // [MAX_BODIES] uint — first vertex index, derived from first geom on upload
    body_vert_count: Buffer, // [MAX_BODIES] uint — vertex count, derived from first geom on upload

    // ---- GPU Geom buffers (collision shapes, each linked to a body via geom_body_idx) ----
    geom_type: Buffer, // [MAX_GEOMS] int — shape type: 1=sphere, 2=box, 5=mesh
    geom_body_idx: Buffer, // [MAX_GEOMS] uint — which body this geom belongs to
    geom_local_pos: Buffer, // [MAX_GEOMS] float3 — offset from body center (local frame)
    geom_local_quat: Buffer, // [MAX_GEOMS] float4 — rotation relative to body (local frame)
    geom_data: Buffer, // [MAX_GEOMS*7] float — shape params: box=[half_x,half_y,half_z,0,0,0,0]
    geom_friction: Buffer, // [MAX_GEOMS] float — Coulomb friction coefficient

    // ---- GPU Broad phase ----
    body_aabb_min: Buffer, // [MAX_BODIES] float3 — AABB min, computed each frame from pos+quat+half
    body_aabb_max: Buffer, // [MAX_BODIES] float3 — AABB max
    collision_pairs: Buffer, // [MAX_COLLISION_PAIRS*2] uint — pairs of body indices from broad phase

    // ---- GPU Contact buffers (written by collision, read by solver) ----
    contact_pos: Buffer, // [MAX_CONTACTS] float3 — contact point in world space
    contact_normal: Buffer, // [MAX_CONTACTS] float3 — contact normal direction
    contact_penetration: Buffer, // [MAX_CONTACTS] float — penetration depth
    contact_body_a: Buffer, // [MAX_CONTACTS] uint — body index A
    contact_body_b: Buffer, // [MAX_CONTACTS] uint — body index B
    contact_lambda_n: Buffer, // [MAX_CONTACTS] float — PGS accumulated normal impulse

    // ---- GPU Counters ----
    atomic_counters: Buffer, // [4] uint — [0]=num_pairs, [1]=num_contacts

    // ---- CPU staging — render ----
    s_vertices: []f32, // staging for vertices
    s_colors: []f32, // staging for colors
    s_indices: []u32, // staging for indices
    s_orig_verts: []f32, // staging for original_vertices

    // ---- CPU staging — body ----
    s_body_pos: []f32,
    s_body_quat: []f32,
    s_body_vel: []f32,
    s_body_omega: []f32,
    s_body_inv_mass: []f32,
    s_body_inertia: []f32,
    s_body_inv_inertia: []f32,
    s_body_vert_start: []u32, // derived from geom in upload()
    s_body_vert_count: []u32, // derived from geom in upload()

    // ---- CPU staging — geom (owns vertices, linked to body) ----
    s_geom_type: []i32,
    s_geom_body_idx: []u32, // geom → body mapping
    s_geom_local_pos: []f32,
    s_geom_local_quat: []f32,
    s_geom_data: []f32, // shape params, box: half-extents in [0..2]
    s_geom_friction: []f32,
    s_geom_vert_start: []u32, // first vertex index in vertices/orig_verts (geom owns these)
    s_geom_vert_count: []u32, // number of vertices belonging to this geom

    num_vertices: u32,
    num_triangles: u32,
    num_bodies: u32,
    num_geoms: u32,

    pub fn init(vk: *Vulkan, alloc: std.mem.Allocator) !Data {
        return Data{
            .vk = vk,
            .alloc = alloc,

            .vertices = try vk.createBuffer(MAX_VERTICES * @sizeOf([3]f32), VERTEX),
            .colors = try vk.createBuffer(MAX_VERTICES * @sizeOf([3]f32), VERTEX),
            .indices = try vk.createBuffer(MAX_TRIANGLES * 3 * @sizeOf(u32), INDEX),
            .original_vertices = try vk.createBuffer(MAX_VERTICES * @sizeOf([3]f32), STORAGE),

            .body_pos = try vk.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .body_quat = try vk.createBuffer(MAX_BODIES * @sizeOf([4]f32), STORAGE),
            .body_vel = try vk.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .body_omega = try vk.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .body_inv_mass = try vk.createBuffer(MAX_BODIES * @sizeOf(f32), STORAGE),
            .body_inertia = try vk.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .body_inv_inertia = try vk.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .body_half = try vk.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .body_vert_start = try vk.createBuffer(MAX_BODIES * @sizeOf(u32), STORAGE),
            .body_vert_count = try vk.createBuffer(MAX_BODIES * @sizeOf(u32), STORAGE),

            .geom_type = try vk.createBuffer(MAX_GEOMS * @sizeOf(i32), STORAGE),
            .geom_body_idx = try vk.createBuffer(MAX_GEOMS * @sizeOf(u32), STORAGE),
            .geom_local_pos = try vk.createBuffer(MAX_GEOMS * @sizeOf([3]f32), STORAGE),
            .geom_local_quat = try vk.createBuffer(MAX_GEOMS * @sizeOf([4]f32), STORAGE),
            .geom_data = try vk.createBuffer(MAX_GEOMS * @sizeOf([7]f32), STORAGE),
            .geom_friction = try vk.createBuffer(MAX_GEOMS * @sizeOf(f32), STORAGE),

            .body_aabb_min = try vk.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .body_aabb_max = try vk.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .collision_pairs = try vk.createBuffer(MAX_COLLISION_PAIRS * 2 * @sizeOf(u32), STORAGE),

            .contact_pos = try vk.createBuffer(MAX_CONTACTS * @sizeOf([3]f32), STORAGE),
            .contact_normal = try vk.createBuffer(MAX_CONTACTS * @sizeOf([3]f32), STORAGE),
            .contact_penetration = try vk.createBuffer(MAX_CONTACTS * @sizeOf(f32), STORAGE),
            .contact_body_a = try vk.createBuffer(MAX_CONTACTS * @sizeOf(u32), STORAGE),
            .contact_body_b = try vk.createBuffer(MAX_CONTACTS * @sizeOf(u32), STORAGE),
            .contact_lambda_n = try vk.createBuffer(MAX_CONTACTS * @sizeOf(f32), STORAGE),

            .atomic_counters = try vk.createBuffer(4 * @sizeOf(u32), STORAGE),

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
            .s_geom_local_pos = try alloc.alloc(f32, MAX_GEOMS * 3),
            .s_geom_local_quat = try alloc.alloc(f32, MAX_GEOMS * 4),
            .s_geom_data = try alloc.alloc(f32, MAX_GEOMS * 7),
            .s_geom_friction = try alloc.alloc(f32, MAX_GEOMS),
            .s_geom_vert_start = try alloc.alloc(u32, MAX_GEOMS),
            .s_geom_vert_count = try alloc.alloc(u32, MAX_GEOMS),

            .num_vertices = 0,
            .num_triangles = 0,
            .num_bodies = 0,
            .num_geoms = 0,
        };
    }

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

        // Body
        const b3 = bi * 3;
        self.s_body_pos[b3..][0..3].* = center;
        self.s_body_quat[bi * 4 ..][0..4].* = .{ 0, 0, 0, 1 }; // (x,y,z,w) identity
        self.s_body_vel[b3..][0..3].* = .{ 0, 0, 0 };
        self.s_body_omega[b3..][0..3].* = .{ 0, 0, 0 };
        self.s_body_inv_mass[bi] = if (mass > 0) 1.0 / mass else 0;
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

        // Geom (type 2 = box)
        self.s_geom_type[gi] = 2;
        self.s_geom_body_idx[gi] = bi;
        self.s_geom_local_pos[gi * 3 ..][0..3].* = .{ 0, 0, 0 };
        self.s_geom_local_quat[gi * 4 ..][0..4].* = .{ 1, 0, 0, 0 };
        self.s_geom_data[gi * 7 ..][0..7].* = .{ half[0], half[1], half[2], 0, 0, 0, 0 };
        self.s_geom_friction[gi] = 0.5;
        self.s_geom_vert_start[gi] = vs;
        self.s_geom_vert_count[gi] = 8;
        self.num_geoms += 1;

        return bi;
    }

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
        try v.uploadSlice(self.geom_type, i32, self.s_geom_type[0..self.num_geoms]);
        try v.uploadSlice(self.geom_body_idx, u32, self.s_geom_body_idx[0..self.num_geoms]);
        try v.uploadSlice(self.geom_local_pos, f32, self.s_geom_local_pos[0 .. self.num_geoms * 3]);
        try v.uploadSlice(self.geom_local_quat, f32, self.s_geom_local_quat[0 .. self.num_geoms * 4]);
        try v.uploadSlice(self.geom_data, f32, self.s_geom_data[0 .. self.num_geoms * 7]);
        try v.uploadSlice(self.geom_friction, f32, self.s_geom_friction[0..self.num_geoms]);

        // Derive body buffers from first geom per body
        var body_half = try self.alloc.alloc(f32, MAX_BODIES * 3);
        defer self.alloc.free(body_half);
        for (0..self.num_geoms) |gi| {
            const bi = self.s_geom_body_idx[gi];
            body_half[bi * 3 + 0] = self.s_geom_data[gi * 7 + 0];
            body_half[bi * 3 + 1] = self.s_geom_data[gi * 7 + 1];
            body_half[bi * 3 + 2] = self.s_geom_data[gi * 7 + 2];
            self.s_body_vert_start[bi] = self.s_geom_vert_start[gi];
            self.s_body_vert_count[bi] = self.s_geom_vert_count[gi];
        }
        try v.uploadSlice(self.body_half, f32, body_half[0 .. self.num_bodies * 3]);
        try v.uploadSlice(self.body_vert_start, u32, self.s_body_vert_start[0..self.num_bodies]);
        try v.uploadSlice(self.body_vert_count, u32, self.s_body_vert_count[0..self.num_bodies]);
    }

    pub fn gpuMemoryBytes(self: *const Data) usize {
        var total: usize = 0;
        inline for (@typeInfo(Data).@"struct".fields) |field| {
            if (field.type == Buffer) {
                total += @field(self, field.name).size;
            }
        }
        return total;
    }

    pub fn deinit(self: *Data) void {
        inline for (@typeInfo(Data).@"struct".fields) |field| {
            if (field.type == Buffer) {
                self.vk.destroyBuffer(@field(self, field.name));
            } else if (comptime isSlice(field.type)) {
                self.alloc.free(@field(self, field.name));
            }
        }
    }

    fn isSlice(comptime T: type) bool {
        return switch (@typeInfo(T)) {
            .pointer => |p| p.size == .slice,
            else => false,
        };
    }
};
