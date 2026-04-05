const std = @import("std");
const Vulkan = @import("vulkan.zig").Vulkan;
const Buffer = Vulkan.Buffer;

const VERTEX = Vulkan.USAGE_VERTEX;
const INDEX = Vulkan.USAGE_INDEX;
const STORAGE = Vulkan.USAGE_STORAGE;

pub const MAX_VERTICES: u32 = 10_000;
pub const MAX_TRIANGLES: u32 = 10_000;
pub const MAX_BODIES: u32 = 256;
pub const MAX_CONTACTS: u32 = 4_000;

pub const Data = struct {
    vk: *Vulkan,
    alloc: std.mem.Allocator,

    // Shader binding order matches physics.comp:
    // 0: body_pos, 1: body_quat, 2: body_vel, 3: body_omega,
    // 4: body_inv_mass, 5: body_inv_inertia, 6: body_half,
    // 7: body_vert_start, 8: body_vert_count,
    // 9: vertices, 10: original_vertices,
    // 11: contact_pos, 12: contact_normal, 13: contact_pen,
    // 14: contact_body_a, 15: contact_body_b, 16: contact_lambda_n,
    // 17: counters

    body_pos: Buffer,           // 0
    body_quat: Buffer,          // 1
    body_vel: Buffer,           // 2
    body_omega: Buffer,         // 3
    body_inv_mass: Buffer,      // 4
    body_inv_inertia: Buffer,   // 5
    body_half: Buffer,          // 6
    body_vert_start: Buffer,    // 7
    body_vert_count: Buffer,    // 8
    vertices: Buffer,           // 9
    original_vertices: Buffer,  // 10
    contact_pos: Buffer,        // 11
    contact_normal: Buffer,     // 12
    contact_pen: Buffer,        // 13
    contact_body_a: Buffer,     // 14
    contact_body_b: Buffer,     // 15
    contact_lambda_n: Buffer,   // 16
    counters: Buffer,           // 17

    // Render-only (not bound to compute)
    colors: Buffer,
    indices: Buffer,

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
    s_body_inv_inertia: []f32,
    s_body_half: []f32,
    s_body_vert_start: []u32,
    s_body_vert_count: []u32,

    num_vertices: u32,
    num_triangles: u32,
    num_bodies: u32,

    pub fn init(vk: *Vulkan, alloc: std.mem.Allocator) !Data {
        return Data{
            .vk = vk,
            .alloc = alloc,

            .body_pos = try vk.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .body_quat = try vk.createBuffer(MAX_BODIES * @sizeOf([4]f32), STORAGE),
            .body_vel = try vk.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .body_omega = try vk.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .body_inv_mass = try vk.createBuffer(MAX_BODIES * @sizeOf(f32), STORAGE),
            .body_inv_inertia = try vk.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .body_half = try vk.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .body_vert_start = try vk.createBuffer(MAX_BODIES * @sizeOf(u32), STORAGE),
            .body_vert_count = try vk.createBuffer(MAX_BODIES * @sizeOf(u32), STORAGE),
            .vertices = try vk.createBuffer(MAX_VERTICES * @sizeOf([3]f32), VERTEX),
            .original_vertices = try vk.createBuffer(MAX_VERTICES * @sizeOf([3]f32), STORAGE),
            .contact_pos = try vk.createBuffer(MAX_CONTACTS * @sizeOf([3]f32), STORAGE),
            .contact_normal = try vk.createBuffer(MAX_CONTACTS * @sizeOf([3]f32), STORAGE),
            .contact_pen = try vk.createBuffer(MAX_CONTACTS * @sizeOf(f32), STORAGE),
            .contact_body_a = try vk.createBuffer(MAX_CONTACTS * @sizeOf(u32), STORAGE),
            .contact_body_b = try vk.createBuffer(MAX_CONTACTS * @sizeOf(u32), STORAGE),
            .contact_lambda_n = try vk.createBuffer(MAX_CONTACTS * @sizeOf(f32), STORAGE),
            .counters = try vk.createBuffer(4 * @sizeOf(u32), STORAGE),

            .colors = try vk.createBuffer(MAX_VERTICES * @sizeOf([3]f32), VERTEX),
            .indices = try vk.createBuffer(MAX_TRIANGLES * 3 * @sizeOf(u32), INDEX),

            .s_vertices = try alloc.alloc(f32, MAX_VERTICES * 3),
            .s_colors = try alloc.alloc(f32, MAX_VERTICES * 3),
            .s_indices = try alloc.alloc(u32, MAX_TRIANGLES * 3),
            .s_orig_verts = try alloc.alloc(f32, MAX_VERTICES * 3),
            .s_body_pos = try alloc.alloc(f32, MAX_BODIES * 3),
            .s_body_quat = try alloc.alloc(f32, MAX_BODIES * 4),
            .s_body_vel = try alloc.alloc(f32, MAX_BODIES * 3),
            .s_body_omega = try alloc.alloc(f32, MAX_BODIES * 3),
            .s_body_inv_mass = try alloc.alloc(f32, MAX_BODIES),
            .s_body_inv_inertia = try alloc.alloc(f32, MAX_BODIES * 3),
            .s_body_half = try alloc.alloc(f32, MAX_BODIES * 3),
            .s_body_vert_start = try alloc.alloc(u32, MAX_BODIES),
            .s_body_vert_count = try alloc.alloc(u32, MAX_BODIES),

            .num_vertices = 0,
            .num_triangles = 0,
            .num_bodies = 0,
        };
    }

    pub fn addBox(self: *Data, center: [3]f32, half: [3]f32, color: [3]f32, mass: f32) !u32 {
        const vs = self.num_vertices;
        const ts = self.num_triangles;
        const bi = self.num_bodies;

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
        // quat (x,y,z,w) — identity = (0,0,0,1)
        self.s_body_quat[bi * 4 ..][0..4].* = .{ 0, 0, 0, 1 };
        self.s_body_vel[b3..][0..3].* = .{ 0, 0, 0 };
        self.s_body_omega[b3..][0..3].* = .{ 0, 0, 0 };
        self.s_body_inv_mass[bi] = if (mass > 0) 1.0 / mass else 0;
        self.s_body_half[b3..][0..3].* = half;
        self.s_body_vert_start[bi] = vs;
        self.s_body_vert_count[bi] = 8;

        if (mass > 0) {
            const wx = half[0] * 2;
            const wy = half[1] * 2;
            const wz = half[2] * 2;
            self.s_body_inv_inertia[b3 + 0] = 12.0 / (mass * (wy * wy + wz * wz));
            self.s_body_inv_inertia[b3 + 1] = 12.0 / (mass * (wx * wx + wz * wz));
            self.s_body_inv_inertia[b3 + 2] = 12.0 / (mass * (wx * wx + wy * wy));
        } else {
            self.s_body_inv_inertia[b3..][0..3].* = .{ 0, 0, 0 };
        }
        self.num_bodies += 1;

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
        try v.uploadSlice(self.body_inv_inertia, f32, self.s_body_inv_inertia[0 .. self.num_bodies * 3]);
        try v.uploadSlice(self.body_half, f32, self.s_body_half[0 .. self.num_bodies * 3]);
        try v.uploadSlice(self.body_vert_start, u32, self.s_body_vert_start[0..self.num_bodies]);
        try v.uploadSlice(self.body_vert_count, u32, self.s_body_vert_count[0..self.num_bodies]);
    }

    /// Get the 18 physics buffers in binding order for compute pipeline
    pub fn physicsBuffers(self: *const Data) [18]Buffer {
        return .{
            self.body_pos, self.body_quat, self.body_vel, self.body_omega,
            self.body_inv_mass, self.body_inv_inertia, self.body_half,
            self.body_vert_start, self.body_vert_count,
            self.vertices, self.original_vertices,
            self.contact_pos, self.contact_normal, self.contact_pen,
            self.contact_body_a, self.contact_body_b, self.contact_lambda_n,
            self.counters,
        };
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
