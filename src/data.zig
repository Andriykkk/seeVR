const std = @import("std");
const Vulkan = @import("vulkan.zig").Vulkan;
const Buffer = Vulkan.Buffer;
const fs = @import("fs.zig");
const Vec = @import("vec.zig").Vec;
const quickhull3d = @import("quickhull.zig").quickhull3d;


pub const MAX_VERTICES: u32 = 100_000;
pub const MAX_TRIANGLES: u32 = 100_000;
pub const MAX_BODIES: u32 = 1_000;
pub const MAX_GEOMS: u32 = 2_000;
pub const MAX_COLLISION_PAIRS: u32 = 10_000;
pub const MAX_CONTACTS: u32 = 10_000;
pub const MAX_COLLISION_VERTS: u32 = 50_000;
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
    geom_local_pos: Buffer, // [MAX_GEOMS] float3 — offset from body center
    geom_local_quat: Buffer, // [MAX_GEOMS] float4 — rotation relative to body
    geom_data: Buffer, // [MAX_GEOMS] float7 — type-specific (box: half_x/y/z, sphere: radius, ...)
    geom_friction: Buffer, // [MAX_GEOMS] float — Coulomb friction coefficient
    geom_world_pos: Buffer, // [MAX_GEOMS] float3 — cached world position (updated each frame)
    geom_world_quat: Buffer, // [MAX_GEOMS] float4 — cached world orientation
    geom_aabb_min: Buffer, // [MAX_GEOMS] float3 — axis-aligned bounding box min
    geom_aabb_max: Buffer, // [MAX_GEOMS] float3 — axis-aligned bounding box max

    // Broad phase output
    collision_pairs: Buffer, // [MAX_COLLISION_PAIRS*2] uint — pairs of geom indices (flat: a0,b0,a1,b1,...)
    atomic_counters: Buffer, // [4] uint — [0]=num_collision_pairs, [1]=num_contacts, ...

    // Contact: collision results from narrow phase, indexed by contact ID
    contact_pos: Buffer, // [MAX_CONTACTS] float3 — contact point in world space
    contact_normal: Buffer, // [MAX_CONTACTS] float3 — contact normal (A→B direction)
    contact_penetration: Buffer, // [MAX_CONTACTS] float — penetration depth
    contact_geom_a: Buffer, // [MAX_CONTACTS] uint — geom index A
    contact_geom_b: Buffer, // [MAX_CONTACTS] uint — geom index B
    contact_lambda_n: Buffer, // [MAX_CONTACTS] float — accumulated normal impulse
    contact_lambda_t1: Buffer, // [MAX_CONTACTS] float — accumulated tangent impulse 1
    contact_lambda_t2: Buffer, // [MAX_CONTACTS] float — accumulated tangent impulse 2

    // Newton constraint solver
    // Per-body: world-space inverse inertia (3x3 matrix = 9 floats), computed each frame from quat + local inertia
    body_inv_inertia_world: Buffer, // [MAX_BODIES*9] float — world-space 3x3 inverse inertia tensor
    // Per-body: generalized acceleration (6 DOFs: 3 linear + 3 angular), solver working state
    solver_qacc: Buffer, // [MAX_BODIES*6] float — DOF accelerations being solved for
    // Per-body: M*qacc product, cached to avoid recomputation
    solver_Ma: Buffer, // [MAX_BODIES*6] float — mass matrix times qacc
    // Per-constraint (4 per contact: 1 normal + 2 friction + 1 spare):
    // Jacobian: how constraint velocity relates to body DOFs. Each row is 12 floats (6 DOFs × 2 bodies)
    solver_jacobian: Buffer, // [MAX_CONTACTS*3*12] float — J[constraint, dof] (normal + 2 friction per contact)
    // Effective mass diagonal: D = 1/(J * M⁻¹ * Jᵀ), controls how much each constraint can push
    solver_efc_D: Buffer, // [MAX_CONTACTS*3] float — inverse effective mass per constraint
    // Constraint force: the impulse magnitude per constraint, output of the solver
    solver_efc_force: Buffer, // [MAX_CONTACTS*3] float — constraint force/impulse
    // Target constraint acceleration: Baumgarte position correction + restitution + damping
    solver_aref: Buffer, // [MAX_CONTACTS*3] float — reference acceleration per constraint
    // Constraint residual: Jaref = -aref + J*qacc, measures how violated each constraint is
    solver_Jaref: Buffer, // [MAX_CONTACTS*3] float — constraint violation
    // Per-body Hessian: H = M + Jᵀ*D*J (6x6 symmetric, stored as 36 floats)
    solver_hessian: Buffer, // [MAX_BODIES*36] float — 6x6 Hessian per body
    // Cholesky factor of Hessian (lower triangle, 6x6 = 21 unique + 36 stored)
    solver_cholesky: Buffer, // [MAX_BODIES*36] float — Cholesky L of Hessian
    // Gradient of cost function: g = M*qacc - qfrc (6 DOFs per body)
    solver_gradient: Buffer, // [MAX_BODIES*6] float — cost gradient
    // Search direction: descent = -H⁻¹ * gradient (solved via Cholesky)
    solver_search: Buffer, // [MAX_BODIES*6] float — Newton descent direction
    // Constraint force projected back to DOFs: qfrc = Jᵀ * efc_force
    solver_qfrc: Buffer, // [MAX_BODIES*6] float — constraint force in DOF space
    solver_mv: Buffer, // [MAX_BODIES*6] float — M * search (for line search)
    solver_jv: Buffer, // [MAX_CONTACTS] float — J * search (for line search)

    // Collision hull vertices (for mesh geoms, used by MPR/GJK support function)
    collision_verts: Buffer, // [MAX_COLLISION_VERTS] float3 — convex hull vertices

    // BVH (for raytracing + broad phase)
    bvh_nodes_min: Buffer, // [MAX_TRIANGLES*2] float3 — node AABB min
    bvh_nodes_max: Buffer, // [MAX_TRIANGLES*2] float3 — node AABB max
    bvh_nodes_left: Buffer, // [MAX_TRIANGLES*2] uint — left child / first prim
    bvh_nodes_right: Buffer, // [MAX_TRIANGLES*2] uint — right child
    bvh_nodes_count: Buffer, // [MAX_TRIANGLES*2] uint — 0=internal, >0=leaf
    bvh_nodes_parent: Buffer, // [MAX_TRIANGLES*2] uint — parent index
    bvh_prim_indices: Buffer, // [MAX_TRIANGLES] uint — sorted triangle indices
    bvh_morton_codes: Buffer, // [MAX_TRIANGLES] uint — morton codes
    bvh_morton_temp: Buffer, // [MAX_TRIANGLES] uint — temp for radix sort
    bvh_sort_indices: Buffer, // [MAX_TRIANGLES] uint — sort indices
    bvh_sort_temp: Buffer, // [MAX_TRIANGLES] uint — temp for radix sort
    bvh_tri_centroids: Buffer, // [MAX_TRIANGLES] float3 — triangle centroids
    bvh_flags: Buffer, // [MAX_TRIANGLES] uint — atomic flags for AABB propagation
    bvh_scene_bounds: Buffer, // [2] float3 — scene AABB min/max

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
    s_geom_local_pos: []f32,
    s_geom_local_quat: []f32,
    s_geom_data: []f32,
    s_geom_friction: []f32,
    s_collision_verts: []f32,
    alloc: std.mem.Allocator,

    // Counters
    num_vertices: u32,
    num_triangles: u32,
    num_bodies: u32,
    num_geoms: u32,
    num_collision_verts: u32,

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
            .geom_local_pos = try vk_ctx.createBuffer(MAX_GEOMS * @sizeOf([3]f32), STORAGE),
            .geom_local_quat = try vk_ctx.createBuffer(MAX_GEOMS * @sizeOf([4]f32), STORAGE),
            .geom_data = try vk_ctx.createBuffer(MAX_GEOMS * @sizeOf([7]f32), STORAGE),
            .geom_friction = try vk_ctx.createBuffer(MAX_GEOMS * @sizeOf(f32), STORAGE),
            .geom_world_pos = try vk_ctx.createBuffer(MAX_GEOMS * @sizeOf([3]f32), STORAGE),
            .geom_world_quat = try vk_ctx.createBuffer(MAX_GEOMS * @sizeOf([4]f32), STORAGE),
            .geom_aabb_min = try vk_ctx.createBuffer(MAX_GEOMS * @sizeOf([3]f32), STORAGE),
            .geom_aabb_max = try vk_ctx.createBuffer(MAX_GEOMS * @sizeOf([3]f32), STORAGE),
            .collision_pairs = try vk_ctx.createBuffer(MAX_COLLISION_PAIRS * 2 * @sizeOf(u32), STORAGE),
            .atomic_counters = try vk_ctx.createBuffer(4 * @sizeOf(u32), STORAGE),
            .contact_pos = try vk_ctx.createBuffer(MAX_CONTACTS * @sizeOf([3]f32), STORAGE),
            .contact_normal = try vk_ctx.createBuffer(MAX_CONTACTS * @sizeOf([3]f32), STORAGE),
            .contact_penetration = try vk_ctx.createBuffer(MAX_CONTACTS * @sizeOf(f32), STORAGE),
            .contact_geom_a = try vk_ctx.createBuffer(MAX_CONTACTS * @sizeOf(u32), STORAGE),
            .contact_geom_b = try vk_ctx.createBuffer(MAX_CONTACTS * @sizeOf(u32), STORAGE),
            .contact_lambda_n = try vk_ctx.createBuffer(MAX_CONTACTS * @sizeOf(f32), STORAGE),
            .contact_lambda_t1 = try vk_ctx.createBuffer(MAX_CONTACTS * @sizeOf(f32), STORAGE),
            .contact_lambda_t2 = try vk_ctx.createBuffer(MAX_CONTACTS * @sizeOf(f32), STORAGE),
            .body_inv_inertia_world = try vk_ctx.createBuffer(MAX_BODIES * 9 * @sizeOf(f32), STORAGE),
            .solver_qacc = try vk_ctx.createBuffer(MAX_BODIES * 6 * @sizeOf(f32), STORAGE),
            .solver_Ma = try vk_ctx.createBuffer(MAX_BODIES * 6 * @sizeOf(f32), STORAGE),
            .solver_jacobian = try vk_ctx.createBuffer(MAX_CONTACTS * 3 * 12 * @sizeOf(f32), STORAGE),
            .solver_efc_D = try vk_ctx.createBuffer(MAX_CONTACTS * 3 * @sizeOf(f32), STORAGE),
            .solver_efc_force = try vk_ctx.createBuffer(MAX_CONTACTS * 3 * @sizeOf(f32), STORAGE),
            .solver_aref = try vk_ctx.createBuffer(MAX_CONTACTS * 3 * @sizeOf(f32), STORAGE),
            .solver_Jaref = try vk_ctx.createBuffer(MAX_CONTACTS * 3 * @sizeOf(f32), STORAGE),
            .solver_hessian = try vk_ctx.createBuffer(MAX_BODIES * 36 * @sizeOf(f32), STORAGE),
            .solver_cholesky = try vk_ctx.createBuffer(MAX_BODIES * 36 * @sizeOf(f32), STORAGE),
            .solver_gradient = try vk_ctx.createBuffer(MAX_BODIES * 6 * @sizeOf(f32), STORAGE),
            .solver_search = try vk_ctx.createBuffer(MAX_BODIES * 6 * @sizeOf(f32), STORAGE),
            .solver_qfrc = try vk_ctx.createBuffer(MAX_BODIES * 6 * @sizeOf(f32), STORAGE),
            .solver_mv = try vk_ctx.createBuffer(MAX_BODIES * 6 * @sizeOf(f32), STORAGE),
            .solver_jv = try vk_ctx.createBuffer(MAX_CONTACTS * @sizeOf(f32), STORAGE),
            .collision_verts = try vk_ctx.createBuffer(MAX_COLLISION_VERTS * @sizeOf([3]f32), STORAGE),

            .bvh_nodes_min = try vk_ctx.createBuffer(MAX_TRIANGLES * 2 * @sizeOf([3]f32), STORAGE),
            .bvh_nodes_max = try vk_ctx.createBuffer(MAX_TRIANGLES * 2 * @sizeOf([3]f32), STORAGE),
            .bvh_nodes_left = try vk_ctx.createBuffer(MAX_TRIANGLES * 2 * @sizeOf(u32), STORAGE),
            .bvh_nodes_right = try vk_ctx.createBuffer(MAX_TRIANGLES * 2 * @sizeOf(u32), STORAGE),
            .bvh_nodes_count = try vk_ctx.createBuffer(MAX_TRIANGLES * 2 * @sizeOf(u32), STORAGE),
            .bvh_nodes_parent = try vk_ctx.createBuffer(MAX_TRIANGLES * 2 * @sizeOf(u32), STORAGE),
            .bvh_prim_indices = try vk_ctx.createBuffer(MAX_TRIANGLES * @sizeOf(u32), STORAGE),
            .bvh_morton_codes = try vk_ctx.createBuffer(MAX_TRIANGLES * @sizeOf(u32), STORAGE),
            .bvh_morton_temp = try vk_ctx.createBuffer(MAX_TRIANGLES * @sizeOf(u32), STORAGE),
            .bvh_sort_indices = try vk_ctx.createBuffer(MAX_TRIANGLES * @sizeOf(u32), STORAGE),
            .bvh_sort_temp = try vk_ctx.createBuffer(MAX_TRIANGLES * @sizeOf(u32), STORAGE),
            .bvh_tri_centroids = try vk_ctx.createBuffer(MAX_TRIANGLES * @sizeOf([3]f32), STORAGE),
            .bvh_flags = try vk_ctx.createBuffer(MAX_TRIANGLES * @sizeOf(u32), STORAGE),
            .bvh_scene_bounds = try vk_ctx.createBuffer(2 * @sizeOf([3]f32), STORAGE),

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
            .s_collision_verts = try alloc.alloc(f32, MAX_COLLISION_VERTS * 3),

            .num_vertices = 0,
            .num_triangles = 0,
            .num_bodies = 0,
            .num_geoms = 0,
            .num_collision_verts = 0,
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
        self.s_geom_local_pos[gi * 3 ..][0..3].* = .{ 0, 0, 0 };
        self.s_geom_local_quat[gi * 4 ..][0..4].* = .{ 1, 0, 0, 0 };
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
        self.s_geom_local_pos[gi * 3 ..][0..3].* = .{ 0, 0, 0 };
        self.s_geom_local_quat[gi * 4 ..][0..4].* = .{ 1, 0, 0, 0 };
        self.s_geom_data[gi * 7 ..][0..7].* = .{ radius, 0, 0, 0, 0, 0, 0 };
        self.s_geom_friction[gi] = 0.5;
        self.num_geoms += 1;

        return bi;
    }

    /// Load an OBJ mesh file, center it, scale it, create body + geom
    pub fn addMesh(self: *Data, filename: [*:0]const u8, center: [3]f32, color: [3]f32, mass: f32, scale: f32) !u32 {
        // Read OBJ file
        const file_data = try fs.readFile(filename, self.alloc);
        defer self.alloc.free(file_data);

        // Parse vertices and faces
        var verts = Vec([3]f32).init(self.alloc);
        defer verts.deinit();
        var faces = Vec([3]u32).init(self.alloc);
        defer faces.deinit();

        var line_start: usize = 0;
        for (file_data, 0..) |byte, i| {
            if (byte == '\n' or i == file_data.len - 1) {
                const line = file_data[line_start .. if (byte == '\n') i else i + 1];
                line_start = i + 1;

                if (line.len < 2) continue;

                if (line[0] == 'v' and line[1] == ' ') {
                    // Parse vertex: "v x y z"
                    var it = std.mem.splitScalar(u8, line[2..], ' ');
                    var v: [3]f32 = undefined;
                    var vi: u32 = 0;
                    while (it.next()) |tok| {
                        if (tok.len == 0) continue;
                        if (vi < 3) {
                            v[vi] = std.fmt.parseFloat(f32, tok) catch 0;
                            vi += 1;
                        }
                    }
                    if (vi == 3) try verts.push(v);
                } else if (line[0] == 'f' and line[1] == ' ') {
                    // Parse face: "f v1 v2 v3" or "f v1/vt1 v2/vt2 v3/vt3" — triangulate fan
                    var it = std.mem.splitScalar(u8, line[2..], ' ');
                    var idx: [32]u32 = undefined;
                    var count: u32 = 0;
                    while (it.next()) |tok| {
                        if (tok.len == 0) continue;
                        // Take first number before '/'
                        var slash_it = std.mem.splitScalar(u8, tok, '/');
                        const num_str = slash_it.next() orelse continue;
                        const val = std.fmt.parseInt(u32, num_str, 10) catch continue;
                        if (count < 32) {
                            idx[count] = val - 1; // OBJ is 1-indexed
                            count += 1;
                        }
                    }
                    // Triangulate as fan
                    var fi: u32 = 1;
                    while (fi + 1 < count) : (fi += 1) {
                        try faces.push(.{ idx[0], idx[fi], idx[fi + 1] });
                    }
                }
            }
        }

        if (verts.len == 0) return error.EmptyMesh;

        // Center and scale vertices
        var centroid = [3]f32{ 0, 0, 0 };
        for (verts.items[0..verts.len]) |v| {
            centroid[0] += v[0];
            centroid[1] += v[1];
            centroid[2] += v[2];
        }
        const n: f32 = @floatFromInt(verts.len);
        centroid[0] /= n;
        centroid[1] /= n;
        centroid[2] /= n;

        for (verts.items[0..verts.len]) |*v| {
            v[0] = (v[0] - centroid[0]) * scale;
            v[1] = (v[1] - centroid[1]) * scale;
            v[2] = (v[2] - centroid[2]) * scale;
        }

        // Write to staging
        const vs = self.num_vertices;
        const ts = self.num_triangles;
        const bi = self.num_bodies;
        const gi = self.num_geoms;
        const num_v: u32 = verts.len;
        const num_t: u32 = faces.len;

        for (0..num_v) |i| {
            const b = (vs + @as(u32, @intCast(i))) * 3;
            for (0..3) |j| {
                self.s_orig_verts[b + j] = verts.items[i][j];
                self.s_vertices[b + j] = center[j] + verts.items[i][j];
                self.s_colors[b + j] = color[j];
            }
        }
        for (0..num_t) |i| {
            const b = (ts + @as(u32, @intCast(i))) * 3;
            for (0..3) |j| self.s_indices[b + j] = vs + faces.items[i][j];
        }
        self.num_vertices += num_v;
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
            // Rough inertia from AABB
            var aabb_min = [3]f32{ 1e10, 1e10, 1e10 };
            var aabb_max = [3]f32{ -1e10, -1e10, -1e10 };
            for (verts.items[0..verts.len]) |v| {
                for (0..3) |j| {
                    aabb_min[j] = @min(aabb_min[j], v[j]);
                    aabb_max[j] = @max(aabb_max[j], v[j]);
                }
            }
            const sx = aabb_max[0] - aabb_min[0];
            const sy = aabb_max[1] - aabb_min[1];
            const sz = aabb_max[2] - aabb_min[2];
            self.s_body_inertia[b3 + 0] = mass / 12.0 * (sy * sy + sz * sz);
            self.s_body_inertia[b3 + 1] = mass / 12.0 * (sx * sx + sz * sz);
            self.s_body_inertia[b3 + 2] = mass / 12.0 * (sx * sx + sy * sy);
            self.s_body_inv_inertia[b3 + 0] = 1.0 / self.s_body_inertia[b3 + 0];
            self.s_body_inv_inertia[b3 + 1] = 1.0 / self.s_body_inertia[b3 + 1];
            self.s_body_inv_inertia[b3 + 2] = 1.0 / self.s_body_inertia[b3 + 2];
        } else {
            self.s_body_inertia[b3..][0..3].* = .{ 0, 0, 0 };
            self.s_body_inv_inertia[b3..][0..3].* = .{ 0, 0, 0 };
        }
        self.num_bodies += 1;

        // Convex hull for collision
        var hull_idx = try quickhull3d(verts.items[0..verts.len], self.alloc);
        defer hull_idx.deinit();
        const cs = self.num_collision_verts;
        const num_hull: u32 = hull_idx.len;
        for (0..num_hull) |i| {
            const hi = hull_idx.items[i];
            const b = (cs + @as(u32, @intCast(i))) * 3;
            for (0..3) |j| {
                self.s_collision_verts[b + j] = verts.items[hi][j];
            }
        }
        self.num_collision_verts += num_hull;

        // Geom (MESH type = 5): data = [hull_start, hull_count, 0, 0, 0, 0, 0]
        self.s_geom_type[gi] = 5;
        self.s_geom_body_idx[gi] = bi;
        self.s_geom_local_pos[gi * 3 ..][0..3].* = .{ 0, 0, 0 };
        self.s_geom_local_quat[gi * 4 ..][0..4].* = .{ 1, 0, 0, 0 };
        self.s_geom_data[gi * 7 ..][0..7].* = .{ @floatFromInt(cs), @floatFromInt(num_hull), 0, 0, 0, 0, 0 };
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
        try v.uploadSlice(self.geom_local_pos, f32, self.s_geom_local_pos[0 .. self.num_geoms * 3]);
        try v.uploadSlice(self.geom_local_quat, f32, self.s_geom_local_quat[0 .. self.num_geoms * 4]);
        try v.uploadSlice(self.geom_data, f32, self.s_geom_data[0 .. self.num_geoms * 7]);
        try v.uploadSlice(self.geom_friction, f32, self.s_geom_friction[0..self.num_geoms]);
        if (self.num_collision_verts > 0)
            try v.uploadSlice(self.collision_verts, f32, self.s_collision_verts[0 .. self.num_collision_verts * 3]);
    }

    /// Total GPU memory allocated by all buffers
    pub fn gpuMemoryBytes(self: *const Data) usize {
        return self.vertices.size + self.colors.size + self.indices.size + self.original_vertices.size +
            self.body_pos.size + self.body_quat.size + self.body_vel.size + self.body_omega.size +
            self.body_inv_mass.size + self.body_inertia.size + self.body_inv_inertia.size +
            self.body_vert_start.size + self.body_vert_count.size +
            self.geom_type.size + self.geom_body_idx.size + self.geom_local_pos.size + self.geom_local_quat.size +
            self.geom_data.size + self.geom_friction.size +
            self.geom_world_pos.size + self.geom_world_quat.size + self.geom_aabb_min.size + self.geom_aabb_max.size +
            self.contact_pos.size + self.contact_normal.size + self.contact_penetration.size +
            self.collision_pairs.size + self.atomic_counters.size +
            self.contact_geom_a.size + self.contact_geom_b.size +
            self.contact_lambda_n.size + self.contact_lambda_t1.size + self.contact_lambda_t2.size +
            self.body_inv_inertia_world.size +
            self.solver_qacc.size + self.solver_Ma.size + self.solver_jacobian.size +
            self.solver_efc_D.size + self.solver_efc_force.size + self.solver_aref.size +
            self.solver_Jaref.size + self.solver_hessian.size + self.solver_cholesky.size +
            self.solver_gradient.size + self.solver_search.size + self.solver_qfrc.size +
            self.solver_mv.size + self.solver_jv.size +
            self.collision_verts.size +
            self.bvh_nodes_min.size + self.bvh_nodes_max.size +
            self.bvh_nodes_left.size + self.bvh_nodes_right.size +
            self.bvh_nodes_count.size + self.bvh_nodes_parent.size +
            self.bvh_prim_indices.size + self.bvh_morton_codes.size +
            self.bvh_morton_temp.size + self.bvh_sort_indices.size +
            self.bvh_sort_temp.size + self.bvh_tri_centroids.size +
            self.bvh_flags.size + self.bvh_scene_bounds.size;
    }

    pub fn deinit(self: *Data) void {
        const bufs = &[_]*Buffer{
            &self.vertices, &self.colors, &self.indices, &self.original_vertices,
            &self.body_pos, &self.body_quat, &self.body_vel, &self.body_omega,
            &self.body_inv_mass, &self.body_inertia, &self.body_inv_inertia,
            &self.body_vert_start, &self.body_vert_count,
            &self.geom_type, &self.geom_body_idx, &self.geom_local_pos, &self.geom_local_quat,
            &self.geom_data, &self.geom_friction,
            &self.geom_world_pos, &self.geom_world_quat, &self.geom_aabb_min, &self.geom_aabb_max,
            &self.contact_pos, &self.contact_normal, &self.contact_penetration,
            &self.collision_pairs, &self.atomic_counters,
            &self.contact_geom_a, &self.contact_geom_b,
            &self.contact_lambda_n, &self.contact_lambda_t1, &self.contact_lambda_t2,
            &self.body_inv_inertia_world,
            &self.solver_qacc, &self.solver_Ma, &self.solver_jacobian,
            &self.solver_efc_D, &self.solver_efc_force, &self.solver_aref,
            &self.solver_Jaref, &self.solver_hessian, &self.solver_cholesky,
            &self.solver_gradient, &self.solver_search, &self.solver_qfrc,
            &self.solver_mv, &self.solver_jv,
            &self.collision_verts,
            &self.bvh_nodes_min, &self.bvh_nodes_max, &self.bvh_nodes_left,
            &self.bvh_nodes_right, &self.bvh_nodes_count, &self.bvh_nodes_parent,
            &self.bvh_prim_indices, &self.bvh_morton_codes, &self.bvh_morton_temp,
            &self.bvh_sort_indices, &self.bvh_sort_temp, &self.bvh_tri_centroids,
            &self.bvh_flags, &self.bvh_scene_bounds,
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
        self.alloc.free(self.s_geom_local_pos);
        self.alloc.free(self.s_geom_local_quat);
        self.alloc.free(self.s_geom_data);
        self.alloc.free(self.s_geom_friction);
        self.alloc.free(self.s_collision_verts);
    }
};
