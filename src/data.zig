const std = @import("std");
const Vulkan = @import("vulkan.zig").Vulkan;
const Buffer = Vulkan.Buffer;
const build_options = @import("build_options");
const is_raytrace = build_options.raytrace;

const VERTEX = Vulkan.USAGE_VERTEX;
const INDEX = Vulkan.USAGE_INDEX;
const STORAGE = Vulkan.USAGE_STORAGE;

pub const MAX_VERTICES: u32 = 10_000_000;
pub const MAX_TRIANGLES: u32 = 10_000_000;
pub const MAX_BODIES: u32 = 256;
pub const MAX_GEOMS: u32 = 256;
pub const MAX_CONTACTS: u32 = 4_000;
pub const MAX_COLLISION_PAIRS: u32 = 10_000;
pub const MAX_HULL_VERTS: u32 = 10_000_000;
pub const MAX_MATERIALS: u32 = 256;

pub const SUPPORT_NUM_AXES: u32 = 3;
pub const SUPPORT_NUM_BINS: u32 = 64;
pub const SUPPORT_LOOKUP_THRESHOLD: u32 = 32;
pub const SUPPORT_ENTRY_SIZE: u32 = SUPPORT_NUM_AXES * SUPPORT_NUM_BINS; // 192 per geom

pub const WORKGROUP_SIZE: u32 = 256;
pub const MAX_HULL_WG: u32 = (MAX_HULL_VERTS + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
pub const MAX_WORKGROUPS: u32 = (MAX_TRIANGLES + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
// Temp buffer sized for radix sort histogram (largest consumer): MAX_WORKGROUPS × 256 bins × 4 bytes
pub const BVH_TEMP_SIZE: u32 = MAX_WORKGROUPS * 256;

pub const Data = struct {
    vk: *Vulkan,
    alloc: std.mem.Allocator,

    // ---- GPU Render ----
    vertices: Buffer, // [MAX_VERTICES] float3 — world-space, updated by compute
    colors: Buffer, // [MAX_VERTICES] float3 — per-vertex RGB
    indices: Buffer, // [MAX_TRIANGLES*3] uint — triangle indices
    original_vertices: Buffer, // [MAX_VERTICES] float3 — local-space, constant

    // ---- GPU Body (dynamics only) ----
    body_pos: Buffer, // [MAX_BODIES] float3
    body_quat: Buffer, // [MAX_BODIES] float4 (x,y,z,w)
    body_vel: Buffer, // [MAX_BODIES] float3
    body_omega: Buffer, // [MAX_BODIES] float3
    body_inv_mass: Buffer, // [MAX_BODIES] float
    body_inertia: Buffer, // [MAX_BODIES] float3
    body_inv_inertia: Buffer, // [MAX_BODIES] float3
    body_vert_start: Buffer, // [MAX_BODIES] uint — derived from geom
    body_vert_count: Buffer, // [MAX_BODIES] uint — derived from geom

    // ---- GPU Geom (collision shapes) ----
    geom_type: Buffer, // [MAX_GEOMS] int — 2=box, etc
    geom_body_idx: Buffer, // [MAX_GEOMS] uint — owning body
    geom_local_pos: Buffer, // [MAX_GEOMS] float3
    geom_local_quat: Buffer, // [MAX_GEOMS] float4
    // geom_data layout per geom (8 floats):
    //   [0]=hull_vert_start  [1]=hull_vert_count
    //   [2]=unused            [3]=unused
    //   [4]=unused            [5]=unused
    //   [6]=unused            [7]=unused
    geom_data: Buffer, // [MAX_GEOMS*8] float
    geom_friction: Buffer, // [MAX_GEOMS] float
    geom_restitution: Buffer, // [MAX_GEOMS] float

    // ---- GPU Support lookup (for fast support function) ----
    // Per geom: offset into support_lookup buffer. 0 = no lookup (use brute force).
    // Nonzero value N means lookup data starts at support_lookup[N].
    // Lookup stores hull vert indices for 3 axes × 64 bins = 192 uints per geom.
    support_lookup_offset: Buffer, // [MAX_GEOMS] uint
    support_lookup: Buffer, // [MAX_GEOMS * SUPPORT_ENTRY_SIZE] uint — vert indices
    geom_material: Buffer, // [MAX_GEOMS] uint — index into material arrays

    // ---- GPU Materials (for path tracing) ----
    material_albedo: Buffer, // [MAX_MATERIALS] float3 — base color
    material_roughness: Buffer, // [MAX_MATERIALS] float — 0=mirror, 1=diffuse
    material_metallic: Buffer, // [MAX_MATERIALS] float — 0=dielectric, 1=metal
    material_emission: Buffer, // [MAX_MATERIALS] float3 — emissive color/intensity
    material_ior: Buffer, // [MAX_MATERIALS] float — index of refraction (0=opaque, 1.5=glass)

    // ---- GPU Hull data (shared, indexed by geom_data) ----
    hull_verts: Buffer, // [MAX_HULL_VERTS] float3 — local-space convex hull vertices

    // ---- GPU Broad phase ----
    body_aabb_min: Buffer, // [MAX_BODIES] float3
    body_aabb_max: Buffer, // [MAX_BODIES] float3
    collision_pairs: Buffer, // [MAX_COLLISION_PAIRS*2] uint
    aabb_temp_min: Buffer, // [MAX_HULL_WG*2] float3 — partial AABB reduce
    aabb_temp_max: Buffer, // [MAX_HULL_WG*2] float3
    aabb_temp_body: Buffer, // [MAX_HULL_WG*2] uint — body index per partial

    // ---- GPU Contacts ----
    contact_pos: Buffer, // [MAX_CONTACTS] float3
    contact_normal: Buffer, // [MAX_CONTACTS] float3
    contact_penetration: Buffer, // [MAX_CONTACTS] float
    contact_geom_a: Buffer, // [MAX_CONTACTS] uint
    contact_geom_b: Buffer, // [MAX_CONTACTS] uint
    contact_lambda_n: Buffer, // [MAX_CONTACTS] float

    // ---- GPU Counters ----
    atomic_counters: Buffer, // [4] uint — [0]=num_pairs, [1]=num_contacts

    // ---- GPU Raytrace (raytrace only) ----
    tri_geom: if (is_raytrace) Buffer else void, // [MAX_TRIANGLES] uint — geom index per triangle

    // ---- GPU BVH (raytrace only, void when raster) ----
    bvh_aabb_min: if (is_raytrace) Buffer else void,
    bvh_aabb_max: if (is_raytrace) Buffer else void,
    bvh_left: if (is_raytrace) Buffer else void,
    bvh_right: if (is_raytrace) Buffer else void,
    bvh_count: if (is_raytrace) Buffer else void,
    bvh_parent: if (is_raytrace) Buffer else void,
    bvh_prim_indices: if (is_raytrace) Buffer else void,
    bvh_morton: if (is_raytrace) Buffer else void,
    bvh_sort_indices: if (is_raytrace) Buffer else void,
    bvh_sort_temp: if (is_raytrace) Buffer else void,
    bvh_morton_temp: if (is_raytrace) Buffer else void,
    bvh_centroids: if (is_raytrace) Buffer else void,
    bvh_flags: if (is_raytrace) Buffer else void,
    bvh_scene_bounds: if (is_raytrace) Buffer else void,
    bvh_temp: if (is_raytrace) Buffer else void, // shared reduction temp buffer

    // ---- CPU staging — render ----
    s_vertices: []f32,
    s_colors: []f32,
    s_indices: []u32,
    s_orig_verts: []f32,

    // ---- CPU staging — body ----
    s_body_pos: []f32,
    s_body_quat: []f32,
    s_body_vel: []f32,
    s_body_omega: []f32,
    s_body_inv_mass: []f32,
    s_body_inertia: []f32,
    s_body_inv_inertia: []f32,
    s_body_vert_start: []u32,
    s_body_vert_count: []u32,

    // ---- CPU staging — geom ----
    s_geom_type: []i32,
    s_geom_body_idx: []u32,
    s_geom_local_pos: []f32,
    s_geom_local_quat: []f32,
    s_geom_data: []f32,
    s_geom_friction: []f32,
    s_geom_restitution: []f32,
    s_geom_vert_start: []u32,
    s_geom_vert_count: []u32,
    s_geom_tri_start: []u32,
    s_geom_tri_count: []u32,

    // ---- CPU staging — material ----
    s_geom_material: []u32,
    s_material_albedo: []f32,
    s_material_roughness: []f32,
    s_material_metallic: []f32,
    s_material_emission: []f32,
    s_material_ior: []f32,

    // ---- CPU staging — material (physics-only, not uploaded to GPU) ----
    s_material_density: []f32,

    // ---- CPU staging — hull ----
    s_hull_verts: []f32,

    // ---- CPU staging — support lookup ----
    s_support_lookup_offset: []u32,
    s_support_lookup: []u32,
    num_support_entries: u32, // total entries allocated in lookup buffer

    num_vertices: u32,
    num_triangles: u32,
    num_bodies: u32,
    num_geoms: u32,
    num_hull_verts: u32,
    num_materials: u32,

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
            .body_vert_start = try vk.createBuffer(MAX_BODIES * @sizeOf(u32), STORAGE),
            .body_vert_count = try vk.createBuffer(MAX_BODIES * @sizeOf(u32), STORAGE),

            .geom_type = try vk.createBuffer(MAX_GEOMS * @sizeOf(i32), STORAGE),
            .geom_body_idx = try vk.createBuffer(MAX_GEOMS * @sizeOf(u32), STORAGE),
            .geom_local_pos = try vk.createBuffer(MAX_GEOMS * @sizeOf([3]f32), STORAGE),
            .geom_local_quat = try vk.createBuffer(MAX_GEOMS * @sizeOf([4]f32), STORAGE),
            .geom_data = try vk.createBuffer(MAX_GEOMS * 8 * @sizeOf(f32), STORAGE),
            .geom_friction = try vk.createBuffer(MAX_GEOMS * @sizeOf(f32), STORAGE),
            .geom_restitution = try vk.createBuffer(MAX_GEOMS * @sizeOf(f32), STORAGE),
            .geom_material = try vk.createBuffer(MAX_GEOMS * @sizeOf(u32), STORAGE),
            .support_lookup_offset = try vk.createBuffer(MAX_GEOMS * @sizeOf(u32), STORAGE),
            .support_lookup = try vk.createBuffer(MAX_GEOMS * SUPPORT_ENTRY_SIZE * @sizeOf(u32), STORAGE),

            .material_albedo = try vk.createBuffer(MAX_MATERIALS * @sizeOf([3]f32), STORAGE),
            .material_roughness = try vk.createBuffer(MAX_MATERIALS * @sizeOf(f32), STORAGE),
            .material_metallic = try vk.createBuffer(MAX_MATERIALS * @sizeOf(f32), STORAGE),
            .material_emission = try vk.createBuffer(MAX_MATERIALS * @sizeOf([3]f32), STORAGE),
            .material_ior = try vk.createBuffer(MAX_MATERIALS * @sizeOf(f32), STORAGE),

            .hull_verts = try vk.createBuffer(MAX_HULL_VERTS * @sizeOf([3]f32), STORAGE),

            .body_aabb_min = try vk.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .body_aabb_max = try vk.createBuffer(MAX_BODIES * @sizeOf([3]f32), STORAGE),
            .collision_pairs = try vk.createBuffer(MAX_COLLISION_PAIRS * 2 * @sizeOf(u32), STORAGE),
            .aabb_temp_min = try vk.createBuffer(MAX_HULL_WG * 2 * @sizeOf([3]f32), STORAGE),
            .aabb_temp_max = try vk.createBuffer(MAX_HULL_WG * 2 * @sizeOf([3]f32), STORAGE),
            .aabb_temp_body = try vk.createBuffer(MAX_HULL_WG * 2 * @sizeOf(u32), STORAGE),

            .contact_pos = try vk.createBuffer(MAX_CONTACTS * @sizeOf([3]f32), STORAGE),
            .contact_normal = try vk.createBuffer(MAX_CONTACTS * @sizeOf([3]f32), STORAGE),
            .contact_penetration = try vk.createBuffer(MAX_CONTACTS * @sizeOf(f32), STORAGE),
            .contact_geom_a = try vk.createBuffer(MAX_CONTACTS * @sizeOf(u32), STORAGE),
            .contact_geom_b = try vk.createBuffer(MAX_CONTACTS * @sizeOf(u32), STORAGE),
            .contact_lambda_n = try vk.createBuffer(MAX_CONTACTS * @sizeOf(f32), STORAGE),

            .atomic_counters = try vk.createBuffer(4 * @sizeOf(u32), STORAGE),

            .tri_geom = if (is_raytrace) try vk.createBuffer(MAX_TRIANGLES * @sizeOf(u32), STORAGE) else {},

            .bvh_aabb_min = if (is_raytrace) try vk.createBuffer(MAX_TRIANGLES * 2 * @sizeOf([3]f32), STORAGE) else {},
            .bvh_aabb_max = if (is_raytrace) try vk.createBuffer(MAX_TRIANGLES * 2 * @sizeOf([3]f32), STORAGE) else {},
            .bvh_left = if (is_raytrace) try vk.createBuffer(MAX_TRIANGLES * 2 * @sizeOf(u32), STORAGE) else {},
            .bvh_right = if (is_raytrace) try vk.createBuffer(MAX_TRIANGLES * 2 * @sizeOf(u32), STORAGE) else {},
            .bvh_count = if (is_raytrace) try vk.createBuffer(MAX_TRIANGLES * 2 * @sizeOf(u32), STORAGE) else {},
            .bvh_parent = if (is_raytrace) try vk.createBuffer(MAX_TRIANGLES * 2 * @sizeOf(u32), STORAGE) else {},
            .bvh_prim_indices = if (is_raytrace) try vk.createBuffer(MAX_TRIANGLES * @sizeOf(u32), STORAGE) else {},
            .bvh_morton = if (is_raytrace) try vk.createBuffer(MAX_TRIANGLES * @sizeOf(u32), STORAGE) else {},
            .bvh_sort_indices = if (is_raytrace) try vk.createBuffer(MAX_TRIANGLES * @sizeOf(u32), STORAGE) else {},
            .bvh_sort_temp = if (is_raytrace) try vk.createBuffer(MAX_TRIANGLES * @sizeOf(u32), STORAGE) else {},
            .bvh_morton_temp = if (is_raytrace) try vk.createBuffer(MAX_TRIANGLES * @sizeOf(u32), STORAGE) else {},
            .bvh_centroids = if (is_raytrace) try vk.createBuffer(MAX_TRIANGLES * @sizeOf([3]f32), STORAGE) else {},
            .bvh_flags = if (is_raytrace) try vk.createBuffer(MAX_TRIANGLES * @sizeOf(u32), STORAGE) else {},
            .bvh_scene_bounds = if (is_raytrace) try vk.createBuffer(2 * @sizeOf([3]f32), STORAGE) else {},
            .bvh_temp = if (is_raytrace) try vk.createBuffer(BVH_TEMP_SIZE * @sizeOf(u32), STORAGE) else {},

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
            .s_geom_data = try alloc.alloc(f32, MAX_GEOMS * 8),
            .s_geom_friction = try alloc.alloc(f32, MAX_GEOMS),
            .s_geom_restitution = try alloc.alloc(f32, MAX_GEOMS),
            .s_geom_vert_start = try alloc.alloc(u32, MAX_GEOMS),
            .s_geom_vert_count = try alloc.alloc(u32, MAX_GEOMS),
            .s_geom_tri_start = try alloc.alloc(u32, MAX_GEOMS),
            .s_geom_tri_count = try alloc.alloc(u32, MAX_GEOMS),
            .s_geom_material = try alloc.alloc(u32, MAX_GEOMS),
            .s_material_albedo = try alloc.alloc(f32, MAX_MATERIALS * 3),
            .s_material_roughness = try alloc.alloc(f32, MAX_MATERIALS),
            .s_material_metallic = try alloc.alloc(f32, MAX_MATERIALS),
            .s_material_emission = try alloc.alloc(f32, MAX_MATERIALS * 3),
            .s_material_ior = try alloc.alloc(f32, MAX_MATERIALS),
            .s_material_density = try alloc.alloc(f32, MAX_MATERIALS),
            .s_hull_verts = try alloc.alloc(f32, MAX_HULL_VERTS * 3),
            .s_support_lookup_offset = try alloc.alloc(u32, MAX_GEOMS),
            .s_support_lookup = try alloc.alloc(u32, MAX_GEOMS * SUPPORT_ENTRY_SIZE),
            .num_support_entries = 0,

            .num_vertices = 0,
            .num_triangles = 0,
            .num_bodies = 0,
            .num_geoms = 0,
            .num_hull_verts = 0,
            .num_materials = 0,
        };
    }

    /// Build support lookup table for a geom if its hull vert count exceeds the threshold.
    /// Called after hull verts are set. Uses local-space hull verts.
    fn buildSupportLookup(self: *Data, gi: u32, hull_start: u32, hull_count: u32) void {
        if (hull_count <= SUPPORT_LOOKUP_THRESHOLD) {
            self.s_support_lookup_offset[gi] = 0; // 0 = brute force
            return;
        }

        // Reserve slot starting at index 1 (0 is reserved for "no lookup")
        const entry_start = if (self.num_support_entries == 0) @as(u32, 1) else self.num_support_entries;
        self.s_support_lookup_offset[gi] = entry_start;

        // 3 axes with fixed perpendicular vectors (local space):
        // Axis X: perp1=(0,1,0) perp2=(0,0,1) → angle = atan2(d.z, d.y)
        // Axis Y: perp1=(1,0,0) perp2=(0,0,1) → angle = atan2(d.z, d.x)
        // Axis Z: perp1=(1,0,0) perp2=(0,1,0) → angle = atan2(d.y, d.x)
        const perps = [3][2][3]f32{
            .{ .{ 0, 1, 0 }, .{ 0, 0, 1 } }, // X axis
            .{ .{ 1, 0, 0 }, .{ 0, 0, 1 } }, // Y axis
            .{ .{ 1, 0, 0 }, .{ 0, 1, 0 } }, // Z axis
        };

        for (0..SUPPORT_NUM_AXES) |ax| {
            const p1 = perps[ax][0];
            const p2 = perps[ax][1];

            for (0..SUPPORT_NUM_BINS) |bin| {
                const angle: f32 = @as(f32, @floatFromInt(bin)) / @as(f32, @floatFromInt(SUPPORT_NUM_BINS)) * 2.0 * std.math.pi;
                const cos_a = @cos(angle);
                const sin_a = @sin(angle);
                // Direction on the great circle
                const dx = cos_a * p1[0] + sin_a * p2[0];
                const dy = cos_a * p1[1] + sin_a * p2[1];
                const dz = cos_a * p1[2] + sin_a * p2[2];

                // Brute force find best hull vert for this direction
                var best_dot: f32 = -1e10;
                var best_idx: u32 = 0;
                for (0..hull_count) |vi| {
                    const v = @as(u32, @intCast(vi));
                    const vx = self.s_hull_verts[(hull_start + v) * 3 + 0];
                    const vy = self.s_hull_verts[(hull_start + v) * 3 + 1];
                    const vz = self.s_hull_verts[(hull_start + v) * 3 + 2];
                    const d = dx * vx + dy * vy + dz * vz;
                    if (d > best_dot) {
                        best_dot = d;
                        best_idx = v;
                    }
                }

                const slot = entry_start + @as(u32, @intCast(ax)) * SUPPORT_NUM_BINS + @as(u32, @intCast(bin));
                self.s_support_lookup[slot] = best_idx;
            }
        }

        self.num_support_entries = entry_start + SUPPORT_ENTRY_SIZE;
    }

    /// Add a material. density is kg/m³ (0 = static/infinite mass). Only used for physics, not uploaded to GPU.
    pub fn addMaterial(self: *Data, albedo: [3]f32, roughness: f32, metallic: f32, emission: [3]f32, ior: f32, density: f32) u32 {
        const mi = self.num_materials;
        self.s_material_albedo[mi * 3 + 0] = albedo[0];
        self.s_material_albedo[mi * 3 + 1] = albedo[1];
        self.s_material_albedo[mi * 3 + 2] = albedo[2];
        self.s_material_roughness[mi] = roughness;
        self.s_material_metallic[mi] = metallic;
        self.s_material_emission[mi * 3 + 0] = emission[0];
        self.s_material_emission[mi * 3 + 1] = emission[1];
        self.s_material_emission[mi * 3 + 2] = emission[2];
        self.s_material_ior[mi] = ior;
        self.s_material_density[mi] = density;
        self.num_materials += 1;
        return mi;
    }

    /// Compute signed volume of a closed triangle mesh using the divergence theorem.
    /// Sum of signed tetrahedron volumes (each triangle + origin).
    fn meshVolume(tri_indices: []const u32, vert_data: []const f32, num_tris: u32) f32 {
        var vol: f32 = 0;
        for (0..num_tris) |ti| {
            const a = tri_indices[ti * 3 + 0];
            const b = tri_indices[ti * 3 + 1];
            const cc = tri_indices[ti * 3 + 2];
            const v0 = [3]f32{ vert_data[a * 3], vert_data[a * 3 + 1], vert_data[a * 3 + 2] };
            const v1 = [3]f32{ vert_data[b * 3], vert_data[b * 3 + 1], vert_data[b * 3 + 2] };
            const v2 = [3]f32{ vert_data[cc * 3], vert_data[cc * 3 + 1], vert_data[cc * 3 + 2] };
            // Signed volume of tetrahedron (origin, v0, v1, v2) = dot(v0, cross(v1, v2)) / 6
            const cx = v1[1] * v2[2] - v1[2] * v2[1];
            const cy = v1[2] * v2[0] - v1[0] * v2[2];
            const cz = v1[0] * v2[1] - v1[1] * v2[0];
            vol += v0[0] * cx + v0[1] * cy + v0[2] * cz;
        }
        return @abs(vol) / 6.0;
    }

    pub fn addBox(self: *Data, center: [3]f32, half: [3]f32, color: [3]f32, friction: f32, restitution: f32, material: u32) !u32 {
        const vs = self.num_vertices;
        const ts = self.num_triangles;
        const bi = self.num_bodies;
        const gi = self.num_geoms;

        const box_v = [8][3]f32{
            .{ -half[0], -half[1], -half[2] }, .{ half[0], -half[1], -half[2] },
            .{ half[0], half[1], -half[2] },   .{ -half[0], half[1], -half[2] },
            .{ -half[0], -half[1], half[2] },  .{ half[0], -half[1], half[2] },
            .{ half[0], half[1], half[2] },    .{ -half[0], half[1], half[2] },
        };
        const box_f = [12][3]u32{
            .{ 0, 2, 1 }, .{ 0, 3, 2 }, .{ 5, 7, 4 }, .{ 5, 6, 7 },
            .{ 4, 3, 0 }, .{ 4, 7, 3 }, .{ 1, 6, 5 }, .{ 1, 2, 6 },
            .{ 3, 6, 2 }, .{ 3, 7, 6 }, .{ 4, 1, 5 }, .{ 4, 0, 1 },
        };

        // Render vertices
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

        // Body — mass from density * volume
        const b3 = bi * 3;
        const density = self.s_material_density[material];
        const volume = half[0] * 2 * half[1] * 2 * half[2] * 2;
        const mass = density * volume;
        self.s_body_pos[b3..][0..3].* = center;
        self.s_body_quat[bi * 4 ..][0..4].* = .{ 0, 0, 0, 1 };
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

        // Hull verts — 8 box corners in local space
        const hv = self.num_hull_verts;
        for (0..8) |i| {
            const b = (hv + @as(u32, @intCast(i))) * 3;
            self.s_hull_verts[b + 0] = box_v[i][0];
            self.s_hull_verts[b + 1] = box_v[i][1];
            self.s_hull_verts[b + 2] = box_v[i][2];
        }
        self.num_hull_verts += 8;

        // Geom
        self.s_geom_type[gi] = 2;
        self.s_geom_body_idx[gi] = bi;
        self.s_geom_local_pos[gi * 3 ..][0..3].* = .{ 0, 0, 0 };
        self.s_geom_local_quat[gi * 4 ..][0..4].* = .{ 1, 0, 0, 0 };
        // geom_data[gi*8]: [vert_start, vert_count, 0, 0, 0, 0, 0, 0]
        self.s_geom_data[gi * 8 ..][0..8].* = .{
            @floatFromInt(hv), @floatFromInt(8),
            0,                 0,
            0,                 0,
            0,                 0,
        };
        self.s_geom_friction[gi] = friction;
        self.s_geom_restitution[gi] = restitution;
        self.s_geom_material[gi] = material;
        self.s_geom_vert_start[gi] = vs;
        self.s_geom_vert_count[gi] = 8;
        self.s_geom_tri_start[gi] = ts;
        self.s_geom_tri_count[gi] = 12;
        self.buildSupportLookup(gi, hv, 8);
        self.num_geoms += 1;

        return bi;
    }

    pub fn addSphere(self: *Data, center: [3]f32, radius: f32, color: [3]f32, segments: u32, friction: f32, restitution: f32, material: u32) !u32 {
        const vs = self.num_vertices;
        const ts = self.num_triangles;
        const bi = self.num_bodies;
        const gi = self.num_geoms;
        const pi = 3.14159265;

        // Generate UV sphere vertices
        const num_v = (segments + 1) * (segments + 1);
        const num_t = segments * segments * 2;

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
                const lx = radius * @cos(lon) * sin_lat;
                const ly = radius * cos_lat;
                const lz = radius * @sin(lon) * sin_lat;
                self.s_orig_verts[idx + 0] = lx;
                self.s_orig_verts[idx + 1] = ly;
                self.s_orig_verts[idx + 2] = lz;
                self.s_vertices[idx + 0] = center[0] + lx;
                self.s_vertices[idx + 1] = center[1] + ly;
                self.s_vertices[idx + 2] = center[2] + lz;
                self.s_colors[idx + 0] = color[0];
                self.s_colors[idx + 1] = color[1];
                self.s_colors[idx + 2] = color[2];
            }
        }
        self.num_vertices += num_v;

        // Triangles (2 per grid quad)
        var ti: u32 = 0;
        for (0..segments) |i_| {
            for (0..segments) |j_| {
                const cur = vs + @as(u32, @intCast(i_)) * (segments + 1) + @as(u32, @intCast(j_));
                const nxt = cur + segments + 1;
                const b = (ts + ti) * 3;
                self.s_indices[b + 0] = cur;
                self.s_indices[b + 1] = nxt;
                self.s_indices[b + 2] = cur + 1;
                self.s_indices[b + 3] = cur + 1;
                self.s_indices[b + 4] = nxt;
                self.s_indices[b + 5] = nxt + 1;
                ti += 2;
            }
        }
        self.num_triangles += num_t;

        // Body — mass from density * volume
        const b3 = bi * 3;
        const density = self.s_material_density[material];
        const pi_val: f32 = 3.14159265;
        const volume = (4.0 / 3.0) * pi_val * radius * radius * radius;
        const mass = density * volume;
        self.s_body_pos[b3..][0..3].* = center;
        self.s_body_quat[bi * 4 ..][0..4].* = .{ 0, 0, 0, 1 };
        self.s_body_vel[b3..][0..3].* = .{ 0, 0, 0 };
        self.s_body_omega[b3..][0..3].* = .{ 0, 0, 0 };
        self.s_body_inv_mass[bi] = if (mass > 0) 1.0 / mass else 0;
        if (mass > 0) {
            const inertia = 0.4 * mass * radius * radius;
            self.s_body_inertia[b3..][0..3].* = .{ inertia, inertia, inertia };
            self.s_body_inv_inertia[b3..][0..3].* = .{ 1.0 / inertia, 1.0 / inertia, 1.0 / inertia };
        } else {
            self.s_body_inertia[b3..][0..3].* = .{ 0, 0, 0 };
            self.s_body_inv_inertia[b3..][0..3].* = .{ 0, 0, 0 };
        }
        self.num_bodies += 1;

        // Hull verts — use the same sphere vertices as collision hull
        // For GJK support function, all render verts work as hull
        const hv = self.num_hull_verts;
        for (0..num_v) |i| {
            const sv = (vs + @as(u32, @intCast(i))) * 3;
            const hb = (hv + @as(u32, @intCast(i))) * 3;
            self.s_hull_verts[hb + 0] = self.s_orig_verts[sv + 0];
            self.s_hull_verts[hb + 1] = self.s_orig_verts[sv + 1];
            self.s_hull_verts[hb + 2] = self.s_orig_verts[sv + 2];
        }
        self.num_hull_verts += num_v;

        // Geom
        self.s_geom_type[gi] = 1;
        self.s_geom_body_idx[gi] = bi;
        self.s_geom_local_pos[gi * 3 ..][0..3].* = .{ 0, 0, 0 };
        self.s_geom_local_quat[gi * 4 ..][0..4].* = .{ 1, 0, 0, 0 };
        self.s_geom_data[gi * 8 ..][0..8].* = .{
            @floatFromInt(hv), @floatFromInt(num_v),
            0,                 0,
            0,                 0,
            0,                 0,
        };
        self.s_geom_friction[gi] = friction;
        self.s_geom_restitution[gi] = restitution;
        self.s_geom_material[gi] = material;
        self.s_geom_vert_start[gi] = vs;
        self.s_geom_vert_count[gi] = num_v;
        self.s_geom_tri_start[gi] = ts;
        self.s_geom_tri_count[gi] = num_t;
        self.buildSupportLookup(gi, hv, num_v);
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
        try v.uploadSlice(self.geom_data, f32, self.s_geom_data[0 .. self.num_geoms * 8]);
        try v.uploadSlice(self.geom_friction, f32, self.s_geom_friction[0..self.num_geoms]);
        try v.uploadSlice(self.geom_restitution, f32, self.s_geom_restitution[0..self.num_geoms]);
        try v.uploadSlice(self.geom_material, u32, self.s_geom_material[0..self.num_geoms]);
        if (self.num_materials > 0) {
            try v.uploadSlice(self.material_albedo, f32, self.s_material_albedo[0 .. self.num_materials * 3]);
            try v.uploadSlice(self.material_roughness, f32, self.s_material_roughness[0..self.num_materials]);
            try v.uploadSlice(self.material_metallic, f32, self.s_material_metallic[0..self.num_materials]);
            try v.uploadSlice(self.material_emission, f32, self.s_material_emission[0 .. self.num_materials * 3]);
            try v.uploadSlice(self.material_ior, f32, self.s_material_ior[0..self.num_materials]);
        }
        try v.uploadSlice(self.hull_verts, f32, self.s_hull_verts[0 .. self.num_hull_verts * 3]);
        try v.uploadSlice(self.support_lookup_offset, u32, self.s_support_lookup_offset[0..self.num_geoms]);
        if (self.num_support_entries > 0) {
            try v.uploadSlice(self.support_lookup, u32, self.s_support_lookup[0..self.num_support_entries]);
        }

        // Derive body vert start/count from first geom per body
        for (0..self.num_geoms) |gi| {
            const bi = self.s_geom_body_idx[gi];
            self.s_body_vert_start[bi] = self.s_geom_vert_start[gi];
            self.s_body_vert_count[bi] = self.s_geom_vert_count[gi];
        }
        try v.uploadSlice(self.body_vert_start, u32, self.s_body_vert_start[0..self.num_bodies]);
        try v.uploadSlice(self.body_vert_count, u32, self.s_body_vert_count[0..self.num_bodies]);

        // Build tri_geom lookup (raytrace only): triangle index → geom index
        if (comptime is_raytrace) {
            var tri_geom_staging = try self.alloc.alloc(u32, self.num_triangles);
            defer self.alloc.free(tri_geom_staging);
            for (0..self.num_geoms) |gi| {
                const ts = self.s_geom_tri_start[gi];
                const tc = self.s_geom_tri_count[gi];
                for (ts..ts + tc) |ti| {
                    tri_geom_staging[ti] = @intCast(gi);
                }
            }
            try v.uploadSlice(self.tri_geom, u32, tri_geom_staging[0..self.num_triangles]);
        }
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
