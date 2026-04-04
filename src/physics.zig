const std = @import("std");
const c = @import("c.zig").c;
const Vulkan = @import("vulkan.zig").Vulkan;
const Data = @import("data.zig").Data;

const PhysicsPC = extern struct {
    step: u32,
    count: u32,
    dt: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
};

const ConstraintPC = extern struct {
    num_contacts: u32,
    dt: f32,
};

const NarrowPC = extern struct {
    step: u32,
    count: u32,
    sign0: f32,
    sign1: f32,
};

const SolverPC = extern struct {
    step: u32,
    count: u32,
    dt: f32,
};

pub const Physics = struct {
    vk: *Vulkan,
    physics_pipe: Vulkan.ComputePipeline,
    narrow_pipe: Vulkan.ComputePipeline,
    constraint_pipe: Vulkan.ComputePipeline,
    solver_pipe: Vulkan.ComputePipeline,

    pub fn init(vk: *Vulkan, data: *const Data, allocator: std.mem.Allocator) !Physics {
        // Physics pipeline (transforms, AABBs, broad phase)
        const phys_shader = try vk.getShader("shaders/physics.comp", .compute, allocator);
        const phys_buffers = [22]Vulkan.Buffer{
            data.body_pos,        data.body_quat,         data.body_vel,         data.body_omega,
            data.body_inv_mass,   data.body_inertia,      data.body_inv_inertia, data.body_vert_start,
            data.body_vert_count, data.geom_type,         data.geom_body_idx,    data.geom_local_pos,
            data.geom_local_quat, data.geom_world_pos,    data.geom_world_quat,  data.geom_aabb_min,
            data.geom_aabb_max,   data.geom_data,         data.vertices,         data.collision_pairs,
            data.atomic_counters, data.original_vertices,
        };
        const physics_pipe = try vk.createComputePipeline(phys_shader, 22, &phys_buffers, @sizeOf(PhysicsPC));

        // Narrow phase pipeline (MPR)
        const narrow_shader = try vk.getShader("shaders/narrow_phase.comp", .compute, allocator);
        const narrow_buffers = [13]Vulkan.Buffer{
            data.geom_type,      data.geom_body_idx,   data.geom_world_pos,      data.geom_world_quat,
            data.geom_data,      data.collision_verts, data.collision_pairs,     data.atomic_counters,
            data.contact_pos,    data.contact_normal,  data.contact_penetration, data.contact_geom_a,
            data.contact_geom_b,
        };
        const narrow_pipe = try vk.createComputePipeline(narrow_shader, 13, &narrow_buffers, @sizeOf(NarrowPC));

        // Constraint building pipeline (16 bindings matching constraints.comp)
        const constraint_shader = try vk.getShader("shaders/constraints.comp", .compute, allocator);
        const constraint_buffers = [17]Vulkan.Buffer{
            data.contact_pos,           // 0
            data.contact_normal,        // 1
            data.contact_penetration,   // 2
            data.contact_geom_a,        // 3
            data.contact_geom_b,        // 4
            data.body_pos,              // 5
            data.body_vel,              // 6
            data.body_omega,            // 7
            data.body_inv_mass,         // 8
            data.body_inv_inertia_world, // 9
            data.geom_body_idx,         // 10
            data.geom_friction,         // 11
            data.solver_jacobian,       // 12
            data.solver_aref,           // 13
            data.solver_efc_D,          // 14
            data.solver_efc_force,      // 15
            data.atomic_counters,       // 16
        };
        const constraint_pipe = try vk.createComputePipeline(constraint_shader, 17, &constraint_buffers, @sizeOf(ConstraintPC));

        // Newton solver pipeline (29 bindings matching solver.comp)
        const solver_shader = try vk.getShader("shaders/solver.comp", .compute, allocator);
        const solver_buffers = [29]Vulkan.Buffer{
            data.body_pos,              // 0
            data.body_quat,             // 1
            data.body_vel,              // 2
            data.body_omega,            // 3
            data.body_inv_mass,         // 4
            data.body_inv_inertia,      // 5
            data.body_inv_inertia_world, // 6
            data.geom_body_idx,         // 7
            data.geom_friction,         // 8
            data.contact_pos,           // 9
            data.contact_normal,        // 10
            data.contact_penetration,   // 11
            data.contact_geom_a,        // 12
            data.contact_geom_b,        // 13
            data.solver_qacc,           // 14
            data.solver_jacobian,       // 15
            data.solver_efc_D,          // 16
            data.solver_efc_force,      // 17
            data.solver_aref,           // 18
            data.solver_Jaref,          // 19
            data.solver_hessian,        // 20
            data.solver_cholesky,       // 21
            data.solver_gradient,       // 22
            data.solver_search,         // 23
            data.solver_qfrc,           // 24
            data.atomic_counters,       // 25
            data.solver_mv,             // 26
            data.solver_jv,             // 27
            data.solver_Ma,             // 28
        };
        const solver_pipe = try vk.createComputePipeline(solver_shader, 29, &solver_buffers, @sizeOf(SolverPC));

        return .{ .vk = vk, .physics_pipe = physics_pipe, .narrow_pipe = narrow_pipe, .constraint_pipe = constraint_pipe, .solver_pipe = solver_pipe };
    }

    /// Full physics step: transforms → AABBs → broad phase → narrow phase (MPR)
    pub fn step(self: *Physics, num_bodies: u32, num_geoms: u32, dt: f32, gravity: [3]f32) !void {
        if (num_geoms == 0) return;

        const vk = self.vk;
        const geom_groups = (num_geoms + 255) / 256;

        var cmd: c.VkCommandBuffer = null;
        _ = c.vkAllocateCommandBuffers(vk.device, &c.VkCommandBufferAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = null,
            .commandPool = vk.cmd_pool,
            .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        }, &cmd);

        _ = c.vkBeginCommandBuffer(cmd, &c.VkCommandBufferBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null,
        });

        const barrier = c.VkMemoryBarrier{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = null,
            .srcAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = c.VK_ACCESS_SHADER_READ_BIT,
        };

        // --- Physics pipeline ---
        c.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.physics_pipe.pipeline);
        c.vkCmdBindDescriptorSets(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.physics_pipe.layout, 0, 1, &self.physics_pipe.desc_set, 0, null);

        const gx = gravity[0];
        const gy = gravity[1];
        const gz = gravity[2];
        const body_groups = @max((num_bodies + 255) / 256, 1);

        // Step 6: clear counters
        pushPhysics(cmd, self.physics_pipe.layout, .{ .step = 6, .count = 1, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz });
        c.vkCmdDispatch(cmd, 1, 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // Gravity removed — now part of solver qacc init (step 2)

        // Step 0: update geom transforms
        pushPhysics(cmd, self.physics_pipe.layout, .{ .step = 0, .count = num_geoms, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz });
        c.vkCmdDispatch(cmd, geom_groups, 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // Step 1: compute AABBs
        pushPhysics(cmd, self.physics_pipe.layout, .{ .step = 1, .count = num_geoms, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz });
        c.vkCmdDispatch(cmd, geom_groups, 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // Step 2: broad phase
        pushPhysics(cmd, self.physics_pipe.layout, .{ .step = 2, .count = num_geoms, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz });
        c.vkCmdDispatch(cmd, geom_groups, 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // --- Narrow phase pipeline (MPR) ---
        const max_pairs = num_geoms * (num_geoms - 1) / 2;
        const narrow_groups = @max((max_pairs + 255) / 256, 1);

        c.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.narrow_pipe.pipeline);
        c.vkCmdBindDescriptorSets(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.narrow_pipe.layout, 0, 1, &self.narrow_pipe.desc_set, 0, null);

        // Step 0: narrow phase
        pushNarrow(cmd, self.narrow_pipe.layout, .{ .step = 0, .count = max_pairs, .sign0 = 0, .sign1 = 0 });
        c.vkCmdDispatch(cmd, narrow_groups, 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // Perturbation disabled for now

        // --- Build constraints (friction pyramid, Jacobian, aref, efc_D) ---
        c.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.constraint_pipe.pipeline);
        c.vkCmdBindDescriptorSets(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.constraint_pipe.layout, 0, 1, &self.constraint_pipe.desc_set, 0, null);

        const constraint_pc = ConstraintPC{ .num_contacts = max_pairs, .dt = dt };
        c.vkCmdPushConstants(cmd, self.constraint_pipe.layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(ConstraintPC), @ptrCast(&constraint_pc));
        c.vkCmdDispatch(cmd, @max((max_pairs + 255) / 256, 1), 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // --- Newton solver pipeline ---
        c.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.solver_pipe.pipeline);
        c.vkCmdBindDescriptorSets(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.solver_pipe.layout, 0, 1, &self.solver_pipe.desc_set, 0, null);

        // Step 0: compute world-space inertia (parallel, per body)
        pushSolver(cmd, self.solver_pipe.layout, .{ .step = 0, .count = num_bodies, .dt = dt });
        c.vkCmdDispatch(cmd, body_groups, 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // Step 1: build jacobian + aref + efc_D (parallel, per contact)
        // Use max_pairs as upper bound; shader reads actual count from counters[1]
        pushSolver(cmd, self.solver_pipe.layout, .{ .step = 1, .count = max_pairs, .dt = dt });
        c.vkCmdDispatch(cmd, @max((max_pairs + 255) / 256, 1), 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // Step 2: init qacc to zero (parallel, per body)
        pushSolver(cmd, self.solver_pipe.layout, .{ .step = 2, .count = num_bodies, .dt = dt });
        c.vkCmdDispatch(cmd, body_groups, 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // Step 3: newton iterations (serial, 1 thread — iterates internally)
        pushSolver(cmd, self.solver_pipe.layout, .{ .step = 3, .count = num_bodies, .dt = dt });
        c.vkCmdDispatch(cmd, 1, 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // Step 4: apply qacc to vel/omega (parallel, per body)
        pushSolver(cmd, self.solver_pipe.layout, .{ .step = 4, .count = num_bodies, .dt = dt });
        c.vkCmdDispatch(cmd, body_groups, 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // --- Back to physics pipeline for integration ---
        c.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.physics_pipe.pipeline);
        c.vkCmdBindDescriptorSets(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.physics_pipe.layout, 0, 1, &self.physics_pipe.desc_set, 0, null);

        // Step 4: integrate bodies (pos += vel*dt)
        pushPhysics(cmd, self.physics_pipe.layout, .{ .step = 4, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz });
        c.vkCmdDispatch(cmd, body_groups, 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // Step 5: update render vertices
        pushPhysics(cmd, self.physics_pipe.layout, .{ .step = 5, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz });
        c.vkCmdDispatch(cmd, body_groups, 1, 1);

        // Barrier: compute writes → vertex reads
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1, &barrier, 0, null, 0, null);

        _ = c.vkEndCommandBuffer(cmd);

        const queue = if (vk.compute_queue != null) vk.compute_queue else vk.graphics_queue;
        _ = c.vkQueueSubmit(queue, 1, &c.VkSubmitInfo{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .pWaitDstStageMask = null,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null,
        }, null);
        _ = c.vkQueueWaitIdle(queue);

        c.vkFreeCommandBuffers(vk.device, vk.cmd_pool, 1, &cmd);
    }

    fn pushPhysics(cmd: c.VkCommandBuffer, layout: c.VkPipelineLayout, pc: PhysicsPC) void {
        c.vkCmdPushConstants(cmd, layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PhysicsPC), @ptrCast(&pc));
    }

    fn pushSolver(cmd: c.VkCommandBuffer, layout: c.VkPipelineLayout, pc: SolverPC) void {
        c.vkCmdPushConstants(cmd, layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(SolverPC), @ptrCast(&pc));
    }

    fn pushNarrow(cmd: c.VkCommandBuffer, layout: c.VkPipelineLayout, pc: NarrowPC) void {
        c.vkCmdPushConstants(cmd, layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(NarrowPC), @ptrCast(&pc));
    }

    /// Read back counters from GPU for debugging: [0]=num_pairs, [1]=num_contacts
    pub fn readCounters(self: *Physics, data: *const Data) ![2]u32 {
        var result: [2]u32 = undefined;
        try self.vk.readBuffer(data.atomic_counters, @ptrCast(&result), @sizeOf([2]u32));
        return result;
    }

    /// Read back first contact for debugging
    pub fn readFirstContact(self: *Physics, data: *const Data) !struct { normal: [3]f32, pen: f32, ga: u32, gb: u32 } {
        var n: [3]f32 = undefined;
        var pen: [1]f32 = undefined;
        var ga: [1]u32 = undefined;
        var gb: [1]u32 = undefined;
        try self.vk.readBuffer(data.contact_normal, @ptrCast(&n), 12);
        try self.vk.readBuffer(data.contact_penetration, @ptrCast(&pen), 4);
        try self.vk.readBuffer(data.contact_geom_a, @ptrCast(&ga), 4);
        try self.vk.readBuffer(data.contact_geom_b, @ptrCast(&gb), 4);
        return .{ .normal = n, .pen = pen[0], .ga = ga[0], .gb = gb[0] };
    }

    pub fn deinit(self: *Physics) void {
        self.vk.destroyComputePipeline(self.physics_pipe);
        self.vk.destroyComputePipeline(self.narrow_pipe);
        self.vk.destroyComputePipeline(self.constraint_pipe);
        self.vk.destroyComputePipeline(self.solver_pipe);
    }
};
