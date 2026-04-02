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

const NarrowPC = extern struct {
    step: u32,
    count: u32,
    sign0: f32,
    sign1: f32,
};

pub const Physics = struct {
    vk: *Vulkan,
    physics_pipe: Vulkan.ComputePipeline,
    narrow_pipe: Vulkan.ComputePipeline,

    pub fn init(vk: *Vulkan, data: *const Data, allocator: std.mem.Allocator) !Physics {
        // Physics pipeline (transforms, AABBs, broad phase)
        const phys_shader = try vk.getShader("shaders/physics.comp", .compute, allocator);
        const phys_buffers = [21]Vulkan.Buffer{
            data.body_pos, data.body_quat, data.body_vel, data.body_omega,
            data.body_inv_mass, data.body_inertia, data.body_inv_inertia,
            data.body_vert_start, data.body_vert_count,
            data.geom_type, data.geom_body_idx, data.geom_local_pos, data.geom_local_quat,
            data.geom_world_pos, data.geom_world_quat, data.geom_aabb_min, data.geom_aabb_max,
            data.geom_data, data.vertices, data.collision_pairs, data.atomic_counters,
        };
        const physics_pipe = try vk.createComputePipeline(phys_shader, 21, &phys_buffers, @sizeOf(PhysicsPC));

        // Narrow phase pipeline (MPR)
        const narrow_shader = try vk.getShader("shaders/narrow_phase.comp", .compute, allocator);
        const narrow_buffers = [13]Vulkan.Buffer{
            data.geom_type, data.geom_body_idx, data.geom_world_pos, data.geom_world_quat,
            data.geom_data, data.collision_verts,
            data.collision_pairs, data.atomic_counters,
            data.contact_pos, data.contact_normal, data.contact_penetration,
            data.contact_geom_a, data.contact_geom_b,
        };
        const narrow_pipe = try vk.createComputePipeline(narrow_shader, 13, &narrow_buffers, @sizeOf(NarrowPC));

        return .{ .vk = vk, .physics_pipe = physics_pipe, .narrow_pipe = narrow_pipe };
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

        // Step 3: apply gravity (before collision detection so solver sees gravity velocity)
        pushPhysics(cmd, self.physics_pipe.layout, .{ .step = 3, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz });
        c.vkCmdDispatch(cmd, body_groups, 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

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

        // Step 1: perturbation (4 directions, dispatched with max_pairs as upper bound for contacts)
        const perturb_dirs = [4][2]f32{ .{ 1, 1 }, .{ -1, 1 }, .{ 1, -1 }, .{ -1, -1 } };
        for (perturb_dirs) |dir| {
            pushNarrow(cmd, self.narrow_pipe.layout, .{ .step = 1, .count = max_pairs, .sign0 = dir[0], .sign1 = dir[1] });
            c.vkCmdDispatch(cmd, narrow_groups, 1, 1);
            c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);
        }

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

    fn pushNarrow(cmd: c.VkCommandBuffer, layout: c.VkPipelineLayout, pc: NarrowPC) void {
        c.vkCmdPushConstants(cmd, layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(NarrowPC), @ptrCast(&pc));
    }

    pub fn deinit(self: *Physics) void {
        self.vk.destroyComputePipeline(self.physics_pipe);
        self.vk.destroyComputePipeline(self.narrow_pipe);
    }
};
