const std = @import("std");
const c = @import("c.zig").c;
const Vulkan = @import("vulkan.zig").Vulkan;
const Data = @import("data.zig").Data;

const PushConstants = extern struct {
    step: u32,
    count: u32,
    dt: f32,
};

pub const Physics = struct {
    vk: *Vulkan,
    pipeline: Vulkan.ComputePipeline,

    pub fn init(vk: *Vulkan, data: *const Data, allocator: std.mem.Allocator) !Physics {
        const shader = try vk.getShader("shaders/physics.comp", .compute, allocator);

        // 21 bindings matching physics.comp layout
        const buffers = [21]Vulkan.Buffer{
            data.body_pos,         // 0
            data.body_quat,        // 1
            data.body_vel,         // 2
            data.body_omega,       // 3
            data.body_inv_mass,    // 4
            data.body_inertia,     // 5
            data.body_inv_inertia, // 6
            data.body_vert_start,  // 7
            data.body_vert_count,  // 8
            data.geom_type,        // 9
            data.geom_body_idx,    // 10
            data.geom_local_pos,   // 11
            data.geom_local_quat,  // 12
            data.geom_world_pos,   // 13
            data.geom_world_quat,  // 14
            data.geom_aabb_min,    // 15
            data.geom_aabb_max,    // 16
            data.geom_data,        // 17
            data.vertices,         // 18
            data.collision_pairs,  // 19
            data.atomic_counters,  // 20
        };

        const pipeline = try vk.createComputePipeline(shader, 21, &buffers, @sizeOf(PushConstants));

        return .{ .vk = vk, .pipeline = pipeline };
    }

    /// Full physics step: update transforms → compute AABBs → broad phase
    pub fn step(self: *Physics, num_bodies: u32, num_geoms: u32, dt: f32) !void {
        if (num_geoms == 0) return;

        const vk = self.vk;
        const geom_groups = (num_geoms + 255) / 256;
        _ = num_bodies;

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

        c.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline.pipeline);
        c.vkCmdBindDescriptorSets(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline.layout, 0, 1, &self.pipeline.desc_set, 0, null);

        const barrier = c.VkMemoryBarrier{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = null,
            .srcAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = c.VK_ACCESS_SHADER_READ_BIT,
        };

        // Step 0: update geom transforms
        pushAndDispatch(cmd, self.pipeline.layout, .{ .step = 0, .count = num_geoms, .dt = dt }, geom_groups);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // Step 1: compute AABBs
        pushAndDispatch(cmd, self.pipeline.layout, .{ .step = 1, .count = num_geoms, .dt = dt }, geom_groups);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // Step 2: broad phase
        pushAndDispatch(cmd, self.pipeline.layout, .{ .step = 2, .count = num_geoms, .dt = dt }, geom_groups);

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

    fn pushAndDispatch(cmd: c.VkCommandBuffer, layout: c.VkPipelineLayout, pc: PushConstants, groups: u32) void {
        c.vkCmdPushConstants(cmd, layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConstants), @ptrCast(&pc));
        c.vkCmdDispatch(cmd, groups, 1, 1);
    }

    pub fn deinit(self: *Physics) void {
        self.vk.destroyComputePipeline(self.pipeline);
    }
};
