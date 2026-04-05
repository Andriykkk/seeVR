const std = @import("std");
const c = @import("c.zig").c;
const Vulkan = @import("vulkan.zig").Vulkan;
const Data = @import("data.zig").Data;

const PC = extern struct {
    step: u32,
    count: u32,
    dt: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
};

pub const Physics = struct {
    vk: *Vulkan,
    pipe: Vulkan.ComputePipeline,

    pub fn init(vk: *Vulkan, data: *const Data, allocator: std.mem.Allocator) !Physics {
        const shader = try vk.getShader("src/shaders/physics.comp", .compute, allocator);
        const buffers = data.physicsBuffers();
        const pipe = try vk.createComputePipeline(shader, 18, &buffers, @sizeOf(PC));
        return .{ .vk = vk, .pipe = pipe };
    }

    pub fn step(self: *Physics, num_bodies: u32, dt: f32, gravity: [3]f32) !void {
        if (num_bodies == 0) return;

        const vk = self.vk;
        const body_groups = @max((num_bodies + 255) / 256, 1);

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

        c.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipe.pipeline);
        c.vkCmdBindDescriptorSets(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipe.layout, 0, 1, &self.pipe.desc_set, 0, null);

        const gx = gravity[0];
        const gy = gravity[1];
        const gz = gravity[2];

        // Step 6: clear counters
        push(cmd, self.pipe.layout, .{ .step = 6, .count = 1, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz });
        c.vkCmdDispatch(cmd, 1, 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // Step 0: gravity + damping
        push(cmd, self.pipe.layout, .{ .step = 0, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz });
        c.vkCmdDispatch(cmd, body_groups, 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // Step 1: detect contacts (SAT, parallel per body row)
        push(cmd, self.pipe.layout, .{ .step = 1, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz });
        c.vkCmdDispatch(cmd, body_groups, 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // Step 2: clear impulses
        push(cmd, self.pipe.layout, .{ .step = 2, .count = 4000, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz });
        c.vkCmdDispatch(cmd, (4000 + 255) / 256, 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // Step 3: PGS solve (serial, 1 thread)
        push(cmd, self.pipe.layout, .{ .step = 3, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz });
        c.vkCmdDispatch(cmd, 1, 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // Step 4: integrate
        push(cmd, self.pipe.layout, .{ .step = 4, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz });
        c.vkCmdDispatch(cmd, body_groups, 1, 1);
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, null, 0, null);

        // Step 5: update render vertices
        push(cmd, self.pipe.layout, .{ .step = 5, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz });
        c.vkCmdDispatch(cmd, body_groups, 1, 1);

        // Final barrier: compute → vertex read
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

    pub fn readCounters(self: *Physics, data: *const Data) ![2]u32 {
        var result: [2]u32 = undefined;
        try self.vk.readBuffer(data.counters, @ptrCast(&result), @sizeOf([2]u32));
        return result;
    }

    fn push(cmd: c.VkCommandBuffer, layout: c.VkPipelineLayout, pc: PC) void {
        c.vkCmdPushConstants(cmd, layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PC), @ptrCast(&pc));
    }

    pub fn deinit(self: *Physics) void {
        self.vk.destroyComputePipeline(self.pipe);
    }
};
