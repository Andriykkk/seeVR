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

        // 18 bindings matching physics.comp layout
        const buffers = [18]Vulkan.Buffer{
            data.body_pos,           // 0
            data.body_quat,          // 1
            data.body_vel,           // 2
            data.body_omega,         // 3
            data.body_inv_mass,      // 4
            data.body_inv_inertia,   // 5
            data.body_half,          // 6
            data.body_vert_start,    // 7
            data.body_vert_count,    // 8
            data.vertices,           // 9
            data.original_vertices,  // 10
            data.contact_pos,        // 11
            data.contact_normal,     // 12
            data.contact_penetration, // 13
            data.contact_body_a,     // 14
            data.contact_body_b,     // 15
            data.contact_lambda_n,   // 16
            data.atomic_counters,    // 17
        };

        const pipe = try vk.createComputePipeline(shader, 18, &buffers, @sizeOf(PC));
        return .{ .vk = vk, .pipe = pipe };
    }

    pub fn step(self: *Physics, num_bodies: u32, dt: f32, gravity: [3]f32) !void {
        if (num_bodies == 0) return;

        const vk = self.vk;
        const groups = @max((num_bodies + 255) / 256, 1);

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

        // Clear counters
        self.dispatch(cmd, .{ .step = 6, .count = 1, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz }, 1);
        self.bar(cmd, &barrier);

        // Gravity + damping
        self.dispatch(cmd, .{ .step = 0, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz }, groups);
        self.bar(cmd, &barrier);

        // Detect contacts (SAT)
        self.dispatch(cmd, .{ .step = 1, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz }, groups);
        self.bar(cmd, &barrier);

        // Clear impulses
        self.dispatch(cmd, .{ .step = 2, .count = @import("data.zig").MAX_CONTACTS, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz }, (@import("data.zig").MAX_CONTACTS + 255) / 256);
        self.bar(cmd, &barrier);

        // PGS solve (serial)
        self.dispatch(cmd, .{ .step = 3, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz }, 1);
        self.bar(cmd, &barrier);

        // Integrate
        self.dispatch(cmd, .{ .step = 4, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz }, groups);
        self.bar(cmd, &barrier);

        // Update render vertices
        self.dispatch(cmd, .{ .step = 5, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz }, groups);

        // Compute → vertex read
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1, &barrier, 0, null, 0, null);

        _ = c.vkEndCommandBuffer(cmd);

        const queue = if (vk.compute_queue != null) vk.compute_queue else vk.graphics_queue;
        _ = c.vkQueueSubmit(queue, 1, &c.VkSubmitInfo{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 0, .pWaitSemaphores = null, .pWaitDstStageMask = null,
            .commandBufferCount = 1, .pCommandBuffers = &cmd,
            .signalSemaphoreCount = 0, .pSignalSemaphores = null,
        }, null);
        _ = c.vkQueueWaitIdle(queue);
        c.vkFreeCommandBuffers(vk.device, vk.cmd_pool, 1, &cmd);
    }

    pub fn readCounters(self: *Physics, data: *const Data) ![2]u32 {
        var result: [2]u32 = undefined;
        try self.vk.readBuffer(data.atomic_counters, @ptrCast(&result), @sizeOf([2]u32));
        return result;
    }

    fn dispatch(self: *Physics, cmd: c.VkCommandBuffer, pc: PC, group_count: u32) void {
        c.vkCmdPushConstants(cmd, self.pipe.layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PC), @ptrCast(&pc));
        c.vkCmdDispatch(cmd, group_count, 1, 1);
    }

    fn bar(_: *Physics, cmd: c.VkCommandBuffer, barrier: *const c.VkMemoryBarrier) void {
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, barrier, 0, null, 0, null);
    }

    pub fn deinit(self: *Physics) void {
        self.vk.destroyComputePipeline(self.pipe);
    }
};
