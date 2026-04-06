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

        // 23 bindings matching physics.comp
        const buffers = [23]Vulkan.Buffer{
            data.body_pos,            // 0
            data.body_quat,           // 1
            data.body_vel,            // 2
            data.body_omega,          // 3
            data.body_inv_mass,       // 4
            data.body_inv_inertia,    // 5
            data.body_vert_start,     // 6
            data.body_vert_count,     // 7
            data.geom_body_idx,       // 8
            data.geom_data,           // 9
            data.hull_verts,          // 10
            data.vertices,            // 11
            data.original_vertices,   // 12
            data.contact_pos,         // 13
            data.contact_normal,      // 14
            data.contact_penetration, // 15
            data.contact_body_a,      // 16
            data.contact_body_b,      // 17
            data.contact_lambda_n,    // 18
            data.atomic_counters,     // 19
            data.body_aabb_min,       // 20
            data.body_aabb_max,       // 21
            data.collision_pairs,     // 22
        };

        const pipe = try vk.createComputePipeline(shader, 23, &buffers, @sizeOf(PC));
        return .{ .vk = vk, .pipe = pipe };
    }

    pub fn step(self: *Physics, num_bodies: u32, dt: f32, gravity: [3]f32) !void {
        if (num_bodies == 0) return;

        const vk = self.vk;
        const groups = @max((num_bodies + 255) / 256, 1);
        const gx = gravity[0];
        const gy = gravity[1];
        const gz = gravity[2];
        const max_c: u32 = @import("data.zig").MAX_CONTACTS;
        const max_p: u32 = @import("data.zig").MAX_COLLISION_PAIRS;

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

        // 8: Clear counters
        dispatch(cmd, self.pipe.layout, .{ .step = 8, .count = 1, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz }, 1);
        bar(cmd, &barrier);

        // 0: Gravity
        dispatch(cmd, self.pipe.layout, .{ .step = 0, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz }, groups);
        bar(cmd, &barrier);

        // 1: Compute AABB
        dispatch(cmd, self.pipe.layout, .{ .step = 1, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz }, groups);
        bar(cmd, &barrier);

        // 2: Broad phase
        dispatch(cmd, self.pipe.layout, .{ .step = 2, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz }, groups);
        bar(cmd, &barrier);

        // 3: Narrow phase (GJK + EPA)
        dispatch(cmd, self.pipe.layout, .{ .step = 3, .count = max_p, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz }, (max_p + 255) / 256);
        bar(cmd, &barrier);

        // 4: Clear impulses
        dispatch(cmd, self.pipe.layout, .{ .step = 4, .count = max_c, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz }, (max_c + 255) / 256);
        bar(cmd, &barrier);

        // 5: PGS solve
        dispatch(cmd, self.pipe.layout, .{ .step = 5, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz }, 1);
        bar(cmd, &barrier);

        // 6: Integrate
        dispatch(cmd, self.pipe.layout, .{ .step = 6, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz }, groups);
        bar(cmd, &barrier);

        // 7: Update render verts
        dispatch(cmd, self.pipe.layout, .{ .step = 7, .count = num_bodies, .dt = dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz }, groups);

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

    fn dispatch(cmd: c.VkCommandBuffer, layout: c.VkPipelineLayout, pc: PC, group_count: u32) void {
        c.vkCmdPushConstants(cmd, layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PC), @ptrCast(&pc));
        c.vkCmdDispatch(cmd, group_count, 1, 1);
    }

    fn bar(cmd: c.VkCommandBuffer, b: *const c.VkMemoryBarrier) void {
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, b, 0, null, 0, null);
    }

    pub fn deinit(self: *Physics) void {
        self.vk.destroyComputePipeline(self.pipe);
    }
};
