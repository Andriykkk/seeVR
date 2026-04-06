const std = @import("std");
const c = @import("c.zig").c;
const Vulkan = @import("vulkan.zig").Vulkan;
const Data = @import("data.zig").Data;

const PushConstants = extern struct {
    step: u32,
    num_tris: u32,
    pass_shift: u32,
};

pub const BVH = struct {
    vk: *Vulkan,
    pipeline: Vulkan.ComputePipeline,

    pub fn init(vk: *Vulkan, data: *const Data, allocator: std.mem.Allocator) !BVH {
        const shader = try vk.getShader("src/shaders/bvh.comp", .compute, allocator);

        // 16 bindings matching bvh.comp
        const buffers = [16]Vulkan.Buffer{
            data.vertices,         // 0
            data.indices,          // 1
            data.bvh_centroids,    // 2
            data.bvh_morton,       // 3
            data.bvh_morton_temp,  // 4
            data.bvh_sort_indices, // 5
            data.bvh_sort_temp,    // 6
            data.bvh_scene_bounds, // 7
            data.bvh_aabb_min,     // 8
            data.bvh_aabb_max,     // 9
            data.bvh_left,         // 10
            data.bvh_right,        // 11
            data.bvh_count,        // 12
            data.bvh_parent,       // 13
            data.bvh_prim_indices, // 14
            data.bvh_flags,        // 15
        };

        const pipeline = try vk.createComputePipeline(shader, 16, &buffers, @sizeOf(PushConstants));
        return .{ .vk = vk, .pipeline = pipeline };
    }

    pub fn build(self: *BVH, num_tris: u32) !void {
        if (num_tris == 0) return;

        const vk = self.vk;
        const groups = (num_tris + 255) / 256;

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

        // 0: centroids + init sort indices
        push(cmd, self.pipeline.layout, .{ .step = 0, .num_tris = num_tris, .pass_shift = 0 }, groups);
        bar(cmd, &barrier);

        // 1: scene bounds (serial)
        push(cmd, self.pipeline.layout, .{ .step = 1, .num_tris = num_tris, .pass_shift = 0 }, 1);
        bar(cmd, &barrier);

        // 2: morton codes
        push(cmd, self.pipeline.layout, .{ .step = 2, .num_tris = num_tris, .pass_shift = 0 }, groups);
        bar(cmd, &barrier);

        // 3: radix sort (4 passes)
        for ([_]u32{ 0, 8, 16, 24 }) |shift| {
            push(cmd, self.pipeline.layout, .{ .step = 3, .num_tris = num_tris, .pass_shift = shift }, 1);
            bar(cmd, &barrier);
        }

        // 4: build hierarchy (Karras)
        push(cmd, self.pipeline.layout, .{ .step = 4, .num_tris = num_tris, .pass_shift = 0 }, groups);
        bar(cmd, &barrier);

        // 5: leaf AABBs
        push(cmd, self.pipeline.layout, .{ .step = 5, .num_tris = num_tris, .pass_shift = 0 }, groups);
        bar(cmd, &barrier);

        // 6: clear flags
        push(cmd, self.pipeline.layout, .{ .step = 6, .num_tris = num_tris, .pass_shift = 0 }, groups);
        bar(cmd, &barrier);

        // 7: propagate AABBs bottom-up
        push(cmd, self.pipeline.layout, .{ .step = 7, .num_tris = num_tris, .pass_shift = 0 }, groups);

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

    fn push(cmd: c.VkCommandBuffer, layout: c.VkPipelineLayout, pc: PushConstants, group_count: u32) void {
        c.vkCmdPushConstants(cmd, layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConstants), @ptrCast(&pc));
        c.vkCmdDispatch(cmd, group_count, 1, 1);
    }

    fn bar(cmd: c.VkCommandBuffer, b: *const c.VkMemoryBarrier) void {
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, b, 0, null, 0, null);
    }

    pub fn deinit(self: *BVH) void {
        self.vk.destroyComputePipeline(self.pipeline);
    }
};
