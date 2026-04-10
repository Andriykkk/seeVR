const std = @import("std");
const c = @import("c.zig").c;
const Vulkan = @import("vulkan.zig").Vulkan;
const Data = @import("data.zig").Data;
const Profiler = @import("profiler.zig").Profiler;

const PushConstants = extern struct {
    step: u32,
    num_tris: u32,
    pass_shift: u32,
};

pub const BVH = struct {
    vk: *Vulkan,
    pipeline: Vulkan.ComputePipeline,
    prof_ids: [8]u32,

    const step_names = [8][]const u8{
        "  bvh/centroids",
        "  bvh/scene_bounds",
        "  bvh/morton_codes",
        "  bvh/radix_sort",
        "  bvh/build_tree",
        "  bvh/leaf_aabbs",
        "  bvh/clear_flags",
        "  bvh/propagate",
    };

    pub fn init(vk: *Vulkan, data: *const Data, allocator: std.mem.Allocator, prof: ?*Profiler) !BVH {
        const shader = try vk.getShader("src/shaders/bvh.comp", .compute, allocator);

        const buffers = [17]Vulkan.Buffer{
            data.vertices, // 0
            data.indices, // 1
            data.bvh_centroids, // 2
            data.bvh_morton, // 3
            data.bvh_morton_temp, // 4
            data.bvh_sort_indices, // 5
            data.bvh_sort_temp, // 6
            data.bvh_scene_bounds, // 7
            data.bvh_aabb_min, // 8
            data.bvh_aabb_max, // 9
            data.bvh_left, // 10
            data.bvh_right, // 11
            data.bvh_count, // 12
            data.bvh_parent, // 13
            data.bvh_prim_indices, // 14
            data.bvh_flags, // 15
            data.bvh_temp, // 16
        };

        const pipeline = try vk.createComputePipeline(shader, 17, &buffers, @sizeOf(PushConstants));

        var prof_ids: [8]u32 = undefined;
        if (prof) |p| {
            for (0..8) |i| prof_ids[i] = p.addSection(step_names[i]);
        }

        return .{ .vk = vk, .pipeline = pipeline, .prof_ids = prof_ids };
    }

    pub fn build(self: *BVH, num_tris: u32, prof: ?*Profiler) !void {
        if (num_tris == 0) return;

        const vk = self.vk;
        const groups = (num_tris + 255) / 256;
        const queue = if (vk.compute_queue != null) vk.compute_queue else vk.graphics_queue;

        if (prof != null) {
            // Profiling mode: one submit per step
            self.submitStep(queue, .{ .step = 0, .num_tris = num_tris, .pass_shift = 0 }, groups, prof.?, 0);
            // Bounds: centroids → temp
            self.submitStep(queue, .{ .step = 1, .num_tris = num_tris, .pass_shift = 0 }, groups, prof.?, 1);
            // Bounds: temp → temp (loop until ≤256)
            var bc = groups;
            while (bc > 256) {
                const bg = (bc + 255) / 256;
                self.submitStep(queue, .{ .step = 8, .num_tris = bc, .pass_shift = 0 }, bg, prof.?, 1);
                bc = bg;
            }
            // Bounds: temp → final
            self.submitStep(queue, .{ .step = 9, .num_tris = bc, .pass_shift = 0 }, 1, prof.?, 1);
            self.submitStep(queue, .{ .step = 2, .num_tris = num_tris, .pass_shift = 0 }, groups, prof.?, 2);
            for ([_]u32{ 0, 8, 16, 24 }) |shift| {
                self.submitStep(queue, .{ .step = 3, .num_tris = num_tris, .pass_shift = shift }, 1, prof.?, 3);
            }
            self.submitStep(queue, .{ .step = 4, .num_tris = num_tris, .pass_shift = 0 }, groups, prof.?, 4);
            self.submitStep(queue, .{ .step = 5, .num_tris = num_tris, .pass_shift = 0 }, groups, prof.?, 5);
            self.submitStep(queue, .{ .step = 6, .num_tris = num_tris, .pass_shift = 0 }, groups, prof.?, 6);
            self.submitStep(queue, .{ .step = 7, .num_tris = num_tris, .pass_shift = 0 }, groups, prof.?, 7);
        } else {
            // Fast mode: single command buffer
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

            push(cmd, self.pipeline.layout, .{ .step = 0, .num_tris = num_tris, .pass_shift = 0 }, groups);
            bar(cmd, &barrier);
            // Bounds: centroids → temp
            push(cmd, self.pipeline.layout, .{ .step = 1, .num_tris = num_tris, .pass_shift = 0 }, groups);
            bar(cmd, &barrier);
            // Bounds: temp → temp (loop until ≤256)
            {
                var bc = groups;
                while (bc > 256) {
                    const bg = (bc + 255) / 256;
                    push(cmd, self.pipeline.layout, .{ .step = 8, .num_tris = bc, .pass_shift = 0 }, bg);
                    bar(cmd, &barrier);
                    bc = bg;
                }
                // Bounds: temp → final
                push(cmd, self.pipeline.layout, .{ .step = 9, .num_tris = bc, .pass_shift = 0 }, 1);
                bar(cmd, &barrier);
            }
            push(cmd, self.pipeline.layout, .{ .step = 2, .num_tris = num_tris, .pass_shift = 0 }, groups);
            bar(cmd, &barrier);
            for ([_]u32{ 0, 8, 16, 24 }) |shift| {
                push(cmd, self.pipeline.layout, .{ .step = 3, .num_tris = num_tris, .pass_shift = shift }, 1);
                bar(cmd, &barrier);
            }
            push(cmd, self.pipeline.layout, .{ .step = 4, .num_tris = num_tris, .pass_shift = 0 }, groups);
            bar(cmd, &barrier);
            push(cmd, self.pipeline.layout, .{ .step = 5, .num_tris = num_tris, .pass_shift = 0 }, groups);
            bar(cmd, &barrier);
            push(cmd, self.pipeline.layout, .{ .step = 6, .num_tris = num_tris, .pass_shift = 0 }, groups);
            bar(cmd, &barrier);
            push(cmd, self.pipeline.layout, .{ .step = 7, .num_tris = num_tris, .pass_shift = 0 }, groups);

            _ = c.vkEndCommandBuffer(cmd);
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
    }

    fn submitStep(self: *BVH, queue: c.VkQueue, pc: PushConstants, group_count: u32, prof: *Profiler, step_idx: usize) void {
        const vk = self.vk;
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
        c.vkCmdPushConstants(cmd, self.pipeline.layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PushConstants), @ptrCast(&pc));
        c.vkCmdDispatch(cmd, group_count, 1, 1);
        _ = c.vkEndCommandBuffer(cmd);
        prof.submitAndTime(queue, cmd, self.prof_ids[step_idx]);
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
