const std = @import("std");
const c = @import("c.zig").c;
const Vulkan = @import("vulkan.zig").Vulkan;
const Data = @import("data.zig").Data;
const Profiler = @import("profiler.zig").Profiler;

const PC = extern struct {
    step: u32,
    count: u32,
    dt: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
    num_bodies: u32,
};

pub const Physics = struct {
    vk: *Vulkan,
    pipe: Vulkan.ComputePipeline,
    prof_ids: [9]u32,

    const step_names = [9][]const u8{
        "  phys/gravity",
        "  phys/aabb",
        "  phys/broad_phase",
        "  phys/narrow_phase",
        "  phys/clear_impulse",
        "  phys/pgs_solve",
        "  phys/integrate",
        "  phys/render_verts",
        "  phys/clear_counters",
    };

    pub fn init(vk: *Vulkan, data: *const Data, allocator: std.mem.Allocator, prof: ?*Profiler) !Physics {
        const shader = try vk.getShader("src/shaders/physics.comp", .compute, allocator);

        const buffers = [25]Vulkan.Buffer{
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
            data.geom_friction,       // 10
            data.geom_restitution,    // 11
            data.hull_verts,          // 12
            data.vertices,            // 13
            data.original_vertices,   // 14
            data.contact_pos,         // 15
            data.contact_normal,      // 16
            data.contact_penetration, // 17
            data.contact_geom_a,      // 18
            data.contact_geom_b,      // 19
            data.contact_lambda_n,    // 20
            data.atomic_counters,     // 21
            data.body_aabb_min,       // 22
            data.body_aabb_max,       // 23
            data.collision_pairs,     // 24
        };

        const pipe = try vk.createComputePipeline(shader, 25, &buffers, @sizeOf(PC));

        var prof_ids: [9]u32 = undefined;
        if (prof) |p| {
            for (0..9) |i| prof_ids[i] = p.addSection(step_names[i]);
        }

        return .{ .vk = vk, .pipe = pipe, .prof_ids = prof_ids };
    }

    pub fn step(self: *Physics, num_bodies: u32, num_vertices: u32, substeps: u32, dt: f32, gravity: [3]f32, prof: ?*Profiler) !void {
        if (num_bodies == 0 or substeps == 0) return;

        const vk = self.vk;
        const groups = @max((num_bodies + 255) / 256, 1);
        const gx = gravity[0];
        const gy = gravity[1];
        const gz = gravity[2];
        const max_c: u32 = @import("data.zig").MAX_CONTACTS;
        const max_p: u32 = @import("data.zig").MAX_COLLISION_PAIRS;
        const sub_dt = dt / @as(f32, @floatFromInt(substeps));
        const queue = if (vk.compute_queue != null) vk.compute_queue else vk.graphics_queue;
        const vert_groups = @max((num_vertices + 255) / 256, 1);

        if (prof != null) {
            // Profiling mode: one submit per step
            for (0..substeps) |_| {
                self.submitStep(queue, .{ .step = 8, .count = 1, .dt = sub_dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz, .num_bodies = 0 }, 1, prof.?, 8);
                self.submitStep(queue, .{ .step = 0, .count = num_bodies, .dt = sub_dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz, .num_bodies = 0 }, groups, prof.?, 0);
                self.submitStep(queue, .{ .step = 1, .count = num_bodies, .dt = sub_dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz, .num_bodies = 0 }, groups, prof.?, 1);
                self.submitStep(queue, .{ .step = 2, .count = num_bodies, .dt = sub_dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz, .num_bodies = 0 }, groups, prof.?, 2);
                self.submitStep(queue, .{ .step = 3, .count = max_p, .dt = sub_dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz, .num_bodies = 0 }, (max_p + 255) / 256, prof.?, 3);
                self.submitStep(queue, .{ .step = 4, .count = max_c, .dt = sub_dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz, .num_bodies = 0 }, (max_c + 255) / 256, prof.?, 4);
                self.submitStep(queue, .{ .step = 5, .count = num_bodies, .dt = sub_dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz, .num_bodies = 0 }, 1, prof.?, 5);
                self.submitStep(queue, .{ .step = 6, .count = num_bodies, .dt = sub_dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz, .num_bodies = 0 }, groups, prof.?, 6);
            }
            self.submitStep(queue, .{ .step = 7, .count = num_vertices, .dt = sub_dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz, .num_bodies = num_bodies }, vert_groups, prof.?, 7);
        } else {
            // Fast mode: single command buffer
            const cmd = self.beginCmd();
            c.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipe.pipeline);
            c.vkCmdBindDescriptorSets(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipe.layout, 0, 1, &self.pipe.desc_set, 0, null);

            const barrier = c.VkMemoryBarrier{
                .sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = null,
                .srcAccessMask = c.VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = c.VK_ACCESS_SHADER_READ_BIT,
            };

            for (0..substeps) |_| {
                dispatch(cmd, self.pipe.layout, .{ .step = 8, .count = 1, .dt = sub_dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz, .num_bodies = 0 }, 1);
                bar(cmd, &barrier);
                dispatch(cmd, self.pipe.layout, .{ .step = 0, .count = num_bodies, .dt = sub_dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz, .num_bodies = 0 }, groups);
                bar(cmd, &barrier);
                dispatch(cmd, self.pipe.layout, .{ .step = 1, .count = num_bodies, .dt = sub_dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz, .num_bodies = 0 }, groups);
                bar(cmd, &barrier);
                dispatch(cmd, self.pipe.layout, .{ .step = 2, .count = num_bodies, .dt = sub_dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz, .num_bodies = 0 }, groups);
                bar(cmd, &barrier);
                dispatch(cmd, self.pipe.layout, .{ .step = 3, .count = max_p, .dt = sub_dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz, .num_bodies = 0 }, (max_p + 255) / 256);
                bar(cmd, &barrier);
                dispatch(cmd, self.pipe.layout, .{ .step = 4, .count = max_c, .dt = sub_dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz, .num_bodies = 0 }, (max_c + 255) / 256);
                bar(cmd, &barrier);
                dispatch(cmd, self.pipe.layout, .{ .step = 5, .count = num_bodies, .dt = sub_dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz, .num_bodies = 0 }, 1);
                bar(cmd, &barrier);
                dispatch(cmd, self.pipe.layout, .{ .step = 6, .count = num_bodies, .dt = sub_dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz, .num_bodies = 0 }, groups);
                bar(cmd, &barrier);
            }
            dispatch(cmd, self.pipe.layout, .{ .step = 7, .count = num_vertices, .dt = sub_dt, .gravity_x = gx, .gravity_y = gy, .gravity_z = gz, .num_bodies = num_bodies }, vert_groups);
            c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1, &barrier, 0, null, 0, null);

            self.endAndSubmit(cmd, queue);
        }
    }

    fn submitStep(self: *Physics, queue: c.VkQueue, pc: PC, group_count: u32, prof: *Profiler, step_idx: usize) void {
        const cmd = self.beginCmd();
        c.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipe.pipeline);
        c.vkCmdBindDescriptorSets(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipe.layout, 0, 1, &self.pipe.desc_set, 0, null);
        c.vkCmdPushConstants(cmd, self.pipe.layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PC), @ptrCast(&pc));
        c.vkCmdDispatch(cmd, group_count, 1, 1);
        _ = c.vkEndCommandBuffer(cmd);
        prof.submitAndTime(queue, cmd, self.prof_ids[step_idx]);
        c.vkFreeCommandBuffers(self.vk.device, self.vk.cmd_pool, 1, &cmd);
    }

    fn beginCmd(self: *Physics) c.VkCommandBuffer {
        var cmd: c.VkCommandBuffer = null;
        _ = c.vkAllocateCommandBuffers(self.vk.device, &c.VkCommandBufferAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .pNext = null,
            .commandPool = self.vk.cmd_pool,
            .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
        }, &cmd);
        _ = c.vkBeginCommandBuffer(cmd, &c.VkCommandBufferBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .pNext = null,
            .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, .pInheritanceInfo = null,
        });
        return cmd;
    }

    fn endAndSubmit(self: *Physics, cmd: c.VkCommandBuffer, queue: c.VkQueue) void {
        _ = c.vkEndCommandBuffer(cmd);
        _ = c.vkQueueSubmit(queue, 1, &c.VkSubmitInfo{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = null,
            .waitSemaphoreCount = 0, .pWaitSemaphores = null, .pWaitDstStageMask = null,
            .commandBufferCount = 1, .pCommandBuffers = &cmd,
            .signalSemaphoreCount = 0, .pSignalSemaphores = null,
        }, null);
        _ = c.vkQueueWaitIdle(queue);
        c.vkFreeCommandBuffers(self.vk.device, self.vk.cmd_pool, 1, &cmd);
    }

    fn dispatch(cmd: c.VkCommandBuffer, layout: c.VkPipelineLayout, pc: PC, group_count: u32) void {
        c.vkCmdPushConstants(cmd, layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PC), @ptrCast(&pc));
        c.vkCmdDispatch(cmd, group_count, 1, 1);
    }

    fn bar(cmd: c.VkCommandBuffer, b: *const c.VkMemoryBarrier) void {
        c.vkCmdPipelineBarrier(cmd, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, b, 0, null, 0, null);
    }

    pub fn readCounters(self: *Physics, data: *const Data) ![2]u32 {
        var result: [2]u32 = undefined;
        try self.vk.readBuffer(data.atomic_counters, @ptrCast(&result), @sizeOf([2]u32));
        return result;
    }

    pub fn deinit(self: *Physics) void {
        self.vk.destroyComputePipeline(self.pipe);
    }
};
