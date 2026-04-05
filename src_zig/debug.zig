const std = @import("std");
const c = @import("c.zig").c;
const Vulkan = @import("vulkan.zig").Vulkan;
const Data = @import("data.zig").Data;
const imgui = @import("imgui.zig");

const MAX_DEBUG_LINES: u32 = 50_000;

const DebugPC = extern struct {
    flags: u32,
    max_lines: u32,
};

pub const Debug = struct {
    vk: *Vulkan,
    line_verts: Vulkan.Buffer,
    line_colors: Vulkan.Buffer,
    num_lines: u32,
    pipeline: Vulkan.ComputePipeline,

    show_contacts: bool,
    show_normals: bool,
    show_aabbs: bool,

    pub fn init(vk: *Vulkan, data: *const Data, allocator: std.mem.Allocator) !Debug {
        const VERTEX = Vulkan.USAGE_VERTEX;
        const line_verts = try vk.createBuffer(MAX_DEBUG_LINES * 2 * @sizeOf([3]f32), VERTEX);
        const line_colors = try vk.createBuffer(MAX_DEBUG_LINES * 2 * @sizeOf([3]f32), VERTEX);

        const shader = try vk.getShader("shaders/debug_lines.comp", .compute, allocator);
        const buffers = [8]Vulkan.Buffer{
            data.contact_pos,
            data.contact_normal,
            data.contact_penetration,
            data.atomic_counters,
            data.geom_aabb_min,
            data.geom_aabb_max,
            line_verts,
            line_colors,
        };
        const pipe = try vk.createComputePipeline(shader, 8, &buffers, @sizeOf(DebugPC));

        return .{
            .vk = vk,
            .line_verts = line_verts,
            .line_colors = line_colors,
            .num_lines = 0,
            .pipeline = pipe,
            .show_contacts = true,
            .show_normals = true,
            .show_aabbs = false,
        };
    }

    pub fn buildLines(self: *Debug, data: *const Data) !void {
        const flags: u32 = (@as(u32, @intFromBool(self.show_contacts)) << 0) |
            (@as(u32, @intFromBool(self.show_normals)) << 1) |
            (@as(u32, @intFromBool(self.show_aabbs)) << 2);

        const pc = DebugPC{ .flags = flags, .max_lines = MAX_DEBUG_LINES };

        var cmd: c.VkCommandBuffer = null;
        _ = c.vkAllocateCommandBuffers(self.vk.device, &c.VkCommandBufferAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = null,
            .commandPool = self.vk.cmd_pool,
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
        c.vkCmdPushConstants(cmd, self.pipeline.layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(DebugPC), @ptrCast(&pc));
        c.vkCmdDispatch(cmd, 1, 1, 1);

        _ = c.vkEndCommandBuffer(cmd);

        const queue = if (self.vk.compute_queue != null) self.vk.compute_queue else self.vk.graphics_queue;
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
        c.vkFreeCommandBuffers(self.vk.device, self.vk.cmd_pool, 1, &cmd);

        // Read back line count from counters[2]
        var counters: [4]u32 = undefined;
        try self.vk.readBuffer(data.atomic_counters, @ptrCast(&counters), 4 * @sizeOf(u32));
        self.num_lines = counters[2];
    }

    pub fn drawGui(self: *Debug) void {
        if (imgui.begin("Debug Vis")) {
            _ = imgui.checkbox("Contacts", &self.show_contacts);
            _ = imgui.checkbox("Normals", &self.show_normals);
            _ = imgui.checkbox("AABBs", &self.show_aabbs);

            var buf: [64]u8 = undefined;
            const lines_str = std.fmt.bufPrintZ(&buf, "Lines: {}", .{self.num_lines}) catch "";
            imgui.text(lines_str);

            imgui.end();
        }
    }

    pub fn deinit(self: *Debug) void {
        self.vk.destroyBuffer(self.line_verts);
        self.vk.destroyBuffer(self.line_colors);
        self.vk.destroyComputePipeline(self.pipeline);
    }
};
