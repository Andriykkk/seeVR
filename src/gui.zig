const std = @import("std");
const c = @import("c.zig").c;
const Vulkan = @import("vulkan.zig").Vulkan;
const Scene = @import("scene.zig").Scene;
const Data = @import("data.zig").Data;
const build_options = @import("build_options");

const imgui = if (build_options.enable_imgui) struct {
    extern fn imgui_create_context() ?*anyopaque;
    extern fn imgui_destroy_context() void;
    extern fn imgui_init_glfw(window: *anyopaque) void;
    extern fn imgui_shutdown_glfw() void;
    extern fn imgui_init_vulkan(info: *const VulkanInfo) void;
    extern fn imgui_shutdown_vulkan() void;
    extern fn imgui_new_frame() void;
    extern fn imgui_render() void;
    extern fn imgui_render_draw_data(cmd: *anyopaque) void;
    extern fn imgui_begin(name: [*:0]const u8) c_int;
    extern fn imgui_end() void;
    extern fn imgui_text(text: [*:0]const u8) void;

    const VulkanInfo = extern struct {
        instance: c.VkInstance,
        physical_device: c.VkPhysicalDevice,
        device: c.VkDevice,
        queue_family: u32,
        queue: c.VkQueue,
        descriptor_pool: c.VkDescriptorPool,
        render_pass: c.VkRenderPass,
        image_count: u32,
    };
} else void;

pub const Gui = struct {
    enabled: bool,
    desc_pool: c.VkDescriptorPool,
    vk: *Vulkan,

    fps: f32,
    pairs: u32,
    contacts: u32,

    pub fn init(vk: *Vulkan, scene: *const Scene, window: *c.GLFWwindow) !Gui {
        if (!build_options.enable_imgui) {
            return .{ .enabled = false, .desc_pool = null, .vk = vk, .fps = 0, .pairs = 0, .contacts = 0 };
        }

        _ = imgui.imgui_create_context();
        imgui.imgui_init_glfw(@ptrCast(window));

        const desc_pool = try vk.createDescriptorPool();
        imgui.imgui_init_vulkan(&imgui.VulkanInfo{
            .instance = vk.instance,
            .physical_device = vk.physical_device,
            .device = vk.device,
            .queue_family = vk.graphics_family,
            .queue = vk.graphics_queue,
            .descriptor_pool = desc_pool,
            .render_pass = scene.render_pass,
            .image_count = scene.swapchain_count,
        });

        return .{ .enabled = true, .desc_pool = desc_pool, .vk = vk, .fps = 0, .pairs = 0, .contacts = 0 };
    }

    pub fn render(self: *Gui, data: *const Data, cmd: c.VkCommandBuffer) void {
        if (!build_options.enable_imgui) return;

        imgui.imgui_new_frame();

        if (imgui.imgui_begin("Stats") != 0) {
            var buf: [128]u8 = undefined;

            const fps_str = std.fmt.bufPrintZ(&buf, "FPS: {d:.1}", .{self.fps}) catch "?";
            imgui.imgui_text(fps_str);

            const vert_str = std.fmt.bufPrintZ(&buf, "Vertices: {}", .{data.num_vertices}) catch "";
            imgui.imgui_text(vert_str);

            const tri_str = std.fmt.bufPrintZ(&buf, "Triangles: {}", .{data.num_triangles}) catch "";
            imgui.imgui_text(tri_str);

            const body_str = std.fmt.bufPrintZ(&buf, "Bodies: {}", .{data.num_bodies}) catch "";
            imgui.imgui_text(body_str);

            const geom_str = std.fmt.bufPrintZ(&buf, "Geoms: {}", .{data.num_geoms}) catch "";
            imgui.imgui_text(geom_str);

            const mem = data.gpuMemoryBytes();
            const mem_str = std.fmt.bufPrintZ(&buf, "GPU Memory: {d:.2} MB", .{@as(f32, @floatFromInt(mem)) / (1024.0 * 1024.0)}) catch "";
            imgui.imgui_text(mem_str);

            const pair_str = std.fmt.bufPrintZ(&buf, "Pairs: {}", .{self.pairs}) catch "";
            imgui.imgui_text(pair_str);

            const ct_str = std.fmt.bufPrintZ(&buf, "Contacts: {}", .{self.contacts}) catch "";
            imgui.imgui_text(ct_str);

            imgui.imgui_end();
        }

        imgui.imgui_render();
        imgui.imgui_render_draw_data(@ptrCast(cmd));
    }

    pub fn deinit(self: *Gui) void {
        if (!build_options.enable_imgui) return;

        imgui.imgui_shutdown_vulkan();
        imgui.imgui_shutdown_glfw();
        imgui.imgui_destroy_context();
        if (self.desc_pool != null) c.vkDestroyDescriptorPool(self.vk.device, self.desc_pool, null);
    }
};
