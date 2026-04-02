const std = @import("std");
const c = @import("c.zig").c;
const Vulkan = @import("vulkan.zig").Vulkan;
const Data = @import("data.zig").Data;
const Scene = @import("scene.zig").Scene;
const Mode = @import("scene.zig").Mode;
const Camera = @import("camera.zig").Camera;
const imgui = @import("imgui.zig");
const BVH = @import("bvh.zig").BVH;
const Physics = @import("physics.zig").Physics;

const WIDTH = 800;
const HEIGHT = 600;

pub fn main() !void {
    const mode: Mode = .raster;
    const allocator = std.heap.page_allocator;

    var window: ?*c.GLFWwindow = null;
    if (mode != .headless) {
        if (c.glfwInit() == 0) return error.GlfwInitFailed;
        c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
        window = c.glfwCreateWindow(WIDTH, HEIGHT, "Robotics Engine", null, null) orelse
            return error.WindowCreateFailed;
    }
    defer {
        if (window) |w| c.glfwDestroyWindow(w);
        if (mode != .headless) c.glfwTerminate();
    }

    var vk_ctx = try Vulkan.init(window);
    defer vk_ctx.deinit();

    var scene = try Scene.init(&vk_ctx, mode, window, WIDTH, HEIGHT, allocator);
    defer scene.deinit();

    var d = try Data.init(&vk_ctx, allocator);
    defer d.deinit();

    // ImGui
    var desc_pool: c.VkDescriptorPool = null;
    if (mode == .raster and window != null) {
        _ = imgui.createContext();
        imgui.initGlfw(window.?);

        desc_pool = try vk_ctx.createDescriptorPool();
        imgui.initVulkan(&imgui.VulkanInfo{
            .instance = vk_ctx.instance,
            .physical_device = vk_ctx.physical_device,
            .device = vk_ctx.device,
            .queue_family = vk_ctx.graphics_family,
            .queue = vk_ctx.graphics_queue,
            .descriptor_pool = desc_pool,
            .render_pass = scene.render_pass,
            .image_count = scene.swapchain_count,
        });
    }
    defer if (mode == .raster) {
        imgui.shutdownVulkan();
        imgui.shutdownGlfw();
        imgui.destroyContext();
        if (desc_pool != null) c.vkDestroyDescriptorPool(vk_ctx.device, desc_pool, null);
    };

    // Demo scene
    _ = try d.addBox(.{ 0, -0.25, 0 }, .{ 10, 0.25, 10 }, .{ 0.3, 0.3, 0.35 }, 0);
    _ = try d.addBox(.{ 0, 1, 0 }, .{ 0.5, 0.5, 0.5 }, .{ 0.8, 0.3, 0.3 }, 1);
    _ = try d.addBox(.{ 2, 0.75, -1 }, .{ 0.75, 0.75, 0.75 }, .{ 0.3, 0.8, 0.3 }, 1);
    _ = try d.addSphere(.{ -2, 1.5, 0 }, 0.7, .{ 0.2, 0.2, 0.9 }, 16, 1);
    _ = try d.addMesh("models/cylinder.obj", .{ 0, 2, 2 }, .{ 0.9, 0.5, 0.2 }, 1, 0.2);

    try d.upload();

    var bvh = try BVH.init(&vk_ctx, &d, allocator);
    defer bvh.deinit();

    var physics = try Physics.init(&vk_ctx, &d, allocator);
    defer physics.deinit();

    std.debug.print("Scene: {} vertices, {} triangles, {} bodies\n", .{ d.num_vertices, d.num_triangles, d.num_bodies });

    var camera = Camera.init(0, 5, 15, -90, -15);
    const aspect: f32 = @as(f32, @floatFromInt(WIDTH)) / @as(f32, @floatFromInt(HEIGHT));
    var last_time: f64 = c.glfwGetTime();
    var fps_smooth: f32 = 0;

    while (!scene.shouldClose()) {
        const now = c.glfwGetTime();
        const dt: f32 = @floatCast(now - last_time);
        last_time = now;
        if (dt > 0) fps_smooth = 0.95 * fps_smooth + 0.05 * (1.0 / dt);

        scene.pollEvents();
        if (window) |w| camera.update(w, dt);
        const mvp = camera.mvp(aspect);

        // Physics: update transforms → compute AABBs → broad phase
        try physics.step(d.num_bodies, d.num_geoms, dt);
        try bvh.build(d.num_triangles);

        if (mode == .raster) {
            try scene.beginFrame();
            scene.draw(&d, &mvp);

            // ImGui overlay
            imgui.newFrame();
            if (imgui.begin("Debug")) {
                var buf: [128]u8 = undefined;
                const fps_str = std.fmt.bufPrintZ(&buf, "FPS: {d:.1}", .{fps_smooth}) catch "FPS: ?";
                imgui.text(fps_str);

                const vert_str = std.fmt.bufPrintZ(&buf, "Vertices: {}", .{d.num_vertices}) catch "";
                imgui.text(vert_str);

                const tri_str = std.fmt.bufPrintZ(&buf, "Triangles: {}", .{d.num_triangles}) catch "";
                imgui.text(tri_str);

                const body_str = std.fmt.bufPrintZ(&buf, "Bodies: {}", .{d.num_bodies}) catch "";
                imgui.text(body_str);

                const mem = d.gpuMemoryBytes();
                const mem_str = std.fmt.bufPrintZ(&buf, "GPU Memory: {d:.2} MB", .{@as(f32, @floatFromInt(mem)) / (1024.0 * 1024.0)}) catch "";
                imgui.text(mem_str);

                imgui.end();
            }
            imgui.render();
            imgui.renderDrawData(scene.cmd_buffers[scene.current_frame]);

            try scene.endFrame();
        }
    }
}
