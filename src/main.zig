const std = @import("std");
const c = @import("c.zig").c;
const Vulkan = @import("vulkan.zig").Vulkan;
const Data = @import("data.zig").Data;
const Scene = @import("scene.zig").Scene;
const Mode = @import("scene.zig").Mode;
const Camera = @import("camera.zig").Camera;

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

    _ = try d.addBox(.{ 0, -0.25, 0 }, .{ 10, 0.25, 10 }, .{ 0.3, 0.3, 0.35 }, 0);
    _ = try d.addBox(.{ 0, 1, 0 }, .{ 0.5, 0.5, 0.5 }, .{ 0.8, 0.3, 0.3 }, 1);
    _ = try d.addBox(.{ 2, 0.75, -1 }, .{ 0.75, 0.75, 0.75 }, .{ 0.3, 0.8, 0.3 }, 1);
    _ = try d.addSphere(.{ -2, 1.5, 0 }, 0.7, .{ 0.2, 0.2, 0.9 }, 16, 1);
    try d.upload();

    std.debug.print("Scene: {} vertices, {} triangles, {} bodies\n", .{ d.num_vertices, d.num_triangles, d.num_bodies });

    var camera = Camera.init(0, 5, 15, -90, -15);
    const aspect: f32 = @as(f32, @floatFromInt(WIDTH)) / @as(f32, @floatFromInt(HEIGHT));
    var last_time: f64 = c.glfwGetTime();

    while (!scene.shouldClose()) {
        const now = c.glfwGetTime();
        const dt: f32 = @floatCast(now - last_time);
        last_time = now;

        scene.pollEvents();

        if (window) |w| camera.update(w, dt);
        const mvp = camera.mvp(aspect);

        if (mode == .raster) {
            try scene.beginFrame();
            scene.draw(&d, &mvp);
            try scene.endFrame();
        }
    }
}
