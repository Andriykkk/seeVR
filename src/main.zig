const std = @import("std");
const c = @import("c.zig").c;
const Vulkan = @import("vulkan.zig").Vulkan;
const Data = @import("data.zig").Data;
const Scene = @import("scene.zig").Scene;
const Camera = @import("camera.zig").Camera;
const Physics = @import("physics.zig").Physics;
const build_options = @import("build_options");
const Gui = if (build_options.enable_imgui) @import("gui.zig").Gui else void;
const raytrace_mode = build_options.raytrace;

const WIDTH = 800;
const HEIGHT = 600;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    if (c.glfwInit() == 0) return error.GlfwInitFailed;
    defer c.glfwTerminate();

    c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
    const title = if (comptime raytrace_mode) "PGS Boxes [raytrace]" else "PGS Boxes [raster]";
    const window = c.glfwCreateWindow(WIDTH, HEIGHT, title, null, null) orelse
        return error.WindowCreateFailed;
    defer c.glfwDestroyWindow(window);

    var vk_ctx = try Vulkan.init(window);
    defer vk_ctx.deinit();

    var scene = try Scene.init(&vk_ctx, window, WIDTH, HEIGHT, allocator);
    defer scene.deinit();

    var d = try Data.init(&vk_ctx, allocator);
    defer d.deinit();

    // Ground (static)
    _ = try d.addBox(.{ 0, -0.25, 0 }, .{ 10, 0.25, 10 }, .{ 0.3, 0.3, 0.35 }, 0);
    // Stacked boxes
    _ = try d.addBox(.{ 0, 1, 0 }, .{ 0.5, 0.5, 0.5 }, .{ 0.85, 0.25, 0.25 }, 1);
    _ = try d.addBox(.{ 0.02, 2.5, 0 }, .{ 0.5, 0.5, 0.5 }, .{ 0.25, 0.55, 0.85 }, 1);
    _ = try d.addBox(.{ -0.02, 4.0, 0 }, .{ 0.5, 0.5, 0.5 }, .{ 0.85, 0.65, 0.2 }, 1);
    _ = try d.addBox(.{ 0.01, 5.5, 0 }, .{ 0.5, 0.5, 0.5 }, .{ 0.35, 0.8, 0.35 }, 1);
    // Side boxes
    _ = try d.addBox(.{ 3, 1, 0 }, .{ 0.75, 0.75, 0.75 }, .{ 0.3, 0.8, 0.3 }, 1);
    _ = try d.addBox(.{ -3, 2, 1 }, .{ 0.6, 0.6, 0.6 }, .{ 0.8, 0.5, 0.2 }, 1);
    // Spheres
    _ = try d.addSphere(.{ 1.5, 3, 0 }, 0.4, .{ 0.9, 0.3, 0.9 }, 12, 1);
    _ = try d.addSphere(.{ -1.0, 3, 0 }, 0.5, .{ 0.3, 0.9, 0.9 }, 12, 1);

    try d.upload();

    var physics = try Physics.init(&vk_ctx, &d, allocator);
    defer physics.deinit();

    var gui = if (comptime Gui != void) try Gui.init(&vk_ctx, &scene, window) else {};
    defer if (comptime Gui != void) gui.deinit();

    std.debug.print("Mode: {s} | {} vertices, {} triangles, {} bodies, {} geoms\n", .{
        if (comptime raytrace_mode) "raytrace" else "raster",
        d.num_vertices, d.num_triangles, d.num_bodies, d.num_geoms,
    });

    var camera = Camera.init(0, 5, 15, -90, -15);
    const aspect: f32 = @as(f32, @floatFromInt(WIDTH)) / @as(f32, @floatFromInt(HEIGHT));
    var last_time: f64 = c.glfwGetTime();

    while (!scene.shouldClose()) {
        const now = c.glfwGetTime();
        const dt: f32 = @floatCast(now - last_time);
        last_time = now;
        if (comptime Gui != void) {
            if (dt > 0) gui.fps = 0.95 * gui.fps + 0.05 * (1.0 / dt);
        }

        scene.pollEvents();
        camera.update(window, dt);

        // Physics
        try physics.step(d.num_bodies, 10, 1.0 / 60.0, .{ 0, -9.81, 0 });

        if (comptime raytrace_mode) {
            // TODO: BVH build + raytrace compute dispatch writing to scene.rt_image
            try scene.beginFrame();
            scene.blitRtImage(); // blit rt_image → swapchain
            try scene.endFrame();
        } else {
            // Rasterize
            const mvp = camera.mvp(aspect);
            try scene.beginFrame();
            scene.draw(&d, &mvp);
            if (comptime Gui != void) {
                const counters = physics.readCounters(&d) catch .{ 0, 0 };
                gui.pairs = counters[0];
                gui.contacts = counters[1];
                gui.render(&d, scene.cmd_buffers[scene.current_frame]);
            }
            try scene.endFrame();
        }
    }
}
