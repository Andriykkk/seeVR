const std = @import("std");
const c = @import("c.zig").c;
const Vulkan = @import("vulkan.zig").Vulkan;
const Data = @import("data.zig").Data;
const Scene = @import("scene.zig").Scene;
const Mode = @import("scene.zig").Mode;

const WIDTH = 800;
const HEIGHT = 600;

fn perspectiveMvp() [16]f32 {
    // Simple perspective projection looking at origin
    const fov = 60.0 * std.math.pi / 180.0;
    const aspect: f32 = @as(f32, @floatFromInt(WIDTH)) / @as(f32, @floatFromInt(HEIGHT));
    const near = 0.1;
    const far = 100.0;
    const f = 1.0 / @tan(fov / 2.0);

    // Projection matrix (column-major)
    var proj = [16]f32{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    proj[0] = f / aspect;
    proj[5] = -f; // Vulkan Y is flipped
    proj[10] = far / (near - far);
    proj[11] = -1.0;
    proj[14] = (near * far) / (near - far);

    // View: translate camera to (0, 5, 15) looking at origin
    // Simple lookAt: just translate then the projection handles it
    var mvp = proj;
    // Apply view translation: shift scene by -eye position
    // For column-major: mvp[12] += dx, mvp[13] += dy, mvp[14] += dz (simplified)
    mvp[12] = proj[0] * 0.0 + proj[4] * -5.0 + proj[8] * -15.0 + proj[12];
    mvp[13] = proj[1] * 0.0 + proj[5] * -5.0 + proj[9] * -15.0 + proj[13];
    mvp[14] = proj[2] * 0.0 + proj[6] * -5.0 + proj[10] * -15.0 + proj[14];
    mvp[15] = proj[3] * 0.0 + proj[7] * -5.0 + proj[11] * -15.0 + proj[15];

    return mvp;
}

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

    const mvp = perspectiveMvp();

    while (!scene.shouldClose()) {
        scene.pollEvents();
        if (mode == .raster) {
            try scene.beginFrame();
            scene.draw(&d, &mvp);
            try scene.endFrame();
        }
    }
}
