const std = @import("std");
const c = @import("c.zig").c;
const Vulkan = @import("vulkan.zig").Vulkan;
const Data = @import("data.zig").Data;

pub fn main() !void {
    if (c.glfwInit() == 0) return error.GlfwInitFailed;
    defer c.glfwTerminate();

    c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
    const window = c.glfwCreateWindow(800, 600, "Robotics Engine", null, null) orelse
        return error.WindowCreateFailed;
    defer c.glfwDestroyWindow(window);

    var vk_ctx = try Vulkan.init(window);
    defer vk_ctx.deinit();

    const allocator = std.heap.page_allocator;
    var d = try Data.init(&vk_ctx, allocator);
    defer d.deinit();

    // Demo scene
    _ = try d.addBox(.{ 0, -0.25, 0 }, .{ 10, 0.25, 10 }, .{ 0.3, 0.3, 0.35 }, 0); // ground
    _ = try d.addBox(.{ 0, 1, 0 }, .{ 0.5, 0.5, 0.5 }, .{ 0.8, 0.3, 0.3 }, 1); // red box
    _ = try d.addBox(.{ 2, 0.75, -1 }, .{ 0.75, 0.75, 0.75 }, .{ 0.3, 0.8, 0.3 }, 1); // green box
    try d.upload();

    std.debug.print("Scene: {} vertices, {} triangles, {} bodies\n", .{ d.num_vertices, d.num_triangles, d.num_bodies });

    while (c.glfwWindowShouldClose(window) == 0) {
        c.glfwPollEvents();
    }
}
