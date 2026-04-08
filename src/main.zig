const std = @import("std");
const c = @import("c.zig").c;
const Vulkan = @import("vulkan.zig").Vulkan;
const Data = @import("data.zig").Data;
const Scene = @import("scene.zig").Scene;
const Camera = @import("camera.zig").Camera;
const Physics = @import("physics.zig").Physics;
const build_options = @import("build_options");
const Gui = if (build_options.enable_imgui) @import("gui.zig").Gui else void;
const BVH = if (build_options.raytrace) @import("bvh.zig").BVH else void;
const RT = if (build_options.raytrace) @import("raytracer.zig").Raytracer else void;
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

    // Materials                                              albedo               rough  metal  emission
    const mat_ground = d.addMaterial(.{ 0.35, 0.35, 0.30 }, 0.95, 0.0, .{ 0, 0, 0 }); // matte concrete
    const mat_red = d.addMaterial(.{ 0.85, 0.15, 0.15 }, 0.4, 0.0, .{ 0, 0, 0 }); // glossy plastic
    const mat_mirror = d.addMaterial(.{ 0.95, 0.95, 0.95 }, 0.02, 1.0, .{ 0, 0, 0 }); // near-perfect mirror
    const mat_gold = d.addMaterial(.{ 1.0, 0.76, 0.33 }, 0.15, 1.0, .{ 0, 0, 0 }); // polished gold
    const mat_rubber = d.addMaterial(.{ 0.35, 0.8, 0.35 }, 1.0, 0.0, .{ 0, 0, 0 }); // matte rubber
    const mat_copper = d.addMaterial(.{ 0.95, 0.64, 0.54 }, 0.25, 1.0, .{ 0, 0, 0 }); // brushed copper
    const mat_glass = d.addMaterial(.{ 0.8, 0.85, 1.0 }, 0.05, 0.0, .{ 0, 0, 0 }); // smooth dielectric
    const mat_emissive = d.addMaterial(.{ 1.0, 0.9, 0.7 }, 1.0, 0.0, .{ 8, 7, 5 }); // warm light
    const mat_chrome = d.addMaterial(.{ 0.55, 0.56, 0.55 }, 0.1, 1.0, .{ 0, 0, 0 }); // polished chrome

    //                                                                       mass  fric  rest  mat
    // Ground (static)
    _ = try d.addBox(.{ 0, -0.25, 0 }, .{ 10, 0.25, 10 }, .{ 0.35, 0.35, 0.30 }, 0, 0.5, 0.0, mat_ground);
    // Stacked boxes — different materials
    _ = try d.addBox(.{ 0, 1, 0 }, .{ 0.5, 0.5, 0.5 }, .{ 0.85, 0.15, 0.15 }, 1, 0.5, 0.1, mat_red);
    _ = try d.addBox(.{ 0.02, 2.5, 0 }, .{ 0.5, 0.5, 0.5 }, .{ 0.95, 0.95, 0.95 }, 1, 0.5, 0.1, mat_mirror);
    _ = try d.addBox(.{ -0.02, 4.0, 0 }, .{ 0.5, 0.5, 0.5 }, .{ 1.0, 0.76, 0.33 }, 1, 0.5, 0.1, mat_gold);
    _ = try d.addBox(.{ 0.01, 5.5, 0 }, .{ 0.5, 0.5, 0.5 }, .{ 0.35, 0.8, 0.35 }, 1, 0.5, 0.1, mat_rubber);
    // Side boxes
    _ = try d.addBox(.{ 3, 1, 0 }, .{ 0.75, 0.75, 0.75 }, .{ 0.95, 0.64, 0.54 }, 1, 0.3, 0.3, mat_copper);
    _ = try d.addBox(.{ -3, 2, 1 }, .{ 0.6, 0.6, 0.6 }, .{ 0.55, 0.56, 0.55 }, 1, 0.8, 0.0, mat_chrome);
    // Spheres
    _ = try d.addSphere(.{ 1.5, 3, 0 }, 0.4, .{ 0.8, 0.85, 1.0 }, 12, 1, 0.2, 0.97, mat_glass);
    _ = try d.addSphere(.{ -1.0, 3, 0 }, 0.5, .{ 1.0, 0.9, 0.7 }, 12, 1, 0.6, 0.4, mat_emissive);

    try d.upload();

    var physics = try Physics.init(&vk_ctx, &d, allocator);
    defer physics.deinit();

    var bvh = if (comptime BVH != void) try BVH.init(&vk_ctx, &d, allocator) else {};
    defer if (comptime BVH != void) bvh.deinit();

    var rt = if (comptime RT != void) try RT.init(&vk_ctx, &d, scene.rt_view, WIDTH, HEIGHT, allocator) else {};
    defer if (comptime RT != void) rt.deinit();

    var gui = if (comptime Gui != void) try Gui.init(&vk_ctx, &scene, window) else {};
    defer if (comptime Gui != void) gui.deinit();

    std.debug.print("Mode: {s} | {} vertices, {} triangles, {} bodies, {} geoms\n", .{
        if (comptime raytrace_mode) "raytrace" else "raster",
        d.num_vertices,
        d.num_triangles,
        d.num_bodies,
        d.num_geoms,
    });

    var camera = Camera.init(0, 5, 15, -90, -15);
    const aspect: f32 = @as(f32, @floatFromInt(WIDTH)) / @as(f32, @floatFromInt(HEIGHT));
    var last_time: f64 = c.glfwGetTime();
    var fps_smooth: f32 = 0;
    var frame: u64 = 0;

    while (!scene.shouldClose()) {
        const now = c.glfwGetTime();
        const dt: f32 = @floatCast(now - last_time);
        last_time = now;
        if (dt > 0) fps_smooth = 0.95 * fps_smooth + 0.05 * (1.0 / dt);
        if (comptime Gui != void) {
            gui.fps = fps_smooth;
        }

        scene.pollEvents();
        camera.update(window, dt);

        // Physics
        try physics.step(d.num_bodies, 10, 1.0 / 60.0, .{ 0, -9.81, 0 });

        if (comptime raytrace_mode) {
            try bvh.build(d.num_triangles);
            try rt.render(camera.pos, camera.direction(), camera.right(), camera.up(), d.num_triangles);
            try scene.beginFrame();
            scene.blitRtImage();
            try scene.endFrame();

            frame += 1;
            if (frame % 60 == 0) {
                const rays_per_frame = @as(u64, WIDTH) * @as(u64, HEIGHT) * 3 * 4;
                const mrays = @as(f32, @floatFromInt(rays_per_frame)) * fps_smooth / 1_000_000.0;
                std.debug.print("FPS: {d:.0}  verts: {}  tris: {}  bodies: {}  rays: {d:.1} Mray/s\n", .{
                    fps_smooth, d.num_vertices, d.num_triangles, d.num_bodies, mrays,
                });
            }
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
