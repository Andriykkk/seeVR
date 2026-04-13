const std = @import("std");
const c = @import("c.zig").c;
const Vulkan = @import("vulkan.zig").Vulkan;
const Data = @import("data.zig").Data;
const Scene = @import("scene.zig").Scene;
const Camera = @import("camera.zig").Camera;
const Physics = @import("physics.zig").Physics;
const Profiler = @import("profiler.zig").Profiler;
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

    // Materials                                              albedo               rough  metal  emission         ior    density (kg/m³)
    const mat_ground = d.addMaterial(.{ 0.35, 0.35, 0.30 }, 0.95, 0.0, .{ 0, 0, 0 }, 0, 0); // concrete, static
    const mat_red = d.addMaterial(.{ 0.85, 0.15, 0.15 }, 0.4, 0.0, .{ 0, 0, 0 }, 0, 1200); // plastic ~1200
    const mat_mirror = d.addMaterial(.{ 0.95, 0.95, 0.95 }, 0.02, 1.0, .{ 0, 0, 0 }, 0, 7800); // steel ~7800
    const mat_gold = d.addMaterial(.{ 1.0, 0.76, 0.33 }, 0.15, 1.0, .{ 0, 0, 0 }, 0, 19300); // gold ~19300
    const mat_rubber = d.addMaterial(.{ 0.35, 0.8, 0.35 }, 1.0, 0.0, .{ 0, 0, 0 }, 0, 1100); // rubber ~1100
    const mat_copper = d.addMaterial(.{ 0.95, 0.64, 0.54 }, 0.25, 1.0, .{ 0, 0, 0 }, 0, 8900); // copper ~8900
    const mat_glass = d.addMaterial(.{ 0.95, 0.95, 1.0 }, 0.0, 0.0, .{ 0, 0, 0 }, 1.5, 2500); // glass ~2500
    const mat_emissive = d.addMaterial(.{ 1.0, 0.9, 0.7 }, 1.0, 0.0, .{ 8, 7, 5 }, 0, 500); // light fixture
    const mat_chrome = d.addMaterial(.{ 0.55, 0.56, 0.55 }, 0.1, 1.0, .{ 0, 0, 0 }, 0, 7200); // chrome ~7200

    //                                                                       fric  rest  mat
    // Ground (static, density=0)
    _ = try d.addBox(.{ 0, -0.25, 0 }, .{ 10, 0.25, 10 }, .{ 0.35, 0.35, 0.30 }, 0.5, 0.0, mat_ground);
    // Stacked boxes
    _ = try d.addBox(.{ 0, 1, 0 }, .{ 0.5, 0.5, 0.5 }, .{ 0.85, 0.15, 0.15 }, 0.5, 0.1, mat_red);
    _ = try d.addBox(.{ 0.02, 2.5, 0 }, .{ 0.5, 0.5, 0.5 }, .{ 0.95, 0.95, 0.95 }, 0.5, 0.1, mat_mirror);
    _ = try d.addBox(.{ -0.02, 4.0, 0 }, .{ 0.5, 0.5, 0.5 }, .{ 1.0, 0.76, 0.33 }, 0.5, 0.1, mat_gold);
    _ = try d.addBox(.{ 0.01, 5.5, 0 }, .{ 0.5, 0.5, 0.5 }, .{ 0.35, 0.8, 0.35 }, 0.5, 0.1, mat_rubber);
    // Side boxes
    _ = try d.addBox(.{ 3, 1, 0 }, .{ 0.75, 0.75, 0.75 }, .{ 0.95, 0.64, 0.54 }, 0.3, 0.3, mat_copper);
    _ = try d.addBox(.{ -3, 2, 1 }, .{ 0.6, 0.6, 0.6 }, .{ 0.55, 0.56, 0.55 }, 0.8, 0.0, mat_chrome);
    // Spheres
    _ = try d.addSphere(.{ 1.5, 3, 0 }, 0.4, .{ 0.8, 0.85, 1.0 }, 128 * 4, 0.2, 0.97, mat_glass);
    _ = try d.addSphere(.{ -1.0, 3, 0 }, 0.5, .{ 1.0, 0.9, 0.7 }, 128 * 4, 0.6, 0.4, mat_emissive);

    try d.upload();

    // Profiler (must be before Physics.init so it can register sections)
    var prof = Profiler.init();
    const p_frame = prof.addSection("frame_total");
    const p_poll = prof.addSection("poll_events");
    const p_physics = prof.addSection("physics");
    const p_bvh = prof.addSection("bvh_build");
    const p_raytrace = prof.addSection("raytrace");
    const p_begin_frame = prof.addSection("begin_frame");
    const p_draw = prof.addSection("draw");
    const p_gui = prof.addSection("gui");
    const p_end_frame = prof.addSection("end_frame");
    defer prof.printSummary();

    var physics = try Physics.init(&vk_ctx, &d, allocator, &prof);
    defer physics.deinit();

    var bvh = if (comptime BVH != void) try BVH.init(&vk_ctx, &d, allocator, &prof) else {};
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
        prof.begin(p_frame);

        const now = c.glfwGetTime();
        const dt: f32 = @floatCast(now - last_time);
        last_time = now;
        if (dt > 0) {
            const alpha: f32 = if (frame < 10) 1.0 else 0.1;
            fps_smooth = (1.0 - alpha) * fps_smooth + alpha * (1.0 / dt);
        }
        if (comptime Gui != void) {
            gui.fps = fps_smooth;
        }

        prof.begin(p_poll);
        scene.pollEvents();
        camera.update(window, dt);
        prof.end(p_poll);

        // Physics
        prof.begin(p_physics);
        try physics.step(d.num_bodies, d.num_vertices, d.num_hull_verts, 10, 1.0 / 60.0, .{ 0, -9.81, 0 }, &prof);
        prof.end(p_physics);

        if (comptime raytrace_mode) {
            prof.begin(p_bvh);
            try bvh.build(d.num_triangles, &prof);
            prof.end(p_bvh);

            prof.begin(p_raytrace);
            try rt.render(camera.pos, camera.direction(), camera.right(), camera.up(), d.num_triangles);
            prof.end(p_raytrace);

            prof.begin(p_begin_frame);
            try scene.beginFrame();
            prof.end(p_begin_frame);

            prof.begin(p_draw);
            scene.blitRtImage();
            prof.end(p_draw);

            prof.begin(p_end_frame);
            try scene.endFrame();
            prof.end(p_end_frame);

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
            prof.begin(p_begin_frame);
            const mvp = camera.mvp(aspect);
            try scene.beginFrame();
            prof.end(p_begin_frame);

            prof.begin(p_draw);
            scene.draw(&d, &mvp);
            prof.end(p_draw);

            if (comptime Gui != void) {
                prof.begin(p_gui);
                const counters = physics.readCounters(&d) catch .{ 0, 0 };
                gui.pairs = counters[0];
                gui.contacts = counters[1];
                gui.render(&d, scene.cmd_buffers[scene.current_frame]);
                prof.end(p_gui);
            }

            prof.begin(p_end_frame);
            try scene.endFrame();
            prof.end(p_end_frame);
        }

        prof.end(p_frame);
        prof.endFrame();
    }
}
