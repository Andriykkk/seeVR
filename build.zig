const std = @import("std");

fn addImgui(mod: *std.Build.Module, b: *std.Build) void {
    mod.addIncludePath(b.path("src/libraries/imgui"));
    mod.addIncludePath(b.path("src/libraries/imgui/backends"));
    mod.addIncludePath(b.path("src/libraries"));

    mod.addCSourceFiles(.{
        .files = &.{
            "src/libraries/imgui/imgui.cpp",
            "src/libraries/imgui/imgui_draw.cpp",
            "src/libraries/imgui/imgui_tables.cpp",
            "src/libraries/imgui/imgui_widgets.cpp",
            "src/libraries/imgui/backends/imgui_impl_glfw.cpp",
            "src/libraries/imgui/backends/imgui_impl_vulkan.cpp",
            "src/libraries/imgui_wrapper.cpp",
        },
        .flags = &.{},
    });
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const enable_imgui = b.option(bool, "imgui", "Enable ImGui overlay (default: true)") orelse true;

    const mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const options = b.addOptions();
    options.addOption(bool, "enable_imgui", enable_imgui);
    mod.addOptions("build_options", options);

    mod.linkSystemLibrary("glfw", .{});
    mod.linkSystemLibrary("vulkan", .{});
    mod.linkSystemLibrary("shaderc", .{});

    if (enable_imgui) {
        mod.linkSystemLibrary("stdc++", .{});
        addImgui(mod, b);
    }

    const exe = b.addExecutable(.{
        .name = "pgs_boxes",
        .root_module = mod,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.setCwd(b.path("."));
    run_cmd.step.dependOn(b.getInstallStep());
    const run_step = b.step("run", "Run the simulation");
    run_step.dependOn(&run_cmd.step);
}
