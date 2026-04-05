const c = @import("c.zig").c;

// Matches ImGuiVulkanInfo in imgui_wrapper.h
pub const VulkanInfo = extern struct {
    instance: c.VkInstance,
    physical_device: c.VkPhysicalDevice,
    device: c.VkDevice,
    queue_family: u32,
    queue: c.VkQueue,
    descriptor_pool: c.VkDescriptorPool,
    render_pass: c.VkRenderPass,
    image_count: u32,
};

// Context
pub extern fn imgui_create_context() ?*anyopaque;
pub extern fn imgui_destroy_context() void;

// GLFW backend
extern fn imgui_init_glfw(window: *anyopaque) void;
pub fn initGlfw(window: *c.GLFWwindow) void {
    imgui_init_glfw(@ptrCast(window));
}
pub extern fn imgui_shutdown_glfw() void;

// Vulkan backend
pub extern fn imgui_init_vulkan(info: *const VulkanInfo) void;
pub extern fn imgui_shutdown_vulkan() void;

// Frame
pub extern fn imgui_new_frame() void;
pub extern fn imgui_render() void;
extern fn imgui_render_draw_data(cmd: *anyopaque) void;
pub fn renderDrawData(cmd: c.VkCommandBuffer) void {
    imgui_render_draw_data(@ptrCast(cmd));
}

// Widgets
extern fn imgui_begin(name: [*:0]const u8) c_int;
pub fn begin(name: [*:0]const u8) bool {
    return imgui_begin(name) != 0;
}
pub extern fn imgui_end() void;
pub extern fn imgui_text(text: [*:0]const u8) void;
extern fn imgui_slider_float(label: [*:0]const u8, v: *f32, v_min: f32, v_max: f32) c_int;
pub fn sliderFloat(label: [*:0]const u8, v: *f32, v_min: f32, v_max: f32) bool {
    return imgui_slider_float(label, v, v_min, v_max) != 0;
}
extern fn imgui_checkbox(label: [*:0]const u8, v: *c_int) c_int;
pub fn checkbox(label: [*:0]const u8, v: *bool) bool {
    var ci: c_int = if (v.*) 1 else 0;
    const changed = imgui_checkbox(label, &ci) != 0;
    v.* = ci != 0;
    return changed;
}

// Re-export with nice names
pub const createContext = imgui_create_context;
pub const destroyContext = imgui_destroy_context;
pub const shutdownGlfw = imgui_shutdown_glfw;
pub const initVulkan = imgui_init_vulkan;
pub const shutdownVulkan = imgui_shutdown_vulkan;
pub const newFrame = imgui_new_frame;
pub const render = imgui_render;
pub const end = imgui_end;
pub const text = imgui_text;
