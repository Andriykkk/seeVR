// Single @cImport so all Vulkan/GLFW/shaderc types are shared
pub const c = @cImport({
    @cDefine("VK_USE_PLATFORM_XLIB_KHR", "1");
    @cDefine("GLFW_INCLUDE_VULKAN", "1");
    @cInclude("GLFW/glfw3.h");
    @cInclude("vulkan/vulkan.h");
    @cInclude("shaderc/shaderc.h");
});
