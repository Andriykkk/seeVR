#pragma once
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

// Context
void* imgui_create_context(void);
void imgui_destroy_context(void);

// GLFW backend
typedef struct GLFWwindow GLFWwindow;
void imgui_init_glfw(GLFWwindow* window);
void imgui_shutdown_glfw(void);

// Vulkan backend
typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    uint32_t queue_family;
    VkQueue queue;
    VkDescriptorPool descriptor_pool;
    VkRenderPass render_pass;
    uint32_t image_count;
} ImGuiVulkanInfo;

void imgui_init_vulkan(const ImGuiVulkanInfo* info);
void imgui_shutdown_vulkan(void);

// Frame
void imgui_new_frame(void);
void imgui_render(void);
void imgui_render_draw_data(VkCommandBuffer cmd);

// Widgets
int imgui_begin(const char* name);
void imgui_end(void);
void imgui_text(const char* text);
int imgui_slider_float(const char* label, float* v, float v_min, float v_max);
int imgui_checkbox(const char* label, int* v);

#ifdef __cplusplus
}
#endif
