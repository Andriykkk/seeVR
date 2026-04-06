#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "imgui_wrapper.h"

extern "C" {

void* imgui_create_context(void) {
    return ImGui::CreateContext();
}

void imgui_destroy_context(void) {
    ImGui::DestroyContext();
}

void imgui_init_glfw(GLFWwindow* window) {
    ImGui_ImplGlfw_InitForVulkan(window, true);
}

void imgui_shutdown_glfw(void) {
    ImGui_ImplGlfw_Shutdown();
}

void imgui_init_vulkan(const ImGuiVulkanInfo* info) {
    ImGui_ImplVulkan_InitInfo vk_info = {};
    vk_info.ApiVersion = VK_API_VERSION_1_2;
    vk_info.Instance = info->instance;
    vk_info.PhysicalDevice = info->physical_device;
    vk_info.Device = info->device;
    vk_info.QueueFamily = info->queue_family;
    vk_info.Queue = info->queue;
    vk_info.DescriptorPool = info->descriptor_pool;
    vk_info.MinImageCount = info->image_count;
    vk_info.ImageCount = info->image_count;
    vk_info.PipelineInfoMain.RenderPass = info->render_pass;
    vk_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ImGui_ImplVulkan_Init(&vk_info);
}

void imgui_shutdown_vulkan(void) {
    ImGui_ImplVulkan_Shutdown();
}

void imgui_new_frame(void) {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void imgui_render(void) {
    ImGui::Render();
}

void imgui_render_draw_data(VkCommandBuffer cmd) {
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

int imgui_begin(const char* name) {
    return ImGui::Begin(name) ? 1 : 0;
}

void imgui_end(void) {
    ImGui::End();
}

void imgui_text(const char* text) {
    ImGui::TextUnformatted(text);
}

int imgui_slider_float(const char* label, float* v, float v_min, float v_max) {
    return ImGui::SliderFloat(label, v, v_min, v_max) ? 1 : 0;
}

int imgui_checkbox(const char* label, int* v) {
    bool b = *v != 0;
    bool changed = ImGui::Checkbox(label, &b);
    *v = b ? 1 : 0;
    return changed ? 1 : 0;
}

} // extern "C"
