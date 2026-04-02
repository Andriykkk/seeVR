const std = @import("std");
const c = @import("c.zig").c;
const Vulkan = @import("vulkan.zig").Vulkan;
const Data = @import("data.zig").Data;

pub const Mode = enum {
    headless,
    raster,
    raytrace,
};

const MAX_FRAMES_IN_FLIGHT = 2;

pub const Scene = struct {
    mode: Mode,
    vk: *Vulkan,
    window: ?*c.GLFWwindow,
    width: u32,
    height: u32,

    // Swapchain
    swapchain: c.VkSwapchainKHR,
    swapchain_images: [8]c.VkImage,
    swapchain_views: [8]c.VkImageView,
    swapchain_format: c.VkFormat,
    swapchain_count: u32,
    extent: c.VkExtent2D,

    // Render pass + framebuffers
    render_pass: c.VkRenderPass,
    framebuffers: [8]c.VkFramebuffer,

    // Raster pipeline
    pipeline_layout: c.VkPipelineLayout,
    pipeline: c.VkPipeline,

    // Command buffers
    cmd_buffers: [MAX_FRAMES_IN_FLIGHT]c.VkCommandBuffer,

    // Sync
    image_available: [MAX_FRAMES_IN_FLIGHT]c.VkSemaphore,
    render_finished: [MAX_FRAMES_IN_FLIGHT]c.VkSemaphore,
    in_flight: [MAX_FRAMES_IN_FLIGHT]c.VkFence,
    current_frame: u32,
    current_image: u32,

    pub fn init(vk: *Vulkan, mode: Mode, window: ?*c.GLFWwindow, width: u32, height: u32, allocator: std.mem.Allocator) !Scene {
        var self = Scene{
            .mode = mode,
            .vk = vk,
            .window = window,
            .width = width,
            .height = height,
            .swapchain = null,
            .swapchain_images = undefined,
            .swapchain_views = undefined,
            .swapchain_format = c.VK_FORMAT_B8G8R8A8_SRGB,
            .swapchain_count = 0,
            .extent = .{ .width = width, .height = height },
            .render_pass = null,
            .framebuffers = undefined,
            .pipeline_layout = null,
            .pipeline = null,
            .cmd_buffers = undefined,
            .image_available = undefined,
            .render_finished = undefined,
            .in_flight = undefined,
            .current_frame = 0,
            .current_image = 0,
        };

        if (mode == .headless) return self;

        // --- Swapchain ---
        var caps: c.VkSurfaceCapabilitiesKHR = undefined;
        _ = c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.physical_device, vk.surface, &caps);

        var format_count: u32 = 0;
        _ = c.vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical_device, vk.surface, &format_count, null);
        var formats: [16]c.VkSurfaceFormatKHR = undefined;
        var fc: u32 = @min(format_count, 16);
        _ = c.vkGetPhysicalDeviceSurfaceFormatsKHR(vk.physical_device, vk.surface, &fc, &formats);

        self.swapchain_format = formats[0].format;
        for (formats[0..fc]) |f| {
            if (f.format == c.VK_FORMAT_B8G8R8A8_SRGB and f.colorSpace == c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                self.swapchain_format = f.format;
                break;
            }
        }

        self.extent = if (caps.currentExtent.width != 0xFFFFFFFF)
            caps.currentExtent
        else
            c.VkExtent2D{ .width = width, .height = height };

        const image_count = @min(caps.minImageCount + 1, if (caps.maxImageCount > 0) caps.maxImageCount else 8);

        if (c.vkCreateSwapchainKHR(vk.device, &c.VkSwapchainCreateInfoKHR{
            .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .pNext = null,
            .flags = 0,
            .surface = vk.surface,
            .minImageCount = image_count,
            .imageFormat = self.swapchain_format,
            .imageColorSpace = c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
            .imageExtent = self.extent,
            .imageArrayLayers = 1,
            .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
            .preTransform = caps.currentTransform,
            .compositeAlpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = c.VK_PRESENT_MODE_FIFO_KHR,
            .clipped = c.VK_TRUE,
            .oldSwapchain = null,
        }, null, &self.swapchain) != c.VK_SUCCESS)
            return error.SwapchainCreateFailed;

        self.swapchain_count = 8;
        _ = c.vkGetSwapchainImagesKHR(vk.device, self.swapchain, &self.swapchain_count, &self.swapchain_images);

        // Image views
        for (0..self.swapchain_count) |i| {
            if (c.vkCreateImageView(vk.device, &c.VkImageViewCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = null,
                .flags = 0,
                .image = self.swapchain_images[i],
                .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
                .format = self.swapchain_format,
                .components = .{
                    .r = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                    .g = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                    .b = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                    .a = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                },
                .subresourceRange = .{
                    .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            }, null, &self.swapchain_views[i]) != c.VK_SUCCESS)
                return error.ImageViewCreateFailed;
        }

        // --- Render pass ---
        if (c.vkCreateRenderPass(vk.device, &c.VkRenderPassCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .attachmentCount = 1,
            .pAttachments = &c.VkAttachmentDescription{
                .flags = 0,
                .format = self.swapchain_format,
                .samples = c.VK_SAMPLE_COUNT_1_BIT,
                .loadOp = c.VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = c.VK_ATTACHMENT_STORE_OP_STORE,
                .stencilLoadOp = c.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .stencilStoreOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
                .finalLayout = c.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            },
            .subpassCount = 1,
            .pSubpasses = &c.VkSubpassDescription{
                .flags = 0,
                .pipelineBindPoint = c.VK_PIPELINE_BIND_POINT_GRAPHICS,
                .inputAttachmentCount = 0,
                .pInputAttachments = null,
                .colorAttachmentCount = 1,
                .pColorAttachments = &c.VkAttachmentReference{
                    .attachment = 0,
                    .layout = c.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                },
                .pResolveAttachments = null,
                .pDepthStencilAttachment = null,
                .preserveAttachmentCount = 0,
                .pPreserveAttachments = null,
            },
            .dependencyCount = 0,
            .pDependencies = null,
        }, null, &self.render_pass) != c.VK_SUCCESS)
            return error.RenderPassCreateFailed;

        // --- Framebuffers ---
        for (0..self.swapchain_count) |i| {
            if (c.vkCreateFramebuffer(vk.device, &c.VkFramebufferCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .pNext = null,
                .flags = 0,
                .renderPass = self.render_pass,
                .attachmentCount = 1,
                .pAttachments = &self.swapchain_views[i],
                .width = self.extent.width,
                .height = self.extent.height,
                .layers = 1,
            }, null, &self.framebuffers[i]) != c.VK_SUCCESS)
                return error.FramebufferCreateFailed;
        }

        // --- Graphics pipeline (raster mode) ---
        if (mode == .raster) {
            const vert_mod = try vk.getShader("shaders/triangle.vert", .vertex, allocator);
            const frag_mod = try vk.getShader("shaders/triangle.frag", .fragment, allocator);

            const shader_stages = [2]c.VkPipelineShaderStageCreateInfo{
                .{
                    .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = null,
                    .flags = 0,
                    .stage = c.VK_SHADER_STAGE_VERTEX_BIT,
                    .module = vert_mod,
                    .pName = "main",
                    .pSpecializationInfo = null,
                },
                .{
                    .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = null,
                    .flags = 0,
                    .stage = c.VK_SHADER_STAGE_FRAGMENT_BIT,
                    .module = frag_mod,
                    .pName = "main",
                    .pSpecializationInfo = null,
                },
            };

            // Vertex input: location 0 = vec3 pos, location 1 = vec3 color (separate buffers)
            const bindings = [2]c.VkVertexInputBindingDescription{
                .{ .binding = 0, .stride = 3 * @sizeOf(f32), .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX },
                .{ .binding = 1, .stride = 3 * @sizeOf(f32), .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX },
            };
            const attrs = [2]c.VkVertexInputAttributeDescription{
                .{ .location = 0, .binding = 0, .format = c.VK_FORMAT_R32G32B32_SFLOAT, .offset = 0 },
                .{ .location = 1, .binding = 1, .format = c.VK_FORMAT_R32G32B32_SFLOAT, .offset = 0 },
            };

            const vertex_input = c.VkPipelineVertexInputStateCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .pNext = null,
                .flags = 0,
                .vertexBindingDescriptionCount = 2,
                .pVertexBindingDescriptions = &bindings,
                .vertexAttributeDescriptionCount = 2,
                .pVertexAttributeDescriptions = &attrs,
            };

            const input_assembly = c.VkPipelineInputAssemblyStateCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .pNext = null,
                .flags = 0,
                .topology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                .primitiveRestartEnable = c.VK_FALSE,
            };

            const viewport = c.VkViewport{
                .x = 0, .y = 0,
                .width = @floatFromInt(self.extent.width),
                .height = @floatFromInt(self.extent.height),
                .minDepth = 0, .maxDepth = 1,
            };
            const scissor = c.VkRect2D{ .offset = .{ .x = 0, .y = 0 }, .extent = self.extent };

            const viewport_state = c.VkPipelineViewportStateCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .pNext = null,
                .flags = 0,
                .viewportCount = 1,
                .pViewports = &viewport,
                .scissorCount = 1,
                .pScissors = &scissor,
            };

            const rasterizer = c.VkPipelineRasterizationStateCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .pNext = null,
                .flags = 0,
                .depthClampEnable = c.VK_FALSE,
                .rasterizerDiscardEnable = c.VK_FALSE,
                .polygonMode = c.VK_POLYGON_MODE_FILL,
                .cullMode = c.VK_CULL_MODE_BACK_BIT,
                .frontFace = c.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                .depthBiasEnable = c.VK_FALSE,
                .depthBiasConstantFactor = 0,
                .depthBiasClamp = 0,
                .depthBiasSlopeFactor = 0,
                .lineWidth = 1.0,
            };

            const multisampling = c.VkPipelineMultisampleStateCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .pNext = null,
                .flags = 0,
                .rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT,
                .sampleShadingEnable = c.VK_FALSE,
                .minSampleShading = 1,
                .pSampleMask = null,
                .alphaToCoverageEnable = c.VK_FALSE,
                .alphaToOneEnable = c.VK_FALSE,
            };

            const blend_attachment = c.VkPipelineColorBlendAttachmentState{
                .blendEnable = c.VK_FALSE,
                .srcColorBlendFactor = c.VK_BLEND_FACTOR_ONE,
                .dstColorBlendFactor = c.VK_BLEND_FACTOR_ZERO,
                .colorBlendOp = c.VK_BLEND_OP_ADD,
                .srcAlphaBlendFactor = c.VK_BLEND_FACTOR_ONE,
                .dstAlphaBlendFactor = c.VK_BLEND_FACTOR_ZERO,
                .alphaBlendOp = c.VK_BLEND_OP_ADD,
                .colorWriteMask = c.VK_COLOR_COMPONENT_R_BIT | c.VK_COLOR_COMPONENT_G_BIT | c.VK_COLOR_COMPONENT_B_BIT | c.VK_COLOR_COMPONENT_A_BIT,
            };

            const color_blend = c.VkPipelineColorBlendStateCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .pNext = null,
                .flags = 0,
                .logicOpEnable = c.VK_FALSE,
                .logicOp = c.VK_LOGIC_OP_COPY,
                .attachmentCount = 1,
                .pAttachments = &blend_attachment,
                .blendConstants = .{ 0, 0, 0, 0 },
            };

            // Push constant: mat4 MVP
            const push_range = c.VkPushConstantRange{
                .stageFlags = c.VK_SHADER_STAGE_VERTEX_BIT,
                .offset = 0,
                .size = 64, // sizeof(mat4)
            };

            if (c.vkCreatePipelineLayout(vk.device, &c.VkPipelineLayoutCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .pNext = null,
                .flags = 0,
                .setLayoutCount = 0,
                .pSetLayouts = null,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &push_range,
            }, null, &self.pipeline_layout) != c.VK_SUCCESS)
                return error.PipelineLayoutCreateFailed;

            if (c.vkCreateGraphicsPipelines(vk.device, null, 1, &c.VkGraphicsPipelineCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .pNext = null,
                .flags = 0,
                .stageCount = 2,
                .pStages = &shader_stages,
                .pVertexInputState = &vertex_input,
                .pInputAssemblyState = &input_assembly,
                .pTessellationState = null,
                .pViewportState = &viewport_state,
                .pRasterizationState = &rasterizer,
                .pMultisampleState = &multisampling,
                .pDepthStencilState = null,
                .pColorBlendState = &color_blend,
                .pDynamicState = null,
                .layout = self.pipeline_layout,
                .renderPass = self.render_pass,
                .subpass = 0,
                .basePipelineHandle = null,
                .basePipelineIndex = -1,
            }, null, &self.pipeline) != c.VK_SUCCESS)
                return error.PipelineCreateFailed;
        }

        // --- Command buffers ---
        var cmd_bufs: [MAX_FRAMES_IN_FLIGHT]c.VkCommandBuffer = undefined;
        _ = c.vkAllocateCommandBuffers(vk.device, &c.VkCommandBufferAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = null,
            .commandPool = vk.cmd_pool,
            .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
        }, &cmd_bufs);
        self.cmd_buffers = cmd_bufs;

        // --- Sync objects ---
        for (0..MAX_FRAMES_IN_FLIGHT) |i| {
            if (c.vkCreateSemaphore(vk.device, &c.VkSemaphoreCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = null, .flags = 0,
            }, null, &self.image_available[i]) != c.VK_SUCCESS) return error.SyncCreateFailed;
            if (c.vkCreateSemaphore(vk.device, &c.VkSemaphoreCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = null, .flags = 0,
            }, null, &self.render_finished[i]) != c.VK_SUCCESS) return error.SyncCreateFailed;
            if (c.vkCreateFence(vk.device, &c.VkFenceCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .pNext = null, .flags = c.VK_FENCE_CREATE_SIGNALED_BIT,
            }, null, &self.in_flight[i]) != c.VK_SUCCESS) return error.SyncCreateFailed;
        }

        std.debug.print("Scene init OK (mode={s}, {}x{}, {} images)\n", .{
            @tagName(mode), self.extent.width, self.extent.height, self.swapchain_count,
        });

        return self;
    }

    // --- Frame loop ---

    pub fn beginFrame(self: *Scene) !void {
        const dev = self.vk.device;
        const f = self.current_frame;

        _ = c.vkWaitForFences(dev, 1, &self.in_flight[f], c.VK_TRUE, std.math.maxInt(u64));
        _ = c.vkResetFences(dev, 1, &self.in_flight[f]);

        _ = c.vkAcquireNextImageKHR(dev, self.swapchain, std.math.maxInt(u64), self.image_available[f], null, &self.current_image);

        _ = c.vkResetCommandBuffer(self.cmd_buffers[f], 0);
        _ = c.vkBeginCommandBuffer(self.cmd_buffers[f], &c.VkCommandBufferBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = 0,
            .pInheritanceInfo = null,
        });

        const clear_color = c.VkClearValue{ .color = .{ .float32 = .{ 0.1, 0.1, 0.12, 1.0 } } };
        c.vkCmdBeginRenderPass(self.cmd_buffers[f], &c.VkRenderPassBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = null,
            .renderPass = self.render_pass,
            .framebuffer = self.framebuffers[self.current_image],
            .renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = self.extent },
            .clearValueCount = 1,
            .pClearValues = &clear_color,
        }, c.VK_SUBPASS_CONTENTS_INLINE);
    }

    pub fn draw(self: *Scene, d: *const Data, mvp: *const [16]f32) void {
        const cmd = self.cmd_buffers[self.current_frame];

        c.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline);

        // Push MVP matrix
        c.vkCmdPushConstants(cmd, self.pipeline_layout, c.VK_SHADER_STAGE_VERTEX_BIT, 0, 64, @ptrCast(mvp));

        // Bind vertex buffers (binding 0 = positions, binding 1 = colors)
        const vertex_buffers = [2]c.VkBuffer{ d.vertices.handle, d.colors.handle };
        const offsets = [2]c.VkDeviceSize{ 0, 0 };
        c.vkCmdBindVertexBuffers(cmd, 0, 2, &vertex_buffers, &offsets);

        // Bind index buffer
        c.vkCmdBindIndexBuffer(cmd, d.indices.handle, 0, c.VK_INDEX_TYPE_UINT32);

        // Draw
        c.vkCmdDrawIndexed(cmd, d.num_triangles * 3, 1, 0, 0, 0);
    }

    pub fn endFrame(self: *Scene) !void {
        const cmd = self.cmd_buffers[self.current_frame];

        c.vkCmdEndRenderPass(cmd);
        _ = c.vkEndCommandBuffer(cmd);

        const wait_stage: c.VkPipelineStageFlags = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        _ = c.vkQueueSubmit(self.vk.graphics_queue, 1, &c.VkSubmitInfo{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &self.image_available[self.current_frame],
            .pWaitDstStageMask = &wait_stage,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &self.render_finished[self.current_frame],
        }, self.in_flight[self.current_frame]);

        _ = c.vkQueuePresentKHR(self.vk.graphics_queue, &c.VkPresentInfoKHR{
            .sType = c.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = null,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &self.render_finished[self.current_frame],
            .swapchainCount = 1,
            .pSwapchains = &self.swapchain,
            .pImageIndices = &self.current_image,
            .pResults = null,
        });

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    pub fn shouldClose(self: *const Scene) bool {
        if (self.window) |w| return c.glfwWindowShouldClose(w) != 0;
        return false;
    }

    pub fn pollEvents(_: *const Scene) void {
        c.glfwPollEvents();
    }

    pub fn deinit(self: *Scene) void {
        const dev = self.vk.device;
        _ = c.vkDeviceWaitIdle(dev);

        for (0..MAX_FRAMES_IN_FLIGHT) |i| {
            c.vkDestroySemaphore(dev, self.image_available[i], null);
            c.vkDestroySemaphore(dev, self.render_finished[i], null);
            c.vkDestroyFence(dev, self.in_flight[i], null);
        }

        for (0..self.swapchain_count) |i| {
            c.vkDestroyFramebuffer(dev, self.framebuffers[i], null);
            c.vkDestroyImageView(dev, self.swapchain_views[i], null);
        }

        if (self.pipeline != null) c.vkDestroyPipeline(dev, self.pipeline, null);
        if (self.pipeline_layout != null) c.vkDestroyPipelineLayout(dev, self.pipeline_layout, null);
        if (self.render_pass != null) c.vkDestroyRenderPass(dev, self.render_pass, null);
        if (self.swapchain != null) c.vkDestroySwapchainKHR(dev, self.swapchain, null);
    }
};
