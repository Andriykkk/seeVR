const std = @import("std");
const c = @import("c.zig").c;
const Vulkan = @import("vulkan.zig").Vulkan;
const Data = @import("data.zig").Data;

const MAX_DEBUG_VERTS = 16384;

pub const DebugDraw = struct {
    vk: *Vulkan,
    pipeline: c.VkPipeline,
    pipeline_layout: c.VkPipelineLayout,
    vert_buf: Vulkan.Buffer,
    color_buf: Vulkan.Buffer,

    // CPU staging
    verts: []f32,
    colors: []f32,
    num_verts: u32,

    pub fn init(vk: *Vulkan, render_pass: c.VkRenderPass, alloc: std.mem.Allocator) !DebugDraw {
        const vert_shader = try vk.getShader("src/shaders/debug.vert", .vertex, alloc);
        const frag_shader = try vk.getShader("src/shaders/debug.frag", .fragment, alloc);

        const push_range = c.VkPushConstantRange{
            .stageFlags = c.VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = 64, // mat4
        };

        var layout: c.VkPipelineLayout = null;
        _ = c.vkCreatePipelineLayout(vk.device, &c.VkPipelineLayoutCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = null, .flags = 0,
            .setLayoutCount = 0, .pSetLayouts = null,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &push_range,
        }, null, &layout);

        const binding_descs = [2]c.VkVertexInputBindingDescription{
            .{ .binding = 0, .stride = 12, .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX },
            .{ .binding = 1, .stride = 12, .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX },
        };
        const attr_descs = [2]c.VkVertexInputAttributeDescription{
            .{ .location = 0, .binding = 0, .format = c.VK_FORMAT_R32G32B32_SFLOAT, .offset = 0 },
            .{ .location = 1, .binding = 1, .format = c.VK_FORMAT_R32G32B32_SFLOAT, .offset = 0 },
        };

        const stages = [2]c.VkPipelineShaderStageCreateInfo{
            .{ .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = null, .flags = 0, .stage = c.VK_SHADER_STAGE_VERTEX_BIT, .module = vert_shader, .pName = "main", .pSpecializationInfo = null },
            .{ .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = null, .flags = 0, .stage = c.VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag_shader, .pName = "main", .pSpecializationInfo = null },
        };

        var pipeline: c.VkPipeline = null;
        _ = c.vkCreateGraphicsPipelines(vk.device, null, 1, &c.VkGraphicsPipelineCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = null, .flags = 0,
            .stageCount = 2, .pStages = &stages,
            .pVertexInputState = &c.VkPipelineVertexInputStateCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .pNext = null, .flags = 0,
                .vertexBindingDescriptionCount = 2, .pVertexBindingDescriptions = &binding_descs,
                .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = &attr_descs,
            },
            .pInputAssemblyState = &c.VkPipelineInputAssemblyStateCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .pNext = null, .flags = 0,
                .topology = c.VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                .primitiveRestartEnable = c.VK_FALSE,
            },
            .pTessellationState = null,
            .pViewportState = &c.VkPipelineViewportStateCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .pNext = null, .flags = 0,
                .viewportCount = 1, .pViewports = null,
                .scissorCount = 1, .pScissors = null,
            },
            .pRasterizationState = &c.VkPipelineRasterizationStateCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .pNext = null, .flags = 0,
                .depthClampEnable = c.VK_FALSE,
                .rasterizerDiscardEnable = c.VK_FALSE,
                .polygonMode = c.VK_POLYGON_MODE_FILL,
                .cullMode = c.VK_CULL_MODE_NONE,
                .frontFace = c.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                .depthBiasEnable = c.VK_FALSE,
                .depthBiasConstantFactor = 0, .depthBiasClamp = 0, .depthBiasSlopeFactor = 0,
                .lineWidth = 2.0,
            },
            .pMultisampleState = &c.VkPipelineMultisampleStateCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .pNext = null, .flags = 0,
                .rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT,
                .sampleShadingEnable = c.VK_FALSE,
                .minSampleShading = 1.0,
                .pSampleMask = null,
                .alphaToCoverageEnable = c.VK_FALSE,
                .alphaToOneEnable = c.VK_FALSE,
            },
            .pDepthStencilState = &c.VkPipelineDepthStencilStateCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .pNext = null, .flags = 0,
                .depthTestEnable = c.VK_TRUE,
                .depthWriteEnable = c.VK_FALSE,
                .depthCompareOp = c.VK_COMPARE_OP_LESS_OR_EQUAL,
                .depthBoundsTestEnable = c.VK_FALSE,
                .stencilTestEnable = c.VK_FALSE,
                .front = std.mem.zeroes(c.VkStencilOpState),
                .back = std.mem.zeroes(c.VkStencilOpState),
                .minDepthBounds = 0, .maxDepthBounds = 1,
            },
            .pColorBlendState = &c.VkPipelineColorBlendStateCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .pNext = null, .flags = 0,
                .logicOpEnable = c.VK_FALSE, .logicOp = c.VK_LOGIC_OP_COPY,
                .attachmentCount = 1,
                .pAttachments = &c.VkPipelineColorBlendAttachmentState{
                    .blendEnable = c.VK_FALSE,
                    .srcColorBlendFactor = c.VK_BLEND_FACTOR_ONE,
                    .dstColorBlendFactor = c.VK_BLEND_FACTOR_ZERO,
                    .colorBlendOp = c.VK_BLEND_OP_ADD,
                    .srcAlphaBlendFactor = c.VK_BLEND_FACTOR_ONE,
                    .dstAlphaBlendFactor = c.VK_BLEND_FACTOR_ZERO,
                    .alphaBlendOp = c.VK_BLEND_OP_ADD,
                    .colorWriteMask = c.VK_COLOR_COMPONENT_R_BIT | c.VK_COLOR_COMPONENT_G_BIT | c.VK_COLOR_COMPONENT_B_BIT | c.VK_COLOR_COMPONENT_A_BIT,
                },
                .blendConstants = .{ 0, 0, 0, 0 },
            },
            .pDynamicState = &c.VkPipelineDynamicStateCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .pNext = null, .flags = 0,
                .dynamicStateCount = 2,
                .pDynamicStates = &[2]c.VkDynamicState{ c.VK_DYNAMIC_STATE_VIEWPORT, c.VK_DYNAMIC_STATE_SCISSOR },
            },
            .layout = layout,
            .renderPass = render_pass,
            .subpass = 0,
            .basePipelineHandle = null,
            .basePipelineIndex = -1,
        }, null, &pipeline);

        return .{
            .vk = vk,
            .pipeline = pipeline,
            .pipeline_layout = layout,
            .vert_buf = try vk.createBuffer(MAX_DEBUG_VERTS * 3 * @sizeOf(f32), Vulkan.USAGE_VERTEX),
            .color_buf = try vk.createBuffer(MAX_DEBUG_VERTS * 3 * @sizeOf(f32), Vulkan.USAGE_VERTEX),
            .verts = try alloc.alloc(f32, MAX_DEBUG_VERTS * 3),
            .colors = try alloc.alloc(f32, MAX_DEBUG_VERTS * 3),
            .num_verts = 0,
        };
    }

    pub fn clear(self: *DebugDraw) void {
        self.num_verts = 0;
    }

    /// Add a line from p0 to p1 with given color
    pub fn line(self: *DebugDraw, p0: [3]f32, p1: [3]f32, col: [3]f32) void {
        if (self.num_verts + 2 > MAX_DEBUG_VERTS) return;
        const i = self.num_verts;
        self.verts[i * 3 + 0] = p0[0]; self.verts[i * 3 + 1] = p0[1]; self.verts[i * 3 + 2] = p0[2];
        self.verts[i * 3 + 3] = p1[0]; self.verts[i * 3 + 4] = p1[1]; self.verts[i * 3 + 5] = p1[2];
        self.colors[i * 3 + 0] = col[0]; self.colors[i * 3 + 1] = col[1]; self.colors[i * 3 + 2] = col[2];
        self.colors[i * 3 + 3] = col[0]; self.colors[i * 3 + 4] = col[1]; self.colors[i * 3 + 5] = col[2];
        self.num_verts += 2;
    }

    /// Add a cross marker at position p with given size and color
    pub fn point(self: *DebugDraw, p: [3]f32, size: f32, col: [3]f32) void {
        const s = size;
        self.line(.{ p[0] - s, p[1], p[2] }, .{ p[0] + s, p[1], p[2] }, col);
        self.line(.{ p[0], p[1] - s, p[2] }, .{ p[0], p[1] + s, p[2] }, col);
        self.line(.{ p[0], p[1], p[2] - s }, .{ p[0], p[1], p[2] + s }, col);
    }

    /// Read contact data from GPU and visualize contacts + normals
    pub fn drawContacts(self: *DebugDraw, data: *const Data) void {
        var counters: [2]u32 = .{ 0, 0 };
        self.vk.readBuffer(data.atomic_counters, @ptrCast(&counters), @sizeOf([2]u32)) catch return;
        const nc = @min(counters[1], @import("data.zig").MAX_CONTACTS);
        if (nc == 0) return;

        const max_draw: u32 = @min(nc, 256); // cap for readback size
        var pos_buf: [256 * 3]f32 = undefined;
        var nrm_buf: [256 * 3]f32 = undefined;
        var pen_buf: [256]f32 = undefined;

        self.vk.readBuffer(data.contact_pos, @ptrCast(&pos_buf), max_draw * 3 * @sizeOf(f32)) catch return;
        self.vk.readBuffer(data.contact_normal, @ptrCast(&nrm_buf), max_draw * 3 * @sizeOf(f32)) catch return;
        self.vk.readBuffer(data.contact_penetration, @ptrCast(&pen_buf), max_draw * @sizeOf(f32)) catch return;

        for (0..max_draw) |i| {
            const px = pos_buf[i * 3 + 0];
            const py = pos_buf[i * 3 + 1];
            const pz = pos_buf[i * 3 + 2];
            const nx = nrm_buf[i * 3 + 0];
            const ny = nrm_buf[i * 3 + 1];
            const nz = nrm_buf[i * 3 + 2];

            // Yellow cross at contact point
            self.point(.{ px, py, pz }, 0.05, .{ 1, 1, 0 });

            // Green line for normal (scaled by 0.3)
            self.line(
                .{ px, py, pz },
                .{ px + nx * 0.3, py + ny * 0.3, pz + nz * 0.3 },
                .{ 0, 1, 0 },
            );

            // Red line for penetration depth (along -normal)
            const pen = pen_buf[i];
            if (pen > 0.001) {
                self.line(
                    .{ px, py, pz },
                    .{ px - nx * pen, py - ny * pen, pz - nz * pen },
                    .{ 1, 0, 0 },
                );
            }
        }
    }

    /// Upload and render debug lines
    pub fn render(self: *DebugDraw, cmd: c.VkCommandBuffer, mvp: *const [16]f32, extent: c.VkExtent2D) void {
        if (self.num_verts == 0) return;

        // Upload
        self.vk.uploadSlice(self.vert_buf, f32, self.verts[0 .. self.num_verts * 3]) catch return;
        self.vk.uploadSlice(self.color_buf, f32, self.colors[0 .. self.num_verts * 3]) catch return;

        c.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline);
        c.vkCmdPushConstants(cmd, self.pipeline_layout, c.VK_SHADER_STAGE_VERTEX_BIT, 0, 64, @ptrCast(mvp));

        const viewport = c.VkViewport{ .x = 0, .y = 0, .width = @floatFromInt(extent.width), .height = @floatFromInt(extent.height), .minDepth = 0, .maxDepth = 1 };
        const scissor = c.VkRect2D{ .offset = .{ .x = 0, .y = 0 }, .extent = extent };
        c.vkCmdSetViewport(cmd, 0, 1, &viewport);
        c.vkCmdSetScissor(cmd, 0, 1, &scissor);

        const bufs = [2]c.VkBuffer{ self.vert_buf.handle, self.color_buf.handle };
        const offsets = [2]c.VkDeviceSize{ 0, 0 };
        c.vkCmdBindVertexBuffers(cmd, 0, 2, &bufs, &offsets);
        c.vkCmdDraw(cmd, self.num_verts, 1, 0, 0);
    }

    pub fn deinit(self: *DebugDraw) void {
        c.vkDestroyPipeline(self.vk.device, self.pipeline, null);
        c.vkDestroyPipelineLayout(self.vk.device, self.pipeline_layout, null);
        self.vk.destroyBuffer(self.vert_buf);
        self.vk.destroyBuffer(self.color_buf);
    }
};
