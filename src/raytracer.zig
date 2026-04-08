const std = @import("std");
const c = @import("c.zig").c;
const Vulkan = @import("vulkan.zig").Vulkan;
const Data = @import("data.zig").Data;

const PC = extern struct {
    cam_pos_x: f32, cam_pos_y: f32, cam_pos_z: f32,
    cam_dir_x: f32, cam_dir_y: f32, cam_dir_z: f32,
    cam_right_x: f32, cam_right_y: f32, cam_right_z: f32,
    cam_up_x: f32, cam_up_y: f32, cam_up_z: f32,
    width: u32, height: u32,
    frame_num: u32,
    num_triangles: u32,
};

const NUM_BINDINGS = 16; // 1 image + 15 storage buffers

pub const Raytracer = struct {
    vk: *Vulkan,
    pipeline: c.VkPipeline,
    layout: c.VkPipelineLayout,
    desc_set_layout: c.VkDescriptorSetLayout,
    desc_pool: c.VkDescriptorPool,
    desc_set: c.VkDescriptorSet,
    width: u32,
    height: u32,
    frame: u32,

    pub fn init(vk: *Vulkan, data: *const Data, rt_view: c.VkImageView, w: u32, h: u32, allocator: std.mem.Allocator) !Raytracer {
        const shader = try vk.getShader("src/shaders/raytrace.comp", .compute, allocator);

        var bindings: [NUM_BINDINGS]c.VkDescriptorSetLayoutBinding = undefined;
        // Binding 0: storage image
        bindings[0] = .{
            .binding = 0,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = null,
        };
        // Bindings 1..14: storage buffers
        for (1..NUM_BINDINGS) |i| {
            bindings[i] = .{
                .binding = @intCast(i),
                .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = null,
            };
        }

        var desc_layout: c.VkDescriptorSetLayout = null;
        if (c.vkCreateDescriptorSetLayout(vk.device, &c.VkDescriptorSetLayoutCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = null, .flags = 0,
            .bindingCount = NUM_BINDINGS, .pBindings = &bindings,
        }, null, &desc_layout) != c.VK_SUCCESS)
            return error.DescLayoutFailed;

        const push_range = c.VkPushConstantRange{ .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = @sizeOf(PC) };
        var pipe_layout: c.VkPipelineLayout = null;
        if (c.vkCreatePipelineLayout(vk.device, &c.VkPipelineLayoutCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pNext = null, .flags = 0,
            .setLayoutCount = 1, .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &push_range,
        }, null, &pipe_layout) != c.VK_SUCCESS)
            return error.PipeLayoutFailed;

        var pipeline: c.VkPipeline = null;
        if (c.vkCreateComputePipelines(vk.device, null, 1, &c.VkComputePipelineCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .pNext = null, .flags = 0,
            .stage = .{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = null, .flags = 0,
                .stage = c.VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main", .pSpecializationInfo = null,
            },
            .layout = pipe_layout, .basePipelineHandle = null, .basePipelineIndex = -1,
        }, null, &pipeline) != c.VK_SUCCESS)
            return error.ComputePipeFailed;

        // Descriptor pool
        const num_buffers = NUM_BINDINGS - 1;
        const pool_sizes = [2]c.VkDescriptorPoolSize{
            .{ .type = c.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1 },
            .{ .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = num_buffers },
        };
        var desc_pool: c.VkDescriptorPool = null;
        if (c.vkCreateDescriptorPool(vk.device, &c.VkDescriptorPoolCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .pNext = null, .flags = 0,
            .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = &pool_sizes,
        }, null, &desc_pool) != c.VK_SUCCESS)
            return error.DescPoolFailed;

        var desc_set: c.VkDescriptorSet = null;
        _ = c.vkAllocateDescriptorSets(vk.device, &c.VkDescriptorSetAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .pNext = null,
            .descriptorPool = desc_pool, .descriptorSetCount = 1, .pSetLayouts = &desc_layout,
        }, &desc_set);

        // Write descriptors
        const image_info = c.VkDescriptorImageInfo{
            .sampler = null,
            .imageView = rt_view,
            .imageLayout = c.VK_IMAGE_LAYOUT_GENERAL,
        };

        // Buffers matching shader bindings 1-14:
        //  1=vertices, 2=indices,
        //  3-8=BVH (aabb_min, aabb_max, left, right, count, prim_indices),
        //  9=tri_geom, 10=geom_material,
        //  11-15=material (albedo, roughness, metallic, emission, ior)
        const buffers = [num_buffers]Vulkan.Buffer{
            data.vertices,           // 1
            data.indices,            // 2
            data.bvh_aabb_min,       // 3
            data.bvh_aabb_max,       // 4
            data.bvh_left,           // 5
            data.bvh_right,          // 6
            data.bvh_count,          // 7
            data.bvh_prim_indices,   // 8
            data.tri_geom,           // 9
            data.geom_material,      // 10
            data.material_albedo,    // 11
            data.material_roughness, // 12
            data.material_metallic,  // 13
            data.material_emission,  // 14
            data.material_ior,       // 15
        };

        var writes: [NUM_BINDINGS]c.VkWriteDescriptorSet = undefined;
        var buf_infos: [num_buffers]c.VkDescriptorBufferInfo = undefined;

        // Binding 0: storage image
        writes[0] = .{
            .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = null,
            .dstSet = desc_set, .dstBinding = 0, .dstArrayElement = 0,
            .descriptorCount = 1, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &image_info, .pBufferInfo = null, .pTexelBufferView = null,
        };

        // Bindings 1..14: storage buffers
        for (0..num_buffers) |i| {
            buf_infos[i] = .{ .buffer = buffers[i].handle, .offset = 0, .range = c.VK_WHOLE_SIZE };
            writes[i + 1] = .{
                .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = null,
                .dstSet = desc_set, .dstBinding = @intCast(i + 1), .dstArrayElement = 0,
                .descriptorCount = 1, .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null, .pBufferInfo = &buf_infos[i], .pTexelBufferView = null,
            };
        }
        c.vkUpdateDescriptorSets(vk.device, NUM_BINDINGS, &writes, 0, null);

        return .{
            .vk = vk, .pipeline = pipeline, .layout = pipe_layout,
            .desc_set_layout = desc_layout, .desc_pool = desc_pool, .desc_set = desc_set,
            .width = w, .height = h, .frame = 0,
        };
    }

    pub fn render(self: *Raytracer, cam_pos: [3]f32, cam_dir: [3]f32, cam_right: [3]f32, cam_up: [3]f32, num_tris: u32) !void {
        const vk = self.vk;

        var cmd: c.VkCommandBuffer = null;
        _ = c.vkAllocateCommandBuffers(vk.device, &c.VkCommandBufferAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .pNext = null,
            .commandPool = vk.cmd_pool, .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
        }, &cmd);

        _ = c.vkBeginCommandBuffer(cmd, &c.VkCommandBufferBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .pNext = null,
            .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, .pInheritanceInfo = null,
        });

        c.vkCmdBindPipeline(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline);
        c.vkCmdBindDescriptorSets(cmd, c.VK_PIPELINE_BIND_POINT_COMPUTE, self.layout, 0, 1, &self.desc_set, 0, null);

        const pc = PC{
            .cam_pos_x = cam_pos[0], .cam_pos_y = cam_pos[1], .cam_pos_z = cam_pos[2],
            .cam_dir_x = cam_dir[0], .cam_dir_y = cam_dir[1], .cam_dir_z = cam_dir[2],
            .cam_right_x = cam_right[0], .cam_right_y = cam_right[1], .cam_right_z = cam_right[2],
            .cam_up_x = cam_up[0], .cam_up_y = cam_up[1], .cam_up_z = cam_up[2],
            .width = self.width, .height = self.height,
            .frame_num = self.frame, .num_triangles = num_tris,
        };
        c.vkCmdPushConstants(cmd, self.layout, c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @sizeOf(PC), @ptrCast(&pc));
        c.vkCmdDispatch(cmd, (self.width + 15) / 16, (self.height + 15) / 16, 1);

        _ = c.vkEndCommandBuffer(cmd);

        const queue = if (vk.compute_queue != null) vk.compute_queue else vk.graphics_queue;
        _ = c.vkQueueSubmit(queue, 1, &c.VkSubmitInfo{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = null,
            .waitSemaphoreCount = 0, .pWaitSemaphores = null, .pWaitDstStageMask = null,
            .commandBufferCount = 1, .pCommandBuffers = &cmd,
            .signalSemaphoreCount = 0, .pSignalSemaphores = null,
        }, null);
        _ = c.vkQueueWaitIdle(queue);
        c.vkFreeCommandBuffers(vk.device, vk.cmd_pool, 1, &cmd);

        self.frame += 1;
    }

    pub fn resetAccumulation(self: *Raytracer) void {
        self.frame = 0;
    }

    pub fn deinit(self: *Raytracer) void {
        c.vkDestroyPipeline(self.vk.device, self.pipeline, null);
        c.vkDestroyPipelineLayout(self.vk.device, self.layout, null);
        c.vkDestroyDescriptorPool(self.vk.device, self.desc_pool, null);
        c.vkDestroyDescriptorSetLayout(self.vk.device, self.desc_set_layout, null);
    }
};
