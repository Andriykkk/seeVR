const std = @import("std");
const c = @import("c.zig").c;
const fs = @import("fs.zig");

pub const Vulkan = struct {
    instance: c.VkInstance,
    physical_device: c.VkPhysicalDevice,
    device: c.VkDevice,
    graphics_queue: c.VkQueue,
    compute_queue: c.VkQueue,
    graphics_family: u32,
    compute_family: u32,
    surface: c.VkSurfaceKHR, // null if headless
    cmd_pool: c.VkCommandPool,

    /// Init Vulkan. Pass null window for headless (compute-only) mode.
    pub fn init(window: ?*c.GLFWwindow) !Vulkan {
        const has_window = window != null;

        // --- Instance ---
        var glfw_ext_count: u32 = 0;
        var glfw_exts: [*c]const [*c]const u8 = undefined;
        if (has_window) {
            glfw_exts = c.glfwGetRequiredInstanceExtensions(&glfw_ext_count);
        }

        const app_info = c.VkApplicationInfo{
            .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = null,
            .pApplicationName = "Robotics Engine",
            .applicationVersion = c.VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "Custom",
            .engineVersion = c.VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = c.VK_API_VERSION_1_2,
        };

        const create_info = c.VkInstanceCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = if (has_window) glfw_ext_count else 0,
            .ppEnabledExtensionNames = if (has_window) glfw_exts else null,
        };

        var instance: c.VkInstance = null;
        if (c.vkCreateInstance(&create_info, null, &instance) != c.VK_SUCCESS)
            return error.InstanceCreateFailed;

        // --- Surface (only if windowed) ---
        var surface: c.VkSurfaceKHR = null;
        if (has_window) {
            if (c.glfwCreateWindowSurface(instance, window.?, null, &surface) != c.VK_SUCCESS)
                return error.SurfaceCreateFailed;
        }

        // --- Physical device ---
        var dev_count: u32 = 0;
        _ = c.vkEnumeratePhysicalDevices(instance, &dev_count, null);
        if (dev_count == 0) return error.NoGpuFound;

        var devices: [16]c.VkPhysicalDevice = undefined;
        var count: u32 = @min(dev_count, 16);
        _ = c.vkEnumeratePhysicalDevices(instance, &count, &devices);

        var physical_device: c.VkPhysicalDevice = null;
        var graphics_family: u32 = 0;
        var compute_family: u32 = 0;

        var best_score: u32 = 0;
        for (devices[0..count]) |dev| {
            var dev_props: c.VkPhysicalDeviceProperties = undefined;
            c.vkGetPhysicalDeviceProperties(dev, &dev_props);
            const score: u32 = if (dev_props.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) 100 else 10;

            var queue_count: u32 = 0;
            c.vkGetPhysicalDeviceQueueFamilyProperties(dev, &queue_count, null);
            var queue_props: [32]c.VkQueueFamilyProperties = undefined;
            var qc: u32 = @min(queue_count, 32);
            c.vkGetPhysicalDeviceQueueFamilyProperties(dev, &qc, &queue_props);

            var found_graphics = false;
            var found_compute = false;
            var gf: u32 = 0;
            var cf: u32 = 0;

            for (0..qc) |i| {
                const flags = queue_props[i].queueFlags;
                if (flags & c.VK_QUEUE_GRAPHICS_BIT != 0 and !found_graphics) {
                    gf = @intCast(i);
                    found_graphics = true;
                }
                if (flags & c.VK_QUEUE_COMPUTE_BIT != 0 and !found_compute) {
                    cf = @intCast(i);
                    found_compute = true;
                }
            }

            // Headless only needs compute; windowed needs both
            const suitable = if (has_window) (found_graphics and found_compute) else found_compute;
            if (suitable and score > best_score) {
                physical_device = dev;
                graphics_family = gf;
                compute_family = cf;
                best_score = score;
            }
        }
        if (physical_device == null) return error.NoSuitableGpu;

        var props: c.VkPhysicalDeviceProperties = undefined;
        c.vkGetPhysicalDeviceProperties(physical_device, &props);
        std.debug.print("Vulkan device: {s}\n", .{@as([*:0]const u8, @ptrCast(&props.deviceName))});

        // --- Logical device ---
        const priority: f32 = 1.0;
        var queue_create_infos: [2]c.VkDeviceQueueCreateInfo = undefined;
        var queue_create_count: u32 = 1;

        queue_create_infos[0] = .{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueFamilyIndex = if (has_window) graphics_family else compute_family,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        };

        if (has_window and compute_family != graphics_family) {
            queue_create_infos[1] = .{
                .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .pNext = null,
                .flags = 0,
                .queueFamilyIndex = compute_family,
                .queueCount = 1,
                .pQueuePriorities = &priority,
            };
            queue_create_count = 2;
        }

        // Swapchain extension only needed for windowed
        const swapchain_ext = [_][*c]const u8{c.VK_KHR_SWAPCHAIN_EXTENSION_NAME};

        const device_create_info = c.VkDeviceCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueCreateInfoCount = queue_create_count,
            .pQueueCreateInfos = &queue_create_infos,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = if (has_window) 1 else 0,
            .ppEnabledExtensionNames = if (has_window) &swapchain_ext else null,
            .pEnabledFeatures = null,
        };

        var device: c.VkDevice = null;
        if (c.vkCreateDevice(physical_device, &device_create_info, null, &device) != c.VK_SUCCESS)
            return error.DeviceCreateFailed;

        var graphics_queue: c.VkQueue = null;
        var compute_queue: c.VkQueue = null;
        if (has_window) c.vkGetDeviceQueue(device, graphics_family, 0, &graphics_queue);
        c.vkGetDeviceQueue(device, compute_family, 0, &compute_queue);

        // --- Command pool ---
        const pool_info = c.VkCommandPoolCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = null,
            .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = if (has_window) graphics_family else compute_family,
        };
        var cmd_pool: c.VkCommandPool = null;
        if (c.vkCreateCommandPool(device, &pool_info, null, &cmd_pool) != c.VK_SUCCESS)
            return error.CommandPoolCreateFailed;

        const mode_str = if (has_window) "windowed" else "headless";
        std.debug.print("Vulkan init OK ({s}, graphics={}, compute={})\n", .{ mode_str, graphics_family, compute_family });

        return Vulkan{
            .instance = instance,
            .physical_device = physical_device,
            .device = device,
            .graphics_queue = graphics_queue,
            .compute_queue = compute_queue,
            .graphics_family = graphics_family,
            .compute_family = compute_family,
            .surface = surface,
            .cmd_pool = cmd_pool,
        };
    }

    // --- Shader compilation (disk-cached) ---

    pub const ShaderStage = enum {
        vertex,
        fragment,
        compute,
    };

    /// Load a GLSL shader file, compile to SPIR-V if needed, return VkShaderModule.
    /// Caches .spv next to source. Delete .spv to force recompile.
    pub fn getShader(self: *Vulkan, path: []const u8, stage: ShaderStage, allocator: std.mem.Allocator) !c.VkShaderModule {
        var spv_path_buf: [512]u8 = undefined;
        const spv_path_slice = std.fmt.bufPrint(&spv_path_buf, "{s}.spv\x00", .{path}) catch return error.PathTooLong;
        const spv_path: [*:0]const u8 = @ptrCast(spv_path_slice.ptr);

        // Null-terminate path for C
        var path_buf: [512]u8 = undefined;
        @memcpy(path_buf[0..path.len], path);
        path_buf[path.len] = 0;
        const path_z: [*:0]const u8 = @ptrCast(&path_buf);

        // Try cached .spv (only if newer than source)
        const spv_data: []u8 = if (fs.fileExists(spv_path) and !fs.isNewer(path_z, spv_path))
            try fs.readFile(spv_path, allocator)
        else blk: {
            // Compile from GLSL source
            const source = try fs.readFile(path_z, allocator);
            defer allocator.free(source);

            const shaderc_stage: c_uint = switch (stage) {
                .vertex => c.shaderc_vertex_shader,
                .fragment => c.shaderc_fragment_shader,
                .compute => c.shaderc_compute_shader,
            };

            const compiler = c.shaderc_compiler_initialize();
            defer c.shaderc_compiler_release(compiler);

            const result = c.shaderc_compile_into_spv(
                compiler,
                source.ptr,
                source.len,
                shaderc_stage,
                path_z,
                "main",
                null,
            );
            defer c.shaderc_result_release(result);

            if (c.shaderc_result_get_compilation_status(result) != c.shaderc_compilation_status_success) {
                const err_msg = c.shaderc_result_get_error_message(result);
                std.debug.print("Shader compile error ({s}):\n{s}\n", .{ path, err_msg });
                return error.ShaderCompileFailed;
            }

            const bytes = c.shaderc_result_get_bytes(result);
            const len = c.shaderc_result_get_length(result);
            const data = try allocator.alloc(u8, len);
            @memcpy(data, @as([*]const u8, @ptrCast(bytes))[0..len]);

            fs.writeFile(spv_path, data);
            std.debug.print("Compiled shader: {s}\n", .{path});
            break :blk data;
        };
        defer allocator.free(spv_data);

        // Create VkShaderModule from SPIR-V
        const module_info = c.VkShaderModuleCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .codeSize = spv_data.len,
            .pCode = @ptrCast(@alignCast(spv_data.ptr)),
        };

        var shader_module: c.VkShaderModule = null;
        if (c.vkCreateShaderModule(self.device, &module_info, null, &shader_module) != c.VK_SUCCESS)
            return error.ShaderModuleCreateFailed;

        return shader_module;
    }

    // --- Buffer usage presets ---
    pub const USAGE_VERTEX = c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    pub const USAGE_INDEX = c.VK_BUFFER_USAGE_INDEX_BUFFER_BIT | c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    pub const USAGE_STORAGE = c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    pub const Buffer = struct {
        handle: c.VkBuffer,
        memory: c.VkDeviceMemory,
        size: usize,
    };

    pub fn findMemoryType(self: *Vulkan, type_filter: u32, properties: c.VkMemoryPropertyFlags) !u32 {
        var mem_props: c.VkPhysicalDeviceMemoryProperties = undefined;
        c.vkGetPhysicalDeviceMemoryProperties(self.physical_device, &mem_props);
        for (0..mem_props.memoryTypeCount) |i| {
            if (type_filter & (@as(u32, 1) << @intCast(i)) != 0 and
                mem_props.memoryTypes[i].propertyFlags & properties == properties)
            {
                return @intCast(i);
            }
        }
        return error.NoSuitableMemory;
    }

    fn allocBuffer(self: *Vulkan, size: usize, usage: c.VkBufferUsageFlags, mem_flags: c.VkMemoryPropertyFlags) !Buffer {
        const buf_info = c.VkBufferCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .size = @intCast(size),
            .usage = usage,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
        };

        var handle: c.VkBuffer = null;
        if (c.vkCreateBuffer(self.device, &buf_info, null, &handle) != c.VK_SUCCESS)
            return error.BufferCreateFailed;

        var mem_reqs: c.VkMemoryRequirements = undefined;
        c.vkGetBufferMemoryRequirements(self.device, handle, &mem_reqs);

        const alloc_info = c.VkMemoryAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = null,
            .allocationSize = mem_reqs.size,
            .memoryTypeIndex = try self.findMemoryType(mem_reqs.memoryTypeBits, mem_flags),
        };

        var memory: c.VkDeviceMemory = null;
        if (c.vkAllocateMemory(self.device, &alloc_info, null, &memory) != c.VK_SUCCESS)
            return error.MemoryAllocFailed;

        _ = c.vkBindBufferMemory(self.device, handle, memory, 0);
        return .{ .handle = handle, .memory = memory, .size = size };
    }

    pub fn createBuffer(self: *Vulkan, size: usize, usage: c.VkBufferUsageFlags) !Buffer {
        return self.allocBuffer(size, usage | c.VK_BUFFER_USAGE_TRANSFER_DST_BIT, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    pub fn upload(self: *Vulkan, buf: Buffer, data: [*]const u8, size: usize) !void {
        if (size == 0) return;
        const staging = try self.allocBuffer(
            size,
            c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        defer self.destroyBuffer(staging);

        var mapped: ?*anyopaque = null;
        _ = c.vkMapMemory(self.device, staging.memory, 0, @intCast(size), 0, &mapped);
        @memcpy(@as([*]u8, @ptrCast(mapped.?))[0..size], data[0..size]);
        c.vkUnmapMemory(self.device, staging.memory);

        try self.copyBuffer(staging.handle, buf.handle, size);
    }

    pub fn uploadSlice(self: *Vulkan, buf: Buffer, comptime T: type, slice: []const T) !void {
        try self.upload(buf, @as([*]const u8, @ptrCast(slice.ptr)), slice.len * @sizeOf(T));
    }

    fn copyBuffer(self: *Vulkan, src: c.VkBuffer, dst: c.VkBuffer, size: usize) !void {
        const queue = if (self.graphics_queue != null) self.graphics_queue else self.compute_queue;

        const cb_alloc = c.VkCommandBufferAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = null,
            .commandPool = self.cmd_pool,
            .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        var cmd: c.VkCommandBuffer = null;
        _ = c.vkAllocateCommandBuffers(self.device, &cb_alloc, &cmd);

        _ = c.vkBeginCommandBuffer(cmd, &c.VkCommandBufferBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null,
        });

        c.vkCmdCopyBuffer(cmd, src, dst, 1, &c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = @intCast(size) });

        _ = c.vkEndCommandBuffer(cmd);

        _ = c.vkQueueSubmit(queue, 1, &c.VkSubmitInfo{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .pWaitDstStageMask = null,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null,
        }, null);
        _ = c.vkQueueWaitIdle(queue);

        c.vkFreeCommandBuffers(self.device, self.cmd_pool, 1, &cmd);
    }

    /// Create a descriptor pool (needed for ImGui)
    pub fn createDescriptorPool(self: *Vulkan) !c.VkDescriptorPool {
        const pool_sizes = [_]c.VkDescriptorPoolSize{
            .{ .type = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 100 },
            .{ .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 100 },
        };
        var pool: c.VkDescriptorPool = null;
        if (c.vkCreateDescriptorPool(self.device, &c.VkDescriptorPoolCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = c.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 100,
            .poolSizeCount = pool_sizes.len,
            .pPoolSizes = &pool_sizes,
        }, null, &pool) != c.VK_SUCCESS)
            return error.DescriptorPoolCreateFailed;
        return pool;
    }

    // --- Compute pipeline ---

    pub const ComputePipeline = struct {
        pipeline: c.VkPipeline,
        layout: c.VkPipelineLayout,
        desc_set_layout: c.VkDescriptorSetLayout,
        desc_pool: c.VkDescriptorPool,
        desc_set: c.VkDescriptorSet,
    };

    /// Create a compute pipeline with N storage buffer bindings + push constants
    pub fn createComputePipeline(
        self: *Vulkan,
        shader: c.VkShaderModule,
        num_bindings: u32,
        buffers: []const Buffer,
        push_size: u32,
    ) !ComputePipeline {
        // Descriptor set layout
        var bindings: [32]c.VkDescriptorSetLayoutBinding = undefined;
        for (0..num_bindings) |i| {
            bindings[i] = .{
                .binding = @intCast(i),
                .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = null,
            };
        }

        var desc_layout: c.VkDescriptorSetLayout = null;
        if (c.vkCreateDescriptorSetLayout(self.device, &c.VkDescriptorSetLayoutCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .bindingCount = num_bindings,
            .pBindings = &bindings,
        }, null, &desc_layout) != c.VK_SUCCESS)
            return error.DescLayoutFailed;

        // Pipeline layout
        const push_range = c.VkPushConstantRange{
            .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = push_size,
        };
        var pipe_layout: c.VkPipelineLayout = null;
        if (c.vkCreatePipelineLayout(self.device, &c.VkPipelineLayoutCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
            .pushConstantRangeCount = if (push_size > 0) 1 else 0,
            .pPushConstantRanges = if (push_size > 0) &push_range else null,
        }, null, &pipe_layout) != c.VK_SUCCESS)
            return error.PipeLayoutFailed;

        // Pipeline
        var pipeline: c.VkPipeline = null;
        if (c.vkCreateComputePipelines(self.device, null, 1, &c.VkComputePipelineCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .stage = .{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = null,
                .flags = 0,
                .stage = c.VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader,
                .pName = "main",
                .pSpecializationInfo = null,
            },
            .layout = pipe_layout,
            .basePipelineHandle = null,
            .basePipelineIndex = -1,
        }, null, &pipeline) != c.VK_SUCCESS)
            return error.ComputePipeFailed;

        // Descriptor pool + set
        const pool_size = c.VkDescriptorPoolSize{
            .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = num_bindings,
        };
        var desc_pool: c.VkDescriptorPool = null;
        if (c.vkCreateDescriptorPool(self.device, &c.VkDescriptorPoolCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        }, null, &desc_pool) != c.VK_SUCCESS)
            return error.DescPoolFailed;

        var desc_set: c.VkDescriptorSet = null;
        _ = c.vkAllocateDescriptorSets(self.device, &c.VkDescriptorSetAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null,
            .descriptorPool = desc_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &desc_layout,
        }, &desc_set);

        // Write buffer descriptors
        var writes: [32]c.VkWriteDescriptorSet = undefined;
        var buf_infos: [32]c.VkDescriptorBufferInfo = undefined;
        for (0..num_bindings) |i| {
            buf_infos[i] = .{
                .buffer = buffers[i].handle,
                .offset = 0,
                .range = c.VK_WHOLE_SIZE,
            };
            writes[i] = .{
                .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null,
                .dstSet = desc_set,
                .dstBinding = @intCast(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null,
                .pBufferInfo = &buf_infos[i],
                .pTexelBufferView = null,
            };
        }
        c.vkUpdateDescriptorSets(self.device, num_bindings, &writes, 0, null);

        return .{
            .pipeline = pipeline,
            .layout = pipe_layout,
            .desc_set_layout = desc_layout,
            .desc_pool = desc_pool,
            .desc_set = desc_set,
        };
    }

    pub fn destroyComputePipeline(self: *Vulkan, cp: ComputePipeline) void {
        c.vkDestroyPipeline(self.device, cp.pipeline, null);
        c.vkDestroyPipelineLayout(self.device, cp.layout, null);
        c.vkDestroyDescriptorPool(self.device, cp.desc_pool, null);
        c.vkDestroyDescriptorSetLayout(self.device, cp.desc_set_layout, null);
    }

    pub fn destroyBuffer(self: *Vulkan, buf: Buffer) void {
        c.vkDestroyBuffer(self.device, buf.handle, null);
        c.vkFreeMemory(self.device, buf.memory, null);
    }

    pub fn deinit(self: *Vulkan) void {
        _ = c.vkDeviceWaitIdle(self.device);
        c.vkDestroyCommandPool(self.device, self.cmd_pool, null);
        c.vkDestroyDevice(self.device, null);
        if (self.surface != null) c.vkDestroySurfaceKHR(self.instance, self.surface, null);
        c.vkDestroyInstance(self.instance, null);
    }
};
