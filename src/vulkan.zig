const std = @import("std");
const c = @import("c.zig").c;

pub const Vulkan = struct {
    instance: c.VkInstance,
    physical_device: c.VkPhysicalDevice,
    device: c.VkDevice,
    graphics_queue: c.VkQueue,
    compute_queue: c.VkQueue,
    graphics_family: u32,
    compute_family: u32,
    surface: c.VkSurfaceKHR,
    cmd_pool: c.VkCommandPool,

    pub fn init(window: *c.GLFWwindow) !Vulkan {
        // --- Instance ---
        var glfw_ext_count: u32 = 0;
        const glfw_exts = c.glfwGetRequiredInstanceExtensions(&glfw_ext_count);

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
            .enabledExtensionCount = glfw_ext_count,
            .ppEnabledExtensionNames = glfw_exts,
        };

        var instance: c.VkInstance = null;
        if (c.vkCreateInstance(&create_info, null, &instance) != c.VK_SUCCESS)
            return error.InstanceCreateFailed;

        // --- Surface ---
        var surface: c.VkSurfaceKHR = null;
        if (c.glfwCreateWindowSurface(instance, window, null, &surface) != c.VK_SUCCESS)
            return error.SurfaceCreateFailed;

        // --- Physical device ---
        var dev_count: u32 = 0;
        _ = c.vkEnumeratePhysicalDevices(instance, &dev_count, null);
        if (dev_count == 0) return error.NoGpuFound;

        var devices: [16]c.VkPhysicalDevice = undefined;
        var count: u32 = @min(dev_count, 16);
        _ = c.vkEnumeratePhysicalDevices(instance, &count, &devices);

        // Pick first device with graphics + compute queues
        var physical_device: c.VkPhysicalDevice = null;
        var graphics_family: u32 = 0;
        var compute_family: u32 = 0;

        // Prefer discrete GPU over integrated
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

            if (found_graphics and found_compute and score > best_score) {
                physical_device = dev;
                graphics_family = gf;
                compute_family = cf;
                best_score = score;
            }
        }
        if (physical_device == null) return error.NoSuitableGpu;

        // Print device name
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
            .queueFamilyIndex = graphics_family,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        };

        if (compute_family != graphics_family) {
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

        const device_extensions = [_][*c]const u8{c.VK_KHR_SWAPCHAIN_EXTENSION_NAME};

        const device_create_info = c.VkDeviceCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueCreateInfoCount = queue_create_count,
            .pQueueCreateInfos = &queue_create_infos,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = 1,
            .ppEnabledExtensionNames = &device_extensions,
            .pEnabledFeatures = null,
        };

        var device: c.VkDevice = null;
        if (c.vkCreateDevice(physical_device, &device_create_info, null, &device) != c.VK_SUCCESS)
            return error.DeviceCreateFailed;

        var graphics_queue: c.VkQueue = null;
        var compute_queue: c.VkQueue = null;
        c.vkGetDeviceQueue(device, graphics_family, 0, &graphics_queue);
        c.vkGetDeviceQueue(device, compute_family, 0, &compute_queue);

        // --- Command pool ---
        const pool_info = c.VkCommandPoolCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = null,
            .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = graphics_family,
        };
        var cmd_pool: c.VkCommandPool = null;
        if (c.vkCreateCommandPool(device, &pool_info, null, &cmd_pool) != c.VK_SUCCESS)
            return error.CommandPoolCreateFailed;

        std.debug.print("Vulkan init OK (graphics={}, compute={})\n", .{ graphics_family, compute_family });

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

    // --- Buffer usage presets ---
    pub const USAGE_VERTEX = c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    pub const USAGE_INDEX = c.VK_BUFFER_USAGE_INDEX_BUFFER_BIT | c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    pub const USAGE_STORAGE = c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    pub const Buffer = struct {
        handle: c.VkBuffer,
        memory: c.VkDeviceMemory,
        size: usize,
    };

    fn findMemoryType(self: *Vulkan, type_filter: u32, properties: c.VkMemoryPropertyFlags) !u32 {
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

    /// Create a GPU-local buffer (for compute + vertex/index/storage use)
    pub fn createBuffer(self: *Vulkan, size: usize, usage: c.VkBufferUsageFlags) !Buffer {
        return self.allocBuffer(size, usage | c.VK_BUFFER_USAGE_TRANSFER_DST_BIT, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    /// Upload CPU data to an existing GPU buffer via staging
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

    /// Upload a typed slice to a GPU buffer
    pub fn uploadSlice(self: *Vulkan, buf: Buffer, comptime T: type, slice: []const T) !void {
        try self.upload(buf, @as([*]const u8, @ptrCast(slice.ptr)), slice.len * @sizeOf(T));
    }

    fn copyBuffer(self: *Vulkan, src: c.VkBuffer, dst: c.VkBuffer, size: usize) !void {
        const alloc_info = c.VkCommandBufferAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = null,
            .commandPool = self.cmd_pool,
            .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        var cmd: c.VkCommandBuffer = null;
        _ = c.vkAllocateCommandBuffers(self.device, &alloc_info, &cmd);

        const begin_info = c.VkCommandBufferBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null,
        };
        _ = c.vkBeginCommandBuffer(cmd, &begin_info);

        const copy_region = c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = @intCast(size) };
        c.vkCmdCopyBuffer(cmd, src, dst, 1, &copy_region);

        _ = c.vkEndCommandBuffer(cmd);

        const submit_info = c.VkSubmitInfo{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .pWaitDstStageMask = null,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null,
        };
        _ = c.vkQueueSubmit(self.graphics_queue, 1, &submit_info, null);
        _ = c.vkQueueWaitIdle(self.graphics_queue);

        c.vkFreeCommandBuffers(self.device, self.cmd_pool, 1, &cmd);
    }

    pub fn destroyBuffer(self: *Vulkan, buf: Buffer) void {
        c.vkDestroyBuffer(self.device, buf.handle, null);
        c.vkFreeMemory(self.device, buf.memory, null);
    }

    pub fn deinit(self: *Vulkan) void {
        _ = c.vkDeviceWaitIdle(self.device);
        c.vkDestroyCommandPool(self.device, self.cmd_pool, null);
        c.vkDestroyDevice(self.device, null);
        c.vkDestroySurfaceKHR(self.instance, self.surface, null);
        c.vkDestroyInstance(self.instance, null);
    }
};
