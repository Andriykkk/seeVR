const std = @import("std");
const c = @import("c.zig").c;

const MAX_SECTIONS = 32;
const HISTORY_SIZE = 300;

fn now_ns() u64 {
    return @intFromFloat(c.glfwGetTime() * 1_000_000_000.0);
}

pub const Profiler = struct {
    sections: [MAX_SECTIONS]Section,
    num_sections: u32,
    frame_count: u64,
    start_time: u64,

    const Section = struct {
        name: [64]u8,
        name_len: u8,
        total_ns: u64,
        count: u64,
        min_ns: u64,
        max_ns: u64,
        start_ns: u64,
        history: [HISTORY_SIZE]u64,
        history_idx: u32,
        history_count: u32,
    };

    pub fn init() Profiler {
        var p: Profiler = undefined;
        p.num_sections = 0;
        p.frame_count = 0;
        p.start_time = now_ns();
        for (&p.sections) |*s| {
            s.total_ns = 0;
            s.count = 0;
            s.min_ns = std.math.maxInt(u64);
            s.max_ns = 0;
            s.start_ns = 0;
            s.name_len = 0;
            s.history_idx = 0;
            s.history_count = 0;
            @memset(&s.history, 0);
        }
        return p;
    }

    pub fn addSection(self: *Profiler, name: []const u8) u32 {
        const idx = self.num_sections;
        self.num_sections += 1;
        const len: u8 = @intCast(@min(name.len, 64));
        @memcpy(self.sections[idx].name[0..len], name[0..len]);
        self.sections[idx].name_len = len;
        return idx;
    }

    pub fn begin(self: *Profiler, section: u32) void {
        self.sections[section].start_ns = now_ns();
    }

    pub fn end(self: *Profiler, section: u32) void {
        const elapsed = now_ns() - self.sections[section].start_ns;
        self.record(section, elapsed);
    }

    pub fn record(self: *Profiler, section: u32, duration_ns: u64) void {
        var s = &self.sections[section];
        s.total_ns += duration_ns;
        s.count += 1;
        s.min_ns = @min(s.min_ns, duration_ns);
        s.max_ns = @max(s.max_ns, duration_ns);
        s.history[s.history_idx] = duration_ns;
        s.history_idx = (s.history_idx + 1) % HISTORY_SIZE;
        s.history_count = @min(s.history_count + 1, HISTORY_SIZE);
    }

    pub fn endFrame(self: *Profiler) void {
        self.frame_count += 1;
    }

    pub fn recentAvgMs(self: *const Profiler, section: u32) f64 {
        const s = &self.sections[section];
        if (s.history_count == 0) return 0;
        var sum: u64 = 0;
        for (0..s.history_count) |i| sum += s.history[i];
        return @as(f64, @floatFromInt(sum / s.history_count)) / 1_000_000.0;
    }

    pub fn avgMs(self: *const Profiler, section: u32) f64 {
        const s = &self.sections[section];
        if (s.count == 0) return 0;
        return @as(f64, @floatFromInt(s.total_ns / s.count)) / 1_000_000.0;
    }

    /// Submit a GPU command buffer, wait for completion, and record the time under `section`.
    pub fn submitAndTime(self: *Profiler, queue: c.VkQueue, cmd: c.VkCommandBuffer, section: u32) void {
        self.begin(section);
        _ = c.vkQueueSubmit(queue, 1, &c.VkSubmitInfo{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = null,
            .waitSemaphoreCount = 0, .pWaitSemaphores = null, .pWaitDstStageMask = null,
            .commandBufferCount = 1, .pCommandBuffers = &cmd,
            .signalSemaphoreCount = 0, .pSignalSemaphores = null,
        }, null);
        _ = c.vkQueueWaitIdle(queue);
        self.end(section);
    }

    pub fn printSummary(self: *const Profiler) void {
        const elapsed_ns = now_ns() - self.start_time;
        const total_s = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;
        const fps = if (total_s > 0) @as(f64, @floatFromInt(self.frame_count)) / total_s else 0;

        std.debug.print("\n======== Profiler Summary ========\n", .{});
        std.debug.print("Frames: {}  Total: {d:.2}s  Avg FPS: {d:.1}\n\n", .{
            self.frame_count, total_s, fps,
        });
        std.debug.print("Section                  avg(ms)  min(ms)  max(ms)  recent    calls\n", .{});
        std.debug.print("------------------------ -------- -------- -------- -------- --------\n", .{});

        for (0..self.num_sections) |i| {
            const s = &self.sections[i];
            if (s.count == 0) continue;
            const name = s.name[0..s.name_len];
            const avg = self.avgMs(@intCast(i));
            const recent = self.recentAvgMs(@intCast(i));
            const min_ms = @as(f64, @floatFromInt(s.min_ns)) / 1_000_000.0;
            const max_ms = @as(f64, @floatFromInt(s.max_ns)) / 1_000_000.0;
            std.debug.print("{s:<24} {d:>8.3} {d:>8.3} {d:>8.3} {d:>8.3} {d:>8}\n", .{
                name, avg, min_ms, max_ms, recent, s.count,
            });
        }
        std.debug.print("\n", .{});
    }
};
