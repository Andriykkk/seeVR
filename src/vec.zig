const std = @import("std");

pub fn Vec(comptime T: type) type {
    return struct {
        items: []T,
        len: u32,
        alloc: std.mem.Allocator,

        const Self = @This();

        pub fn init(alloc: std.mem.Allocator) Self {
            return .{ .items = &.{}, .len = 0, .alloc = alloc };
        }

        pub fn deinit(self: *Self) void {
            if (self.items.len > 0) self.alloc.free(self.items);
        }

        pub fn push(self: *Self, val: T) !void {
            if (self.len >= self.items.len) {
                const new_cap = if (self.items.len == 0) 16 else self.items.len * 2;
                const new_buf = try self.alloc.alloc(T, new_cap);
                if (self.len > 0) @memcpy(new_buf[0..self.len], self.items[0..self.len]);
                if (self.items.len > 0) self.alloc.free(self.items);
                self.items = new_buf;
            }
            self.items[self.len] = val;
            self.len += 1;
        }

        pub fn slice(self: *const Self) []const T {
            return self.items[0..self.len];
        }
    };
}
