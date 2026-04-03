const std = @import("std");

const libc = @cImport({
    @cInclude("stdio.h");
    @cInclude("sys/stat.h");
});

pub fn readFile(path: [*:0]const u8, allocator: std.mem.Allocator) ![]u8 {
    const file = libc.fopen(path, "rb") orelse return error.FileNotFound;
    _ = libc.fseek(file, 0, libc.SEEK_END);
    const size: usize = @intCast(libc.ftell(file));
    _ = libc.fseek(file, 0, libc.SEEK_SET);
    const buf = try allocator.alloc(u8, size);
    _ = libc.fread(buf.ptr, 1, size, file);
    _ = libc.fclose(file);
    return buf;
}

pub fn writeFile(path: [*:0]const u8, data: []const u8) void {
    const file = libc.fopen(path, "wb") orelse return;
    _ = libc.fwrite(data.ptr, 1, data.len, file);
    _ = libc.fclose(file);
}

pub fn fileExists(path: [*:0]const u8) bool {
    const file = libc.fopen(path, "rb") orelse return false;
    _ = libc.fclose(file);
    return true;
}

/// Get file modification time (seconds since epoch). Returns 0 if file doesn't exist.
pub fn mtime(path: [*:0]const u8) i64 {
    var st: libc.struct_stat = undefined;
    if (libc.stat(path, &st) != 0) return 0;
    return @intCast(st.st_mtim.tv_sec);
}

/// Check if file a is newer than file b
pub fn isNewer(a: [*:0]const u8, b: [*:0]const u8) bool {
    return mtime(a) > mtime(b);
}
