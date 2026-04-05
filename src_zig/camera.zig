const std = @import("std");
const c = @import("c.zig").c;
const math = std.math;

pub const Camera = struct {
    pos: [3]f32,
    yaw: f32, // degrees
    pitch: f32, // degrees
    speed: f32,
    sensitivity: f32,
    last_mouse: ?[2]f64,

    pub fn init(x: f32, y: f32, z: f32, yaw: f32, pitch: f32) Camera {
        return .{
            .pos = .{ x, y, z },
            .yaw = yaw,
            .pitch = pitch,
            .speed = 10.0,
            .sensitivity = 0.15,
            .last_mouse = null,
        };
    }

    pub fn update(self: *Camera, window: *c.GLFWwindow, dt: f32) void {
        // Movement
        const ry = self.yaw * math.pi / 180.0;
        const cos_ry = @cos(ry);
        const sin_ry = @sin(ry);
        const move = self.speed * dt;

        if (c.glfwGetKey(window, c.GLFW_KEY_W) == c.GLFW_PRESS) {
            self.pos[0] += cos_ry * move;
            self.pos[2] += sin_ry * move;
        }
        if (c.glfwGetKey(window, c.GLFW_KEY_S) == c.GLFW_PRESS) {
            self.pos[0] -= cos_ry * move;
            self.pos[2] -= sin_ry * move;
        }
        if (c.glfwGetKey(window, c.GLFW_KEY_A) == c.GLFW_PRESS) {
            const ry_l = (self.yaw - 90.0) * math.pi / 180.0;
            self.pos[0] += @cos(ry_l) * move;
            self.pos[2] += @sin(ry_l) * move;
        }
        if (c.glfwGetKey(window, c.GLFW_KEY_D) == c.GLFW_PRESS) {
            const ry_r = (self.yaw + 90.0) * math.pi / 180.0;
            self.pos[0] += @cos(ry_r) * move;
            self.pos[2] += @sin(ry_r) * move;
        }
        if (c.glfwGetKey(window, c.GLFW_KEY_SPACE) == c.GLFW_PRESS) {
            self.pos[1] += move;
        }
        if (c.glfwGetKey(window, c.GLFW_KEY_LEFT_SHIFT) == c.GLFW_PRESS) {
            self.pos[1] -= move;
        }

        // Mouse look (right click held)
        var mx: f64 = 0;
        var my: f64 = 0;
        c.glfwGetCursorPos(window, &mx, &my);
        if (c.glfwGetMouseButton(window, c.GLFW_MOUSE_BUTTON_RIGHT) == c.GLFW_PRESS) {
            if (self.last_mouse) |last| {
                const dx: f32 = @floatCast(mx - last[0]);
                const dy: f32 = @floatCast(my - last[1]);
                self.yaw += dx * self.sensitivity;
                self.pitch = @max(-89.0, @min(89.0, self.pitch + dy * self.sensitivity));
            }
        }
        self.last_mouse = .{ mx, my };
    }

    /// Build a view-projection matrix (column-major, Vulkan clip space)
    pub fn mvp(self: *const Camera, aspect: f32) [16]f32 {
        // Direction from yaw/pitch
        const ry = self.yaw * math.pi / 180.0;
        const rp = self.pitch * math.pi / 180.0;
        const dx = @cos(rp) * @cos(ry);
        const dy = @sin(rp);
        const dz = @cos(rp) * @sin(ry);

        // Right = normalize(direction x up)
        const rx = -dz;
        const rz = dx;
        const rl = @sqrt(rx * rx + rz * rz);
        const right = [3]f32{ rx / rl, 0, rz / rl };

        // Up = right x direction
        const up = [3]f32{
            0 * dz - right[2] * dy,
            right[2] * dx - right[0] * dz,
            right[0] * dy - 0 * dx,
        };

        // View matrix (column-major): transpose of [right, up, -forward] with translation
        const tx = -(right[0] * self.pos[0] + right[1] * self.pos[1] + right[2] * self.pos[2]);
        const ty = -(up[0] * self.pos[0] + up[1] * self.pos[1] + up[2] * self.pos[2]);
        const tz = dx * self.pos[0] + dy * self.pos[1] + dz * self.pos[2];

        const view = [16]f32{
            right[0], up[0], -dx, 0,
            right[1], up[1], -dy, 0,
            right[2], up[2], -dz, 0,
            tx,       ty,    tz,  1,
        };

        // Perspective projection (Vulkan: Y flipped, depth 0..1)
        const fov = 60.0 * math.pi / 180.0;
        const near = 0.1;
        const far = 100.0;
        const f = 1.0 / @tan(fov / 2.0);

        const proj = [16]f32{
            f / aspect, 0,  0,                              0,
            0,          -f, 0,                              0,
            0,          0,  far / (near - far),             -1,
            0,          0,  (near * far) / (near - far),    0,
        };

        // result = proj * view
        return matMul(proj, view);
    }
};

fn matMul(a: [16]f32, b: [16]f32) [16]f32 {
    var result: [16]f32 = undefined;
    for (0..4) |col| {
        for (0..4) |row| {
            var sum: f32 = 0;
            for (0..4) |k| {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            result[col * 4 + row] = sum;
        }
    }
    return result;
}
