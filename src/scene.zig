const std = @import("std");
const data = @import("data.zig");
const c = @import("c.zig");

pub const Scene = struct {
    window: *c.GLFWwindow,
    width: i32,
    height: i32,
    vbo: c.GLuint,
    ebo: c.GLuint,
    cbo: c.GLuint,
    vao: c.GLuint,

    pub fn init(width: i32, height: i32, title: [*c]const u8) !Scene {
        if (c.glfwInit() == 0) return error.GlfwInitFailed;

        c.glfwWindowHint(c.GLFW_CONTEXT_VERSION_MAJOR, 3);
        c.glfwWindowHint(c.GLFW_CONTEXT_VERSION_MINOR, 3);
        c.glfwWindowHint(c.GLFW_OPENGL_PROFILE, c.GLFW_OPENGL_CORE_PROFILE);

        const window = c.glfwCreateWindow(width, height, title, null, null) orelse {
            c.glfwTerminate();
            return error.WindowCreateFailed;
        };

        c.glfwMakeContextCurrent(window);
        c.glViewport(0, 0, width, height);
        c.glEnable(c.GL_DEPTH_TEST);

        var vao: c.GLuint = 0;
        var vbo: c.GLuint = 0;
        var ebo: c.GLuint = 0;
        var cbo: c.GLuint = 0;
        c.glGenVertexArrays(1, &vao);
        c.glGenBuffers(1, &vbo);
        c.glGenBuffers(1, &ebo);
        c.glGenBuffers(1, &cbo);

        c.glBindVertexArray(vao);

        // Positions (location 0)
        c.glBindBuffer(c.GL_ARRAY_BUFFER, vbo);
        c.glBufferData(c.GL_ARRAY_BUFFER, @intCast(data.MAX_VERTICES * 3 * @sizeOf(f32)), null, c.GL_DYNAMIC_DRAW);
        c.glVertexAttribPointer(0, 3, c.GL_FLOAT, c.GL_FALSE, 3 * @sizeOf(f32), null);
        c.glEnableVertexAttribArray(0);

        // Colors (location 1)
        c.glBindBuffer(c.GL_ARRAY_BUFFER, cbo);
        c.glBufferData(c.GL_ARRAY_BUFFER, @intCast(data.MAX_VERTICES * 3 * @sizeOf(f32)), null, c.GL_DYNAMIC_DRAW);
        c.glVertexAttribPointer(1, 3, c.GL_FLOAT, c.GL_FALSE, 3 * @sizeOf(f32), null);
        c.glEnableVertexAttribArray(1);

        // Indices
        c.glBindBuffer(c.GL_ELEMENT_ARRAY_BUFFER, ebo);
        c.glBufferData(c.GL_ELEMENT_ARRAY_BUFFER, @intCast(data.MAX_TRIANGLES * 3 * @sizeOf(u32)), null, c.GL_DYNAMIC_DRAW);

        c.glBindVertexArray(0);

        return Scene{
            .window = window,
            .width = width,
            .height = height,
            .vbo = vbo,
            .ebo = ebo,
            .cbo = cbo,
            .vao = vao,
        };
    }

    pub fn deinit(self: *Scene) void {
        c.glDeleteVertexArrays(1, &self.vao);
        c.glDeleteBuffers(1, &self.vbo);
        c.glDeleteBuffers(1, &self.ebo);
        c.glDeleteBuffers(1, &self.cbo);
        c.glfwDestroyWindow(self.window);
        c.glfwTerminate();
    }

    pub fn shouldClose(self: *const Scene) bool {
        return c.glfwWindowShouldClose(self.window) != 0;
    }

    pub fn beginFrame(_: *const Scene) void {
        c.glClearColor(0.1, 0.1, 0.12, 1.0);
        c.glClear(c.GL_COLOR_BUFFER_BIT | c.GL_DEPTH_BUFFER_BIT);
    }

    pub fn draw(self: *const Scene, d: *const data.Data) void {
        c.glBindVertexArray(self.vao);
        c.glDrawElements(c.GL_TRIANGLES, @intCast(d.num_triangles * 3), c.GL_UNSIGNED_INT, null);
        c.glBindVertexArray(0);
    }

    pub fn endFrame(self: *const Scene) void {
        c.glfwSwapBuffers(self.window);
        c.glfwPollEvents();
    }
};
