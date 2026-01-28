import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np
import pyopencl as cl
import ctypes
import time


class Window:
    def __init__(self, width=800, height=600, title="OpenGL + OpenCL"):
        # Initialize GLFW
        if not glfw.init():
            raise Exception("GLFW failed to initialize")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Window creation failed")

        glfw.make_context_current(self.window)

        # OpenCL setup
        self._init_opencl()

        # OpenGL setup
        self._init_opengl()

        # Data setup
        self._init_data()

        self.start_time = time.time()

    def _init_opencl(self):
        platform = cl.get_platforms()[0]
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if not devices:
            devices = platform.get_devices(device_type=cl.device_type.CPU)

        self.cl_ctx = cl.Context(devices=devices)
        self.cl_queue = cl.CommandQueue(self.cl_ctx)
        print(f"OpenCL device: {devices[0].name}")

        cl_kernel_src = """
        __kernel void rotate_vertices(
            __global float* vertices,
            __global const float* original,
            const float angle
        ) {
            int i = get_global_id(0);
            int base = i * 6;

            float x = original[base + 0];
            float y = original[base + 1];

            float cos_a = cos(angle);
            float sin_a = sin(angle);

            vertices[base + 0] = x * cos_a - y * sin_a;
            vertices[base + 1] = x * sin_a + y * cos_a;
            vertices[base + 2] = original[base + 2];

            vertices[base + 3] = original[base + 3];
            vertices[base + 4] = original[base + 4];
            vertices[base + 5] = original[base + 5];
        }
        """

        cl_program = cl.Program(self.cl_ctx, cl_kernel_src).build()
        self.rotate_kernel = cl_program.rotate_vertices

    def _init_opengl(self):
        vertex_shader = """
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aColor;
        out vec3 color;

        void main() {
            gl_Position = vec4(aPos, 1.0);
            color = aColor;
        }
        """

        fragment_shader = """
        #version 330 core
        in vec3 color;
        out vec4 FragColor;

        void main() {
            FragColor = vec4(color, 1.0);
        }
        """

        self.program = shaders.compileProgram(
            shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
            shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )

    def _init_data(self):
        self.original_vertices = np.array([
            -0.5, -0.5, 0.0,  1.0, 0.0, 0.0,
             0.5, -0.5, 0.0,  0.0, 1.0, 0.0,
             0.0,  0.5, 0.0,  0.0, 0.0, 1.0,
        ], dtype=np.float32)

        self.vertices = self.original_vertices.copy()
        self.vertex_count = 3

        # OpenCL buffers
        self.cl_original = cl.Buffer(
            self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=self.original_vertices
        )
        self.cl_vertices = cl.Buffer(
            self.cl_ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=self.vertices
        )

        # OpenGL buffers
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_DYNAMIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)

    def render_frame(self):
        angle = time.time() - self.start_time

        # OpenCL: rotate
        self.rotate_kernel(
            self.cl_queue, (self.vertex_count,), None,
            self.cl_vertices,
            self.cl_original,
            np.float32(angle)
        )

        # Copy to CPU
        cl.enqueue_copy(self.cl_queue, self.vertices, self.cl_vertices)
        self.cl_queue.finish()

        # Upload to OpenGL
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.vertices.nbytes, self.vertices)

        # Render
        glClearColor(0.1, 0.1, 0.15, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self.program)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

        glfw.swap_buffers(self.window)

    def should_close(self):
        return glfw.window_should_close(self.window)

    def poll_events(self):
        glfw.poll_events()

    def destroy(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteProgram(self.program)
        glfw.terminate()


# Main
window = Window(800, 600, "Triangle + OpenCL Rotation")

while not window.should_close():
    window.poll_events()
    window.render_frame()

window.destroy()
