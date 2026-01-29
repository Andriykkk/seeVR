import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np
import pyopencl as cl
import ctypes
import time
import math


class Shader:
    """Static shader manager - compiles once, reuses"""
    _program = None
    _view_loc = None
    _proj_loc = None
    _model_loc = None

    @classmethod
    def get_program(cls):
        if cls._program is None:
            cls._compile()
        return cls._program

    @classmethod
    def _compile(cls):
        vertex_shader = """
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aColor;
        layout(location = 2) in vec3 aNormal;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        out vec3 color;
        out vec3 normal;
        out vec3 fragPos;

        void main() {
            vec4 worldPos = model * vec4(aPos, 1.0);
            gl_Position = projection * view * worldPos;
            color = aColor;
            normal = mat3(transpose(inverse(model))) * aNormal;
            fragPos = vec3(worldPos);
        }
        """

        fragment_shader = """
        #version 330 core
        in vec3 color;
        in vec3 normal;
        in vec3 fragPos;
        out vec4 FragColor;

        void main() {
            vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
            vec3 norm = normalize(normal);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 ambient = 0.2 * color;
            vec3 diffuse = diff * color;
            FragColor = vec4(ambient + diffuse, 1.0);
        }
        """

        cls._program = shaders.compileProgram(
            shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
            shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )
        cls._view_loc = glGetUniformLocation(cls._program, "view")
        cls._proj_loc = glGetUniformLocation(cls._program, "projection")
        cls._model_loc = glGetUniformLocation(cls._program, "model")

    @classmethod
    def use(cls):
        glUseProgram(cls.get_program())

    @classmethod
    def set_view(cls, matrix):
        glUniformMatrix4fv(cls._view_loc, 1, GL_TRUE, matrix)

    @classmethod
    def set_projection(cls, matrix):
        glUniformMatrix4fv(cls._proj_loc, 1, GL_TRUE, matrix)

    @classmethod
    def set_model(cls, matrix):
        glUniformMatrix4fv(cls._model_loc, 1, GL_TRUE, matrix)


class Mesh:
    """Holds geometry data - can be shared between objects"""

    def __init__(self, vertices, indices):
        self.vertices = vertices
        self.indices = indices
        self.index_count = len(indices)

        # Create OpenGL buffers
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Stride = 9 floats = 36 bytes
        stride = 36
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)

    def destroy(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteBuffers(1, [self.ebo])


class Sphere:
    """Sphere instance with its own transform"""

    # Shared mesh for all spheres (created on first use)
    _mesh = None
    _mesh_settings = None

    def __init__(self, position=(0, 0, 0), rotation=(0, 0, 0), scale=1.0, color=(0.8, 0.4, 0.2),
                 velocity=(0, 0, 0), angular_velocity=(0, 0, 0)):
        self.position = np.array(position, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)  # euler angles in degrees
        self.scale = scale
        self.color = color

        # Movement
        self.velocity = np.array(velocity, dtype=np.float32)  # units per second
        self.angular_velocity = np.array(angular_velocity, dtype=np.float32)  # degrees per second

        # Create shared mesh if needed
        if Sphere._mesh is None:
            Sphere._create_mesh()

    def update(self, dt):
        """Update position and rotation based on velocities"""
        self.position += self.velocity * dt
        self.rotation += self.angular_velocity * dt

    @classmethod
    def _create_mesh(cls, lat_segments=16, lon_segments=32):
        """Create shared sphere mesh"""
        vertices = []
        indices = []

        # Unit sphere, white color (color applied via uniform later if needed)
        for lat in range(lat_segments + 1):
            theta = lat * math.pi / lat_segments
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)

            for lon in range(lon_segments + 1):
                phi = lon * 2 * math.pi / lon_segments
                sin_phi = math.sin(phi)
                cos_phi = math.cos(phi)

                x = cos_phi * sin_theta
                y = cos_theta
                z = sin_phi * sin_theta

                # position + color (white, will be tinted) + normal
                vertices.extend([
                    x, y, z,
                    1.0, 1.0, 1.0,
                    x, y, z
                ])

        for lat in range(lat_segments):
            for lon in range(lon_segments):
                current = lat * (lon_segments + 1) + lon
                next_row = current + lon_segments + 1
                indices.extend([current, next_row, current + 1])
                indices.extend([current + 1, next_row, next_row + 1])

        cls._mesh = Mesh(
            np.array(vertices, dtype=np.float32),
            np.array(indices, dtype=np.uint32)
        )

    def get_model_matrix(self):
        """Build model matrix from position, rotation, scale"""
        # Scale
        s = np.identity(4, dtype=np.float32)
        s[0, 0] = s[1, 1] = s[2, 2] = self.scale

        # Rotation (Y * X * Z order)
        rx, ry, rz = np.radians(self.rotation)

        rot_x = np.identity(4, dtype=np.float32)
        rot_x[1, 1] = math.cos(rx)
        rot_x[1, 2] = -math.sin(rx)
        rot_x[2, 1] = math.sin(rx)
        rot_x[2, 2] = math.cos(rx)

        rot_y = np.identity(4, dtype=np.float32)
        rot_y[0, 0] = math.cos(ry)
        rot_y[0, 2] = math.sin(ry)
        rot_y[2, 0] = -math.sin(ry)
        rot_y[2, 2] = math.cos(ry)

        rot_z = np.identity(4, dtype=np.float32)
        rot_z[0, 0] = math.cos(rz)
        rot_z[0, 1] = -math.sin(rz)
        rot_z[1, 0] = math.sin(rz)
        rot_z[1, 1] = math.cos(rz)

        r = rot_y @ rot_x @ rot_z

        # Translation
        t = np.identity(4, dtype=np.float32)
        t[0, 3] = self.position[0]
        t[1, 3] = self.position[1]
        t[2, 3] = self.position[2]

        return t @ r @ s

    def draw(self):
        Shader.set_model(self.get_model_matrix())
        self._mesh.draw()

    @classmethod
    def destroy_shared(cls):
        if cls._mesh:
            cls._mesh.destroy()
            cls._mesh = None


class Camera:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 10.0], dtype=np.float32)
        self.yaw = -90.0
        self.pitch = 0.0
        self.speed = 5.0
        self.sensitivity = 0.1
        self._update_vectors()

    def _update_vectors(self):
        rad_yaw = math.radians(self.yaw)
        rad_pitch = math.radians(self.pitch)

        self.front = np.array([
            math.cos(rad_pitch) * math.cos(rad_yaw),
            math.sin(rad_pitch),
            math.cos(rad_pitch) * math.sin(rad_yaw)
        ], dtype=np.float32)
        self.front /= np.linalg.norm(self.front)

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.right = np.cross(self.front, world_up)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.front)

    def process_keyboard(self, direction, delta_time):
        velocity = self.speed * delta_time
        if direction == "FORWARD":
            self.position += self.front * velocity
        if direction == "BACKWARD":
            self.position -= self.front * velocity
        if direction == "LEFT":
            self.position -= self.right * velocity
        if direction == "RIGHT":
            self.position += self.right * velocity
        if direction == "UP":
            self.position += self.up * velocity
        if direction == "DOWN":
            self.position -= self.up * velocity

    def process_mouse(self, x_offset, y_offset):
        self.yaw += x_offset * self.sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch + y_offset * self.sensitivity))
        self._update_vectors()

    def get_view_matrix(self):
        target = self.position + self.front
        f = self.front
        r = self.right
        u = self.up

        mat = np.identity(4, dtype=np.float32)
        mat[0, 0:3] = r
        mat[1, 0:3] = u
        mat[2, 0:3] = -f
        mat[0, 3] = -np.dot(r, self.position)
        mat[1, 3] = -np.dot(u, self.position)
        mat[2, 3] = np.dot(f, self.position)
        return mat


class Window:
    def __init__(self, width=800, height=600, title="OpenGL"):
        self.width = width
        self.height = height

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

        # Input
        self.keys = {}
        self.camera = Camera()
        self.last_mouse_x = width / 2
        self.last_mouse_y = height / 2
        self.first_mouse = True
        self.mouse_captured = True

        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_cursor_pos_callback(self.window, self._mouse_callback)
        glfw.set_framebuffer_size_callback(self.window, self._resize_callback)

        # Objects list
        self.objects = []

        self.last_frame = time.time()

    def add_object(self, obj):
        self.objects.append(obj)

    def _key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.keys[key] = True
            if key == glfw.KEY_ESCAPE:
                self.mouse_captured = not self.mouse_captured
                mode = glfw.CURSOR_DISABLED if self.mouse_captured else glfw.CURSOR_NORMAL
                glfw.set_input_mode(self.window, glfw.CURSOR, mode)
        elif action == glfw.RELEASE:
            self.keys[key] = False

    def _mouse_callback(self, window, x_pos, y_pos):
        if not self.mouse_captured:
            return
        if self.first_mouse:
            self.last_mouse_x = x_pos
            self.last_mouse_y = y_pos
            self.first_mouse = False

        x_offset = x_pos - self.last_mouse_x
        y_offset = self.last_mouse_y - y_pos
        self.last_mouse_x = x_pos
        self.last_mouse_y = y_pos
        self.camera.process_mouse(x_offset, y_offset)

    def _resize_callback(self, window, width, height):
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)

    def _process_input(self, dt):
        if self.keys.get(glfw.KEY_W) or self.keys.get(glfw.KEY_UP):
            self.camera.process_keyboard("FORWARD", dt)
        if self.keys.get(glfw.KEY_S) or self.keys.get(glfw.KEY_DOWN):
            self.camera.process_keyboard("BACKWARD", dt)
        if self.keys.get(glfw.KEY_A) or self.keys.get(glfw.KEY_LEFT):
            self.camera.process_keyboard("LEFT", dt)
        if self.keys.get(glfw.KEY_D) or self.keys.get(glfw.KEY_RIGHT):
            self.camera.process_keyboard("RIGHT", dt)
        if self.keys.get(glfw.KEY_SPACE):
            self.camera.process_keyboard("UP", dt)
        if self.keys.get(glfw.KEY_LEFT_SHIFT):
            self.camera.process_keyboard("DOWN", dt)

    def _perspective(self, fov, aspect, near, far):
        f = 1.0 / math.tan(math.radians(fov) / 2.0)
        mat = np.zeros((4, 4), dtype=np.float32)
        mat[0, 0] = f / aspect
        mat[1, 1] = f
        mat[2, 2] = (far + near) / (near - far)
        mat[2, 3] = (2 * far * near) / (near - far)
        mat[3, 2] = -1.0
        return mat

    def render_frame(self):
        current_time = time.time()
        dt = current_time - self.last_frame
        self.last_frame = current_time

        self._process_input(dt)

        # Setup
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.1, 0.1, 0.15, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Matrices
        view = self.camera.get_view_matrix()
        aspect = self.width / self.height if self.height > 0 else 1.0
        projection = self._perspective(45.0, aspect, 0.1, 100.0)

        Shader.use()
        Shader.set_view(view)
        Shader.set_projection(projection)

        # Update and draw all objects
        for obj in self.objects:
            obj.update(dt)
            obj.draw()

        glfw.swap_buffers(self.window)

    def should_close(self):
        return glfw.window_should_close(self.window)

    def poll_events(self):
        glfw.poll_events()

    def destroy(self):
        Sphere.destroy_shared()
        glfw.terminate()


# Main
window = Window(800, 600, "Multiple Spheres")

# Create 100 spheres in a grid with random rotation and angular velocity
import random
for y in range(-5, 5):
    for x in range(-5, 5):
        for z in range(-5, 5):
            sphere = Sphere(
                position=(x * 2, y * 2, z * 2),
                rotation=(random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360)),
                velocity=(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)),
                angular_velocity=(random.uniform(-90, 90), random.uniform(-90, 90), random.uniform(-90, 90)),
                scale=0.5
            )
            window.add_object(sphere)

while not window.should_close():
    window.poll_events()
    window.render_frame()

window.destroy()
