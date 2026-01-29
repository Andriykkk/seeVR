import taichi as ti
import random
import math

ti.init(arch=ti.gpu)

# Configuration
N_X, N_Z, N_Y = 10, 10, 10
N = N_X * N_Z * N_Y
RADIUS = 0.5

# Fields for sphere data
positions = ti.Vector.field(3, dtype=ti.f32, shape=N)
velocities = ti.Vector.field(3, dtype=ti.f32, shape=N)
rotations = ti.Vector.field(3, dtype=ti.f32, shape=N)
angular_velocities = ti.Vector.field(3, dtype=ti.f32, shape=N)
colors = ti.Vector.field(3, dtype=ti.f32, shape=N)


@ti.kernel
def update(dt: ti.f32):
    for i in range(N):
        positions[i] += velocities[i] * dt
        rotations[i] += angular_velocities[i] * dt


@ti.kernel
def check_collisions():
    # Reset colors to white
    for i in range(N):
        colors[i] = ti.Vector([1.0, 1.0, 1.0])

    # Check all pairs
    for i in range(N):
        for j in range(i + 1, N):
            dist = (positions[i] - positions[j]).norm()
            min_dist = RADIUS * 2
            if dist < min_dist:
                colors[i] = ti.Vector([1.0, 0.0, 0.0])
                colors[j] = ti.Vector([1.0, 0.0, 0.0])


def initialize():
    idx = 0
    for y in range(N_Y):
        for x in range(N_X):
            for z in range(N_Z):
                positions[idx] = [x * 2, y * 2, z * 2]
                velocities[idx] = [
                    random.uniform(-0.5, 0.5),
                    random.uniform(-0.5, 0.5),
                    random.uniform(-0.5, 0.5)
                ]
                rotations[idx] = [
                    random.uniform(0, 360),
                    random.uniform(0, 360),
                    random.uniform(0, 360)
                ]
                angular_velocities[idx] = [
                    random.uniform(-90, 90),
                    random.uniform(-90, 90),
                    random.uniform(-90, 90)
                ]
                colors[idx] = [1.0, 1.0, 1.0]
                idx += 1


# Initialize data
initialize()

# Create window and rendering objects
window = ti.ui.Window("Taichi Spheres", (800, 600), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# Camera setup
camera.position(0, 5, 20)
camera.lookat(0, 0, 0)
camera.up(0, 1, 0)

# Movement state
camera_pos = [0.0, 5.0, 20.0]
camera_yaw = -90.0
camera_pitch = -15.0
camera_speed = 10.0
mouse_sensitivity = 0.5
last_mouse_pos = None
mouse_captured = True

while window.running:
    dt = 1.0 / 60.0

    # Handle keyboard input
    if window.is_pressed('w'):
        # Move forward
        rad_yaw = math.radians(camera_yaw)
        rad_pitch = math.radians(camera_pitch)
        camera_pos[0] += math.cos(rad_yaw) * camera_speed * dt
        camera_pos[2] += math.sin(rad_yaw) * camera_speed * dt
    if window.is_pressed('s'):
        rad_yaw = math.radians(camera_yaw)
        camera_pos[0] -= math.cos(rad_yaw) * camera_speed * dt
        camera_pos[2] -= math.sin(rad_yaw) * camera_speed * dt
    if window.is_pressed('a'):
        rad_yaw = math.radians(camera_yaw - 90)
        camera_pos[0] += math.cos(rad_yaw) * camera_speed * dt
        camera_pos[2] += math.sin(rad_yaw) * camera_speed * dt
    if window.is_pressed('d'):
        rad_yaw = math.radians(camera_yaw + 90)
        camera_pos[0] += math.cos(rad_yaw) * camera_speed * dt
        camera_pos[2] += math.sin(rad_yaw) * camera_speed * dt
    if window.is_pressed(ti.ui.SPACE):
        camera_pos[1] += camera_speed * dt
    if window.is_pressed(ti.ui.SHIFT):
        camera_pos[1] -= camera_speed * dt

    # Handle mouse look (only when right mouse button held)
    curr_mouse = window.get_cursor_pos()
    if window.is_pressed(ti.ui.RMB) and last_mouse_pos is not None:
        dx = (curr_mouse[0] - last_mouse_pos[0]) * 200
        dy = (curr_mouse[1] - last_mouse_pos[1]) * 200
        camera_yaw += dx * mouse_sensitivity
        camera_pitch = max(-89, min(89, camera_pitch + dy * mouse_sensitivity))
    last_mouse_pos = curr_mouse

    # Update camera
    rad_yaw = math.radians(camera_yaw)
    rad_pitch = math.radians(camera_pitch)
    look_x = camera_pos[0] + math.cos(rad_pitch) * math.cos(rad_yaw)
    look_y = camera_pos[1] + math.sin(rad_pitch)
    look_z = camera_pos[2] + math.cos(rad_pitch) * math.sin(rad_yaw)

    camera.position(camera_pos[0], camera_pos[1], camera_pos[2])
    camera.lookat(look_x, look_y, look_z)

    # Physics update
    update(dt)
    check_collisions()

    # Render
    scene.set_camera(camera)
    scene.ambient_light((0.2, 0.2, 0.2))
    scene.point_light(pos=(10, 10, 10), color=(1, 1, 1))

    # Draw spheres as particles
    scene.particles(positions, radius=RADIUS, per_vertex_color=colors)

    canvas.scene(scene)
    window.show()
