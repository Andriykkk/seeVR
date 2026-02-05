"""Test script: Two spheres touching with mesh-based collision detection."""
import taichi as ti
import numpy as np
import time

ti.init(arch=ti.gpu)

# Initialize shared data fields
import kernels.data as data
data.init_scene()

from kernels.physics import (
    compute_local_vertices, update_geom_transforms, broad_phase_n_squared,
    narrow_phase, build_debug_geom_verts, build_debug_contacts
)
from kernels.mesh_processor import convex_hull


# GPU kernels for batch data copying
@ti.kernel
def _copy_vertices_batch(verts: ti.types.ndarray(dtype=ti.f32, ndim=2), offset: ti.i32):
    for i in range(verts.shape[0]):
        data.vertices[offset + i] = ti.Vector([verts[i, 0], verts[i, 1], verts[i, 2]])


@ti.kernel
def _copy_colors_batch(color: ti.types.ndarray(dtype=ti.f32, ndim=1), offset: ti.i32, count: ti.i32):
    for i in range(count):
        data.vertex_colors[offset + i] = ti.Vector([color[0], color[1], color[2]])


@ti.kernel
def _copy_indices_batch(indices: ti.types.ndarray(dtype=ti.i32, ndim=1), offset: ti.i32):
    for i in range(indices.shape[0]):
        data.indices[offset + i] = indices[i]


@ti.kernel
def _create_rigid_body(center: ti.types.vector(3, ti.f32), mass: ti.f32,
                       vert_start: ti.i32, vert_count: ti.i32) -> ti.i32:
    """Create a rigid body and return its index."""
    body_idx = ti.atomic_add(data.num_bodies[None], 1)

    data.bodies[body_idx].pos = center
    data.bodies[body_idx].quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
    data.bodies[body_idx].vel = ti.Vector([0.0, 0.0, 0.0])
    data.bodies[body_idx].omega = ti.Vector([0.0, 0.0, 0.0])
    data.bodies[body_idx].mass = mass
    data.bodies[body_idx].inv_mass = 1.0 / mass if mass > 0.0 else 0.0
    data.bodies[body_idx].vert_start = vert_start
    data.bodies[body_idx].vert_count = vert_count

    if mass > 0.0:
        data.bodies[body_idx].inertia = ti.Vector([1.0, 1.0, 1.0])
        data.bodies[body_idx].inv_inertia = ti.Vector([1.0, 1.0, 1.0])
    else:
        data.bodies[body_idx].inertia = ti.Vector([0.0, 0.0, 0.0])
        data.bodies[body_idx].inv_inertia = ti.Vector([0.0, 0.0, 0.0])

    return body_idx


@ti.kernel
def _create_hull_geom(body_idx: ti.i32,
                      hull_verts: ti.types.ndarray(dtype=ti.f32, ndim=2),
                      hull_faces: ti.types.ndarray(dtype=ti.i32, ndim=2)) -> ti.i32:
    """Create a mesh collision geometry from convex hull."""
    geom_idx = ti.atomic_add(data.num_geoms[None], 1)
    num_verts = hull_verts.shape[0]
    num_faces = hull_faces.shape[0]

    # Allocate space in collision_verts
    vert_start = ti.atomic_add(data.num_collision_verts[None], num_verts)
    for i in range(num_verts):
        data.collision_verts[vert_start + i] = ti.Vector([
            hull_verts[i, 0], hull_verts[i, 1], hull_verts[i, 2]
        ])

    # Allocate space in collision_faces (offset indices by vert_start)
    face_start = ti.atomic_add(data.num_collision_faces[None], num_faces)
    for i in range(num_faces):
        data.collision_faces[face_start + i] = ti.Vector([
            hull_faces[i, 0] + vert_start,
            hull_faces[i, 1] + vert_start,
            hull_faces[i, 2] + vert_start
        ])

    # Compute bounding box
    min_coord = ti.Vector([1e10, 1e10, 1e10])
    max_coord = ti.Vector([-1e10, -1e10, -1e10])
    for i in range(num_verts):
        v = ti.Vector([hull_verts[i, 0], hull_verts[i, 1], hull_verts[i, 2]])
        min_coord = ti.min(min_coord, v)
        max_coord = ti.max(max_coord, v)

    # Create GEOM_MESH
    data.geoms[geom_idx].geom_type = data.GEOM_MESH
    data.geoms[geom_idx].body_idx = body_idx
    data.geoms[geom_idx].local_pos = ti.Vector([0.0, 0.0, 0.0])
    data.geoms[geom_idx].local_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
    data.geoms[geom_idx].data = ti.Vector([
        ti.cast(vert_start, ti.f32),
        ti.cast(num_verts, ti.f32),
        ti.cast(face_start, ti.f32),
        ti.cast(num_faces, ti.f32),
        0.0, 0.0, 0.0
    ])
    data.geoms[geom_idx].world_pos = ti.Vector([0.0, 0.0, 0.0])
    data.geoms[geom_idx].world_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
    data.geoms[geom_idx].aabb_min = min_coord
    data.geoms[geom_idx].aabb_max = max_coord

    return geom_idx


def generate_sphere_mesh(radius, segments=12):
    """Generate UV sphere mesh vertices and faces."""
    lat_angles = np.linspace(0, np.pi, segments + 1)
    lon_angles = np.linspace(0, 2 * np.pi, segments + 1)

    lat_grid, lon_grid = np.meshgrid(lat_angles, lon_angles, indexing='ij')

    sin_lat = np.sin(lat_grid)
    cos_lat = np.cos(lat_grid)
    sin_lon = np.sin(lon_grid)
    cos_lon = np.cos(lon_grid)

    x = radius * cos_lon * sin_lat
    y = radius * cos_lat
    z = radius * sin_lon * sin_lat

    verts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)

    # Generate triangle indices
    faces = []
    for i in range(segments):
        for j in range(segments):
            current = i * (segments + 1) + j
            next_row = current + segments + 1
            faces.append([current, next_row, current + 1])
            faces.append([current + 1, next_row, next_row + 1])

    faces = np.array(faces, dtype=np.int32)
    return verts, faces


def add_sphere_with_mesh_collision(center, radius, color, segments=12, mass=1.0):
    """Add a sphere with mesh-based collision geometry."""
    cx, cy, cz = center
    start_vertex = data.num_vertices[None]

    # Generate sphere mesh
    local_verts, faces = generate_sphere_mesh(radius, segments)
    num_verts = local_verts.shape[0]

    # World-space vertices for rendering
    world_verts = local_verts + np.array([cx, cy, cz], dtype=np.float32)

    # Generate indices with offset
    indices = []
    for f in faces:
        indices.extend([start_vertex + f[0], start_vertex + f[1], start_vertex + f[2]])
    indices = np.array(indices, dtype=np.int32)
    num_tris = len(faces)

    # Copy to GPU
    _copy_vertices_batch(world_verts, start_vertex)
    _copy_colors_batch(np.array(color, dtype=np.float32), start_vertex, num_verts)
    _copy_indices_batch(indices, data.num_triangles[None] * 3)

    # Update counts
    data.num_vertices[None] = start_vertex + num_verts
    data.num_triangles[None] = data.num_triangles[None] + num_tris

    # Create rigid body
    body_idx = _create_rigid_body(ti.Vector(center), mass, start_vertex, num_verts)

    # Create convex hull for collision (sphere hull = sphere itself)
    hull_data = convex_hull(local_verts)
    hull_verts = hull_data['vertices']
    hull_faces = hull_data['faces']

    print(f"Sphere at {center}: {len(hull_verts)} hull verts, {len(hull_faces)} hull faces")

    # Create mesh collision geom
    _create_hull_geom(body_idx, hull_verts, hull_faces)

    return body_idx


def run_collision_detection():
    """Run the collision detection pipeline."""
    num_geoms = data.num_geoms[None]

    # Update world transforms and AABBs
    update_geom_transforms(num_geoms)

    # Broad phase
    data.num_collision_pairs[None] = 0
    broad_phase_n_squared(num_geoms)

    # Narrow phase
    data.num_contacts[None] = 0
    num_pairs = data.num_collision_pairs[None]
    if num_pairs > 0:
        narrow_phase(num_pairs)

    return data.num_contacts[None]


def main():
    # Disable gravity for this test
    data.gravity[None] = [0.0, 0.0, 0.0]

    # Two spheres with radius 1.0
    radius = 1.0
    # Position them so they just touch (centers 2*radius apart = touching)
    # Or slightly overlapping (centers < 2*radius)
    separation = 1.9  # Slightly overlapping (2.0 would be exactly touching)

    sphere1_center = (-separation / 2, 0, 0)
    sphere2_center = (separation / 2, 0, 0)

    print("Creating two spheres with mesh collision...")
    print(f"  Sphere 1: center={sphere1_center}, radius={radius}")
    print(f"  Sphere 2: center={sphere2_center}, radius={radius}")
    print(f"  Distance between centers: {separation}")
    print(f"  Expected overlap: {2 * radius - separation}")

    # Add spheres with mesh collision
    body1 = add_sphere_with_mesh_collision(sphere1_center, radius, (1.0, 0.3, 0.3), segments=16, mass=1.0)
    body2 = add_sphere_with_mesh_collision(sphere2_center, radius, (0.3, 0.3, 1.0), segments=16, mass=1.0)

    # Compute local vertices for physics
    compute_local_vertices(data.num_bodies[None])

    print(f"\nScene: {data.num_vertices[None]} vertices, {data.num_triangles[None]} triangles")
    print(f"Physics: {data.num_bodies[None]} bodies, {data.num_geoms[None]} geoms")
    print(f"Collision verts: {data.num_collision_verts[None]}, faces: {data.num_collision_faces[None]}")

    # Run collision detection
    print("\nRunning collision detection...")
    num_contacts = run_collision_detection()

    print(f"\nCollision pairs (broad phase): {data.num_collision_pairs[None]}")
    print(f"Contacts detected (narrow phase): {num_contacts}")

    if num_contacts > 0:
        print("\nContact details:")
        for i in range(min(num_contacts, 5)):  # Show up to 5 contacts
            contact = data.contacts[i]
            print(f"  Contact {i}:")
            print(f"    Point: ({contact.point[0]:.3f}, {contact.point[1]:.3f}, {contact.point[2]:.3f})")
            print(f"    Normal: ({contact.normal[0]:.3f}, {contact.normal[1]:.3f}, {contact.normal[2]:.3f})")
            print(f"    Depth: {contact.depth:.4f}")
            print(f"    Bodies: {contact.body_a} <-> {contact.body_b}")

    # Create window for visualization
    print("\nOpening visualization window...")
    print("Controls: WASD to move, Right-click drag to look, Space/Shift for up/down")

    window = ti.ui.Window("Sphere Collision Test", (800, 600), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    camera.position(0, 0, 6)
    camera.lookat(0, 0, 0)

    # Build debug visualization
    num_coll_verts = data.num_collision_verts[None]
    num_coll_faces = data.num_collision_faces[None]
    if num_coll_faces > 0:
        build_debug_geom_verts(data.num_geoms[None], num_coll_verts, num_coll_faces)
    if num_contacts > 0:
        build_debug_contacts(num_contacts)

    last_mouse = None
    cam_yaw = 0.0
    cam_pitch = 0.0
    cam_pos = [0.0, 0.0, 6.0]

    while window.running:
        # Handle camera input
        dt = 1.0 / 60.0

        # Keyboard
        if window.is_pressed('w'):
            cam_pos[2] -= 5.0 * dt
        if window.is_pressed('s'):
            cam_pos[2] += 5.0 * dt
        if window.is_pressed('a'):
            cam_pos[0] -= 5.0 * dt
        if window.is_pressed('d'):
            cam_pos[0] += 5.0 * dt
        if window.is_pressed(ti.ui.SPACE):
            cam_pos[1] += 5.0 * dt
        if window.is_pressed(ti.ui.SHIFT):
            cam_pos[1] -= 5.0 * dt

        # Mouse look
        curr_mouse = window.get_cursor_pos()
        if window.is_pressed(ti.ui.RMB) and last_mouse is not None:
            dx = (curr_mouse[0] - last_mouse[0]) * 100
            dy = (curr_mouse[1] - last_mouse[1]) * 100
            cam_yaw += dx * 0.5
            cam_pitch = max(-89, min(89, cam_pitch + dy * 0.5))
        last_mouse = curr_mouse

        # Update camera
        import math
        rad_yaw = math.radians(cam_yaw)
        rad_pitch = math.radians(cam_pitch)
        look_x = math.cos(rad_pitch) * math.sin(rad_yaw)
        look_y = math.sin(rad_pitch)
        look_z = -math.cos(rad_pitch) * math.cos(rad_yaw)

        camera.position(cam_pos[0], cam_pos[1], cam_pos[2])
        camera.lookat(cam_pos[0] + look_x, cam_pos[1] + look_y, cam_pos[2] + look_z)

        # Render
        scene.set_camera(camera)
        scene.ambient_light((0.3, 0.3, 0.3))
        scene.point_light(pos=(5, 5, 5), color=(1, 1, 1))

        # Render sphere meshes
        scene.mesh(
            data.vertices,
            indices=data.indices,
            per_vertex_color=data.vertex_colors,
            two_sided=True
        )

        # Render collision hulls (wireframe-ish via different color)
        if num_coll_faces > 0:
            scene.mesh(
                data.debug_geom_verts,
                indices=data.debug_geom_indices,
                per_vertex_color=data.debug_geom_colors,
                two_sided=True
            )

        # Render contact points
        if num_contacts > 0:
            scene.particles(
                data.debug_contact_points,
                radius=0.05,
                color=(1.0, 1.0, 0.0),
                index_count=num_contacts
            )
            scene.lines(
                data.debug_contact_normals,
                width=4.0,
                color=(0.0, 1.0, 1.0),
                vertex_count=num_contacts * 2
            )

        canvas.scene(scene)

        # GUI
        gui = window.get_gui()
        gui.begin("Info", 0.02, 0.02, 0.3, 0.2)
        gui.text(f"Collision pairs: {data.num_collision_pairs[None]}")
        gui.text(f"Contacts: {num_contacts}")
        if num_contacts > 0:
            gui.text(f"Depth: {data.contacts[0].depth:.4f}")
        gui.end()

        window.show()


if __name__ == '__main__':
    main()
