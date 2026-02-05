import taichi as ti
import numpy as np
import math
import argparse
import time
from benchmark import benchmark, is_enabled_benchmark

ti.init(arch=ti.gpu)

# Initialize shared data fields (must be after ti.init)
import kernels.data as data
data.init_scene()

# Initialize radix sort (must be after ti.init and before using LBVH)
from kernels.radix_sort import init_radix_sort
init_radix_sort()

# Import kernels (must be after init_scene)
from kernels.bvh import build_lbvh
from kernels.raytracing import run_raytrace
from kernels.debug import run_debug_bvh
from kernels.physics import (
    compute_local_vertices, apply_gravity, integrate_bodies,
    update_render_vertices, update_geom_transforms, broad_phase_n_squared,
    narrow_phase, highlight_contact_bodies, solve_contacts, build_debug_geom_verts,
    build_debug_contacts
)
from kernels.mesh_processor import load_collision_mesh

# Constants from data module
WIDTH, HEIGHT = data.WIDTH, data.HEIGHT

# Path tracing settings (can be changed at runtime via GUI)
class Settings:
    def __init__(self):
        self.max_bounces = 2
        self.samples_per_pixel = 2
        self.sky_intensity = 1.0
        self.debug_bvh = False
        self.highlight_contacts = False
        self.target_fps = 30  # FPS limiter (0 = unlimited)
        self.render_geoms = False  # Visual debug: render collision geom wireframes
        self.debug_contacts = False  # Print contact info after narrow phase

settings = Settings()


# GPU kernels for batch data copying
@ti.kernel
def _copy_vertices_batch(verts: ti.types.ndarray(dtype=ti.f32, ndim=2), offset: ti.i32):
    for i in range(verts.shape[0]):
        data.vertices[offset + i] = ti.Vector([verts[i, 0], verts[i, 1], verts[i, 2]])


@ti.kernel
def _copy_velocities_batch(vel: ti.types.ndarray(dtype=ti.f32, ndim=1), offset: ti.i32, count: ti.i32):
    for i in range(count):
        data.velocities[offset + i] = ti.Vector([vel[0], vel[1], vel[2]])


@ti.kernel
def _copy_indices_batch(indices: ti.types.ndarray(dtype=ti.i32, ndim=1), offset: ti.i32):
    for i in range(indices.shape[0]):
        data.indices[offset + i] = indices[i]


@ti.kernel
def _copy_colors_batch(color: ti.types.ndarray(dtype=ti.f32, ndim=1), offset: ti.i32, count: ti.i32):
    for i in range(count):
        data.vertex_colors[offset + i] = ti.Vector([color[0], color[1], color[2]])


@ti.data_oriented
class Scene:
    """GPU-first scene builder. All geometry generation happens on GPU."""

    def __init__(self):
        pass

    @ti.kernel
    def _create_rigid_body(self, center: ti.types.vector(3, ti.f32), mass: ti.f32,
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
    def _create_sphere_geom(self, body_idx: ti.i32, radius: ti.f32) -> ti.i32:
        """Create a sphere collision geometry."""
        geom_idx = ti.atomic_add(data.num_geoms[None], 1)

        data.geoms[geom_idx].geom_type = data.GEOM_SPHERE
        data.geoms[geom_idx].body_idx = body_idx
        data.geoms[geom_idx].local_pos = ti.Vector([0.0, 0.0, 0.0])
        data.geoms[geom_idx].local_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
        data.geoms[geom_idx].data = ti.Vector([radius, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        data.geoms[geom_idx].world_pos = ti.Vector([0.0, 0.0, 0.0])
        data.geoms[geom_idx].world_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
        data.geoms[geom_idx].aabb_min = ti.Vector([0.0, 0.0, 0.0])
        data.geoms[geom_idx].aabb_max = ti.Vector([0.0, 0.0, 0.0])

        return geom_idx

    @ti.kernel
    def _create_box_geom(self, body_idx: ti.i32, half_extents: ti.types.vector(3, ti.f32)) -> ti.i32:
        """Create a box collision geometry."""
        geom_idx = ti.atomic_add(data.num_geoms[None], 1)

        data.geoms[geom_idx].geom_type = data.GEOM_BOX
        data.geoms[geom_idx].body_idx = body_idx
        data.geoms[geom_idx].local_pos = ti.Vector([0.0, 0.0, 0.0])
        data.geoms[geom_idx].local_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
        data.geoms[geom_idx].data = ti.Vector([half_extents[0], half_extents[1], half_extents[2], 0.0, 0.0, 0.0, 0.0])
        data.geoms[geom_idx].world_pos = ti.Vector([0.0, 0.0, 0.0])
        data.geoms[geom_idx].world_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
        data.geoms[geom_idx].aabb_min = ti.Vector([0.0, 0.0, 0.0])
        data.geoms[geom_idx].aabb_max = ti.Vector([0.0, 0.0, 0.0])

        return geom_idx

    @ti.kernel
    def _create_hull_geom(self, body_idx: ti.i32,
                          hull_verts: ti.types.ndarray(dtype=ti.f32, ndim=2),
                          hull_faces: ti.types.ndarray(dtype=ti.i32, ndim=2),
                          mesh_subtype: ti.i32) -> ti.i32:
        """Create a mesh collision geometry from convex hull vertices and faces.

        Hull vertices should already be in local space (centered, scaled, rotated).
        """
        geom_idx = ti.atomic_add(data.num_geoms[None], 1)
        num_verts = hull_verts.shape[0]
        num_faces = hull_faces.shape[0]

        # Allocate space in collision_verts and copy hull vertices (already in local space)
        vert_start = ti.atomic_add(data.num_collision_verts[None], num_verts)
        for i in range(num_verts):
            data.collision_verts[vert_start + i] = ti.Vector([
                hull_verts[i, 0], hull_verts[i, 1], hull_verts[i, 2]
            ])

        # Allocate space in collision_faces and copy (offset indices by vert_start)
        face_start = ti.atomic_add(data.num_collision_faces[None], num_faces)
        for i in range(num_faces):
            data.collision_faces[face_start + i] = ti.Vector([
                hull_faces[i, 0] + vert_start,
                hull_faces[i, 1] + vert_start,
                hull_faces[i, 2] + vert_start
            ])

        # Compute bounding box of hull vertices
        min_coord = ti.Vector([1e10, 1e10, 1e10])
        max_coord = ti.Vector([-1e10, -1e10, -1e10])
        for i in range(num_verts):
            v = ti.Vector([hull_verts[i, 0], hull_verts[i, 1], hull_verts[i, 2]])
            min_coord = ti.min(min_coord, v)
            max_coord = ti.max(max_coord, v)

        # Create GEOM_MESH with hull data
        # MESH data: [vert_start, vert_count, face_start, face_count, 0, 0, mesh_subtype]
        data.geoms[geom_idx].geom_type = data.GEOM_MESH
        data.geoms[geom_idx].body_idx = body_idx
        data.geoms[geom_idx].local_pos = ti.Vector([0.0, 0.0, 0.0])
        data.geoms[geom_idx].local_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
        data.geoms[geom_idx].data = ti.Vector([
            ti.cast(vert_start, ti.f32),
            ti.cast(num_verts, ti.f32),
            ti.cast(face_start, ti.f32),
            ti.cast(num_faces, ti.f32),
            0.0, 0.0,
            ti.cast(mesh_subtype, ti.f32)
        ])
        data.geoms[geom_idx].world_pos = ti.Vector([0.0, 0.0, 0.0])
        data.geoms[geom_idx].world_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
        data.geoms[geom_idx].aabb_min = min_coord
        data.geoms[geom_idx].aabb_max = max_coord

        return geom_idx

    @ti.kernel
    def _set_box_inertia(self, body_idx: ti.i32, half_extents: ti.types.vector(3, ti.f32)):
        """Set proper box inertia tensor."""
        mass = data.bodies[body_idx].mass
        if mass > 0.0:
            wx, wy, wz = half_extents[0] * 2.0, half_extents[1] * 2.0, half_extents[2] * 2.0
            ix = mass / 12.0 * (wy*wy + wz*wz)
            iy = mass / 12.0 * (wx*wx + wz*wz)
            iz = mass / 12.0 * (wx*wx + wy*wy)
            data.bodies[body_idx].inertia = ti.Vector([ix, iy, iz])
            data.bodies[body_idx].inv_inertia = ti.Vector([
                1.0/ix if ix > 0.0 else 0.0,
                1.0/iy if iy > 0.0 else 0.0,
                1.0/iz if iz > 0.0 else 0.0
            ])

    @ti.kernel
    def _set_sphere_inertia(self, body_idx: ti.i32, radius: ti.f32):
        """Set proper sphere inertia tensor."""
        mass = data.bodies[body_idx].mass
        if mass > 0.0:
            inertia = 0.4 * mass * radius * radius
            data.bodies[body_idx].inertia = ti.Vector([inertia, inertia, inertia])
            data.bodies[body_idx].inv_inertia = ti.Vector([1.0/inertia, 1.0/inertia, 1.0/inertia])

    @ti.kernel
    def _set_mesh_inertia(self, body_idx: ti.i32, hull_verts: ti.types.ndarray(dtype=ti.f32, ndim=2),
                          mesh_volume: ti.f32):
        """Set inertia tensor based on original mesh volume and bounding box."""
        mass = data.bodies[body_idx].mass
        if mass > 0.0:
            # Compute bounding box dimensions
            min_coord = ti.Vector([1e10, 1e10, 1e10])
            max_coord = ti.Vector([-1e10, -1e10, -1e10])

            for i in range(hull_verts.shape[0]):
                v = ti.Vector([hull_verts[i, 0], hull_verts[i, 1], hull_verts[i, 2]])
                min_coord = ti.min(min_coord, v)
                max_coord = ti.max(max_coord, v)

            # Box dimensions (width, height, depth)
            dims = max_coord - min_coord

            # Calculate bounding box volume
            bbox_volume = dims[0] * dims[1] * dims[2]

            # Volume ratio: how much of the bounding box is filled by the original mesh
            # This represents the actual mass distribution (not the hull)
            volume_ratio = mesh_volume / bbox_volume if bbox_volume > 1e-8 else 1.0
            volume_ratio = ti.max(volume_ratio, 0.1)  # Clamp to avoid too small values

            # Box inertia tensor scaled by volume ratio
            # For a solid box: I = (mass/12) * (height^2 + depth^2), etc.
            # Scaled by volume_ratio since mass is distributed in mesh_volume, not bbox_volume
            ix = (mass * volume_ratio) / 12.0 * (dims[1]*dims[1] + dims[2]*dims[2])
            iy = (mass * volume_ratio) / 12.0 * (dims[0]*dims[0] + dims[2]*dims[2])
            iz = (mass * volume_ratio) / 12.0 * (dims[0]*dims[0] + dims[1]*dims[1])

            data.bodies[body_idx].inertia = ti.Vector([ix, iy, iz])
            data.bodies[body_idx].inv_inertia = ti.Vector([
                1.0/ix if ix > 0.0 else 0.0,
                1.0/iy if iy > 0.0 else 0.0,
                1.0/iz if iz > 0.0 else 0.0
            ])

    @ti.kernel
    def _add_box_gpu(self, center: ti.types.vector(3, ti.f32),
                     half_size: ti.types.vector(3, ti.f32),
                     color: ti.types.vector(3, ti.f32),
                     mass: ti.f32) -> ti.i32:
        """Add a box - everything on GPU, returns body index."""
        # Atomically allocate slots on GPU
        vert_start = ti.atomic_add(data.num_vertices[None], 8)
        tri_start = ti.atomic_add(data.num_triangles[None], 12)
        body_idx = ti.atomic_add(data.num_bodies[None], 1)
        geom_idx = ti.atomic_add(data.num_geoms[None], 1)

        # Box topology
        box_verts = ti.static([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ])
        box_faces = ti.static([
            [0, 1, 2], [0, 2, 3],  # Front
            [5, 4, 7], [5, 7, 6],  # Back
            [4, 0, 3], [4, 3, 7],  # Left
            [1, 5, 6], [1, 6, 2],  # Right
            [3, 2, 6], [3, 6, 7],  # Top
            [4, 5, 1], [4, 1, 0],  # Bottom
        ])

        # Generate vertices and indices
        for i in ti.static(range(8)):
            data.vertices[vert_start + i] = center + half_size * ti.Vector(box_verts[i])
            data.vertex_colors[vert_start + i] = color
            data.velocities[vert_start + i] = ti.Vector([0.0, 0.0, 0.0])

        for i in ti.static(range(12)):
            for j in ti.static(range(3)):
                data.indices[tri_start * 3 + i * 3 + j] = vert_start + box_faces[i][j]

        # Create rigid body
        data.bodies[body_idx].pos = center
        data.bodies[body_idx].quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
        data.bodies[body_idx].vel = ti.Vector([0.0, 0.0, 0.0])
        data.bodies[body_idx].omega = ti.Vector([0.0, 0.0, 0.0])
        data.bodies[body_idx].mass = mass
        data.bodies[body_idx].inv_mass = 1.0 / mass if mass > 0.0 else 0.0
        data.bodies[body_idx].vert_start = vert_start
        data.bodies[body_idx].vert_count = 8

        # Set inertia
        if mass > 0.0:
            wx, wy, wz = half_size[0] * 2.0, half_size[1] * 2.0, half_size[2] * 2.0
            ix = mass / 12.0 * (wy*wy + wz*wz)
            iy = mass / 12.0 * (wx*wx + wz*wz)
            iz = mass / 12.0 * (wx*wx + wy*wy)
            data.bodies[body_idx].inertia = ti.Vector([ix, iy, iz])
            data.bodies[body_idx].inv_inertia = ti.Vector([
                1.0/ix if ix > 0.0 else 0.0,
                1.0/iy if iy > 0.0 else 0.0,
                1.0/iz if iz > 0.0 else 0.0
            ])
        else:
            data.bodies[body_idx].inertia = ti.Vector([0.0, 0.0, 0.0])
            data.bodies[body_idx].inv_inertia = ti.Vector([0.0, 0.0, 0.0])

        # Create collision geom
        data.geoms[geom_idx].geom_type = data.GEOM_BOX
        data.geoms[geom_idx].body_idx = body_idx
        data.geoms[geom_idx].local_pos = ti.Vector([0.0, 0.0, 0.0])
        data.geoms[geom_idx].local_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
        data.geoms[geom_idx].data = ti.Vector([half_size[0], half_size[1], half_size[2], 0.0, 0.0, 0.0, 0.0])
        data.geoms[geom_idx].world_pos = ti.Vector([0.0, 0.0, 0.0])
        data.geoms[geom_idx].world_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
        data.geoms[geom_idx].aabb_min = ti.Vector([0.0, 0.0, 0.0])
        data.geoms[geom_idx].aabb_max = ti.Vector([0.0, 0.0, 0.0])

        return body_idx

    @benchmark
    def add_sphere(self, center, radius, color=(1.0, 1.0, 1.0), velocity=(0, 0, 0),
                   segments=16, is_static=False):
        """Add a UV sphere as triangles (batch CPU -> GPU transfer)"""
        cx, cy, cz = center
        start_vertex = data.num_vertices[None]

        # Generate vertices using NumPy (vectorized on CPU)
        lat_angles = np.linspace(0, np.pi, segments + 1)
        lon_angles = np.linspace(0, 2 * np.pi, segments + 1)

        # Meshgrid for all lat/lon combinations
        lat_grid, lon_grid = np.meshgrid(lat_angles, lon_angles, indexing='ij')

        # Spherical to Cartesian (vectorized)
        sin_lat = np.sin(lat_grid)
        cos_lat = np.cos(lat_grid)
        sin_lon = np.sin(lon_grid)
        cos_lon = np.cos(lon_grid)

        x = cx + radius * cos_lon * sin_lat
        y = cy + radius * cos_lat
        z = cz + radius * sin_lon * sin_lat

        # Stack into (N, 3) array
        verts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)
        num_verts = verts.shape[0]

        # Generate triangle indices using NumPy (vectorized)
        lat_idx = np.arange(segments)
        lon_idx = np.arange(segments)
        lat_grid_i, lon_grid_i = np.meshgrid(lat_idx, lon_idx, indexing='ij')

        current = start_vertex + lat_grid_i * (segments + 1) + lon_grid_i
        next_row = current + segments + 1

        # Two triangles per quad
        tri1 = np.stack([current, next_row, current + 1], axis=-1)
        tri2 = np.stack([current + 1, next_row, next_row + 1], axis=-1)

        # Interleave and flatten to get final indices
        indices = np.empty((segments, segments, 2, 3), dtype=np.int32)
        indices[:, :, 0, :] = tri1
        indices[:, :, 1, :] = tri2
        indices = indices.reshape(-1).astype(np.int32)
        num_tris = len(indices) // 3

        # Batch copy to GPU
        _copy_vertices_batch(verts, start_vertex)
        _copy_velocities_batch(np.array(velocity, dtype=np.float32), start_vertex, num_verts)
        _copy_indices_batch(indices, data.num_triangles[None] * 3)
        _copy_colors_batch(np.array(color, dtype=np.float32), start_vertex, num_verts)

        # Update counts
        data.num_vertices[None] = start_vertex + num_verts
        data.num_triangles[None] = data.num_triangles[None] + num_tris

        # Create physics body and collision geom
        mass = 0.0 if is_static else 1.0
        body_idx = self._create_rigid_body(ti.Vector(center), mass, start_vertex, num_verts)
        self._create_sphere_geom(body_idx, radius)
        self._set_sphere_inertia(body_idx, radius)

        return body_idx

    @benchmark
    def add_box(self, center, size, color=(1.0, 1.0, 1.0), velocity=(0, 0, 0), is_static=False):
        """Add a box - everything happens on GPU in one kernel call."""
        mass = 0.0 if is_static else 1.0
        half_size = ti.Vector([size[0] / 2, size[1] / 2, size[2] / 2])
        box = self._add_box_gpu(ti.Vector(center), half_size, ti.Vector(color), mass)

        if is_enabled_benchmark():
            ti.sync()

        return box

    @benchmark
    def add_plane(self, center, normal, size, color=(0.5, 0.5, 0.5)):
        """Add a plane as 2 triangles"""
        nx, ny, nz = normal
        if abs(nx) < 0.9:
            tangent = (1, 0, 0)
        else:
            tangent = (0, 1, 0)

        tx, ty, tz = tangent
        bx = ny * tz - nz * ty
        by = nz * tx - nx * tz
        bz = nx * ty - ny * tx
        length = math.sqrt(bx * bx + by * by + bz * bz)
        bx, by, bz = bx / length, by / length, bz / length

        t2x = ny * bz - nz * by
        t2y = nz * bx - nx * bz
        t2z = nx * by - ny * bx

        cx, cy, cz = center
        s = size / 2
        start_vertex = data.num_vertices[None]

        verts = [
            (cx - s * bx - s * t2x, cy - s * by - s * t2y, cz - s * bz - s * t2z),
            (cx + s * bx - s * t2x, cy + s * by - s * t2y, cz + s * bz - s * t2z),
            (cx + s * bx + s * t2x, cy + s * by + s * t2y, cz + s * bz + s * t2z),
            (cx - s * bx + s * t2x, cy - s * by + s * t2y, cz - s * bz + s * t2z),
        ]

        for v in verts:
            data.vertices[data.num_vertices[None]] = v
            data.velocities[data.num_vertices[None]] = (0, 0, 0)
            data.num_vertices[None] += 1

        idx = data.num_triangles[None] * 3
        data.indices[idx] = start_vertex
        data.indices[idx + 1] = start_vertex + 1
        data.indices[idx + 2] = start_vertex + 2
        data.num_triangles[None] += 1

        idx = data.num_triangles[None] * 3
        data.indices[idx] = start_vertex
        data.indices[idx + 1] = start_vertex + 2
        data.indices[idx + 2] = start_vertex + 3
        data.num_triangles[None] += 1

        # Set vertex colors
        for i in range(start_vertex, data.num_vertices[None]):
            data.vertex_colors[i] = color

        # Create physics body and plane geom (static infinite plane)
        vert_count = 4
        body_idx = self._create_rigid_body(ti.Vector(center), 0.0, start_vertex, vert_count)  # mass=0 (static)

        # Create plane geom - store normal in data field
        geom_idx = ti.atomic_add(data.num_geoms[None], 1)
        data.geoms[geom_idx].geom_type = data.GEOM_PLANE
        data.geoms[geom_idx].body_idx = body_idx
        data.geoms[geom_idx].local_pos = ti.Vector([0.0, 0.0, 0.0])
        data.geoms[geom_idx].local_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
        # Store plane normal in data field
        data.geoms[geom_idx].data = ti.Vector([normal[0], normal[1], normal[2], 0.0, 0.0, 0.0, 0.0])
        data.geoms[geom_idx].world_pos = ti.Vector(center)
        data.geoms[geom_idx].world_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
        data.geoms[geom_idx].aabb_min = ti.Vector([-1e6, -1e6, -1e6])
        data.geoms[geom_idx].aabb_max = ti.Vector([1e6, 1e6, 1e6])

        return body_idx

    @ti.kernel
    def _process_mesh_gpu(self, raw_verts: ti.types.ndarray(dtype=ti.f32, ndim=2),
                          faces: ti.types.ndarray(dtype=ti.i32, ndim=2),
                          center: ti.types.vector(3, ti.f32),
                          target_size: ti.f32,
                          rotation: ti.types.vector(3, ti.f32),  # rx, ry, rz in radians
                          color: ti.types.vector(3, ti.f32),
                          velocity: ti.types.vector(3, ti.f32),
                          mass: ti.f32) -> ti.i32:
        """Process mesh entirely on GPU: transform, create body, store geometry.

        Returns body_idx.
        """
        num_verts = raw_verts.shape[0]
        num_faces = faces.shape[0]

        # Atomically allocate slots
        vert_start = ti.atomic_add(data.num_vertices[None], num_verts)
        tri_start = ti.atomic_add(data.num_triangles[None], num_faces)
        body_idx = ti.atomic_add(data.num_bodies[None], 1)

        # Step 1: Calculate bounding box on GPU
        min_coord = ti.Vector([1e10, 1e10, 1e10])
        max_coord = ti.Vector([-1e10, -1e10, -1e10])

        for i in range(num_verts):
            v = ti.Vector([raw_verts[i, 0], raw_verts[i, 1], raw_verts[i, 2]])
            min_coord = ti.min(min_coord, v)
            max_coord = ti.max(max_coord, v)

        mesh_center = (min_coord + max_coord) * 0.5
        extent = max_coord - min_coord
        max_extent = ti.max(ti.max(extent[0], extent[1]), extent[2])
        scale = target_size / max_extent if max_extent > 1e-8 else 1.0

        # Step 2: Build rotation matrix (Y * X * Z order)
        rx, ry, rz = rotation[0], rotation[1], rotation[2]
        cos_x, sin_x = ti.cos(rx), ti.sin(rx)
        cos_y, sin_y = ti.cos(ry), ti.sin(ry)
        cos_z, sin_z = ti.cos(rz), ti.sin(rz)

        # Rotation matrices (row-major)
        Ry = ti.Matrix([
            [cos_y, 0.0, sin_y],
            [0.0, 1.0, 0.0],
            [-sin_y, 0.0, cos_y]
        ])
        Rx = ti.Matrix([
            [1.0, 0.0, 0.0],
            [0.0, cos_x, -sin_x],
            [0.0, sin_x, cos_x]
        ])
        Rz = ti.Matrix([
            [cos_z, -sin_z, 0.0],
            [sin_z, cos_z, 0.0],
            [0.0, 0.0, 1.0]
        ])
        R = Rz @ Rx @ Ry

        # Step 3: Transform vertices and store
        for i in range(num_verts):
            v = ti.Vector([raw_verts[i, 0], raw_verts[i, 1], raw_verts[i, 2]])
            # Center, scale, rotate, translate
            v = (v - mesh_center) * scale
            v = R @ v
            v = v + center

            data.vertices[vert_start + i] = v
            data.vertex_colors[vert_start + i] = color
            data.velocities[vert_start + i] = velocity

        # Step 4: Store face indices
        for i in range(num_faces):
            for j in ti.static(range(3)):
                data.indices[tri_start * 3 + i * 3 + j] = vert_start + faces[i, j]

        # Step 5: Create rigid body
        data.bodies[body_idx].pos = center
        data.bodies[body_idx].quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
        data.bodies[body_idx].vel = ti.Vector([0.0, 0.0, 0.0])
        data.bodies[body_idx].omega = ti.Vector([0.0, 0.0, 0.0])
        data.bodies[body_idx].mass = mass
        data.bodies[body_idx].inv_mass = 1.0 / mass if mass > 0.0 else 0.0
        data.bodies[body_idx].vert_start = vert_start
        data.bodies[body_idx].vert_count = num_verts

        # Default inertia (will be updated by collision geom creation)
        if mass > 0.0:
            data.bodies[body_idx].inertia = ti.Vector([1.0, 1.0, 1.0])
            data.bodies[body_idx].inv_inertia = ti.Vector([1.0, 1.0, 1.0])
        else:
            data.bodies[body_idx].inertia = ti.Vector([0.0, 0.0, 0.0])
            data.bodies[body_idx].inv_inertia = ti.Vector([0.0, 0.0, 0.0])

        return body_idx

    @benchmark
    def add_mesh_from_obj(self, filename, center=(0, 0, 0), size=1.0, rotation=(0, 0, 0),
                          color=(1.0, 1.0, 1.0), velocity=(0, 0, 0), is_static=False,
                          convexify=True, collision_threshold=0.05):
        """Load mesh from OBJ file - CPU reads file, GPU processes everything else.

        Args:
            rotation: (rx, ry, rz) in degrees - applied in Y, X, Z order
            is_static: If True, mass = 0 (immovable)
            convexify: If True, use convex hulls. If False, use SDF.
            collision_threshold: Max volume error for single hull
        """
        # CPU only reads file and extracts raw data
        raw_verts = []
        faces = []

        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'v':
                    raw_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'f':
                    indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                    # Triangulate polygon faces
                    for i in range(1, len(indices) - 1):
                        faces.append([indices[0], indices[i], indices[i + 1]])

        if not raw_verts:
            return -1

        # Convert to numpy arrays
        verts_np = np.array(raw_verts, dtype=np.float32)
        faces_np = np.array(faces, dtype=np.int32)

        # Convert rotation to radians
        rotation_rad = np.radians(rotation).astype(np.float32)

        # GPU does all processing: transform, create body, store geometry
        mass = 0.0 if is_static else 1.0
        body_idx = self._process_mesh_gpu(
            verts_np, faces_np,
            ti.Vector(center), size,
            ti.Vector(rotation_rad),
            ti.Vector(color),
            ti.Vector(velocity),
            mass
        )

        # Read GPU-transformed vertices directly (they are exact)
        # GPU stores: (raw - mesh_center) * scale rotated + center
        vert_start = data.bodies[body_idx].vert_start
        vert_count = data.bodies[body_idx].vert_count
        center_np = np.array(center, dtype=np.float32)

        # Read vertices from GPU and subtract center to get local space
        gpu_verts = np.zeros((vert_count, 3), dtype=np.float32)
        for i in range(vert_count):
            v = data.vertices[vert_start + i]
            gpu_verts[i] = [v[0] - center_np[0], v[1] - center_np[1], v[2] - center_np[2]]

        # Compute hull from GPU local-space vertices (exact match with visual mesh)
        collision_data = load_collision_mesh(gpu_verts, faces_np, convexify, collision_threshold)

        # Create collision geoms based on strategy
        if collision_data['geom_type'] == data.GEOM_MESH:
            mesh_subtype = collision_data['mesh_subtype']
            hulls = collision_data['hulls']
            mesh_volume = collision_data['mesh_volume']

            print(f"  Collision: {len(hulls)} convex hull(s), error={collision_data['stats']['volume_error']:.4f}")

            # Create hull geoms - hull vertices are in local space (same as original_vertices)
            for i, hull in enumerate(hulls):
                hv = hull['vertices']  # already in local space from transformed_verts
                print(f"    Hull {i}: {len(hv)} verts, {len(hull['faces'])} faces")
                self._create_hull_geom(body_idx, hv, hull['faces'], mesh_subtype)

            # Set inertia based on original mesh volume (not hull volume)
            if mass > 0.0:
                self._set_mesh_inertia(body_idx, hulls[0]['vertices'], mesh_volume)

            print(f"  Created {len(hulls)} hull collision geom(s)")

        elif collision_data['geom_type'] == data.GEOM_SDF:
            sdf = collision_data['sdf']
            print(f"  Collision: SDF grid resolution={sdf['resolution']}")
            print(f"  WARNING: SDF collision not yet implemented - no collision geoms created")

        print(f"Loaded {filename}: {len(raw_verts)} vertices, {len(faces)} triangles (body {body_idx})")
        return body_idx

    def clear(self):
        """Clear all objects"""
        data.num_vertices[None] = 0
        data.num_triangles[None] = 0
        data.num_bodies[None] = 0
        data.num_geoms[None] = 0

scene = Scene()

@benchmark
def run_build_bvh():
    """Build BVH acceleration structure using GPU LBVH"""
    build_lbvh(data.num_triangles[None])
    if is_enabled_benchmark():
        ti.sync()

class Camera:
    """First-person camera with WASD + mouse look"""

    def __init__(self, position=(0, 5, 15), yaw=-90.0, pitch=-15.0):
        self.pos = list(position)
        self.yaw = yaw
        self.pitch = pitch
        self.speed = 10.0
        self.sensitivity = 0.5
        self._last_mouse = None

        # Computed vectors
        self.direction = [0, 0, -1]
        self.right = [1, 0, 0]
        self.up = [0, 1, 0]
        self._update_vectors()

    def _update_vectors(self):
        rad_yaw = math.radians(self.yaw)
        rad_pitch = math.radians(self.pitch)

        # Direction
        self.direction = [
            math.cos(rad_pitch) * math.cos(rad_yaw),
            math.sin(rad_pitch),
            math.cos(rad_pitch) * math.sin(rad_yaw)
        ]
        length = math.sqrt(sum(x * x for x in self.direction))
        self.direction = [x / length for x in self.direction]

        # Right (cross product with world up)
        world_up = [0, 1, 0]
        self.right = [
            self.direction[1] * world_up[2] - self.direction[2] * world_up[1],
            self.direction[2] * world_up[0] - self.direction[0] * world_up[2],
            self.direction[0] * world_up[1] - self.direction[1] * world_up[0]
        ]
        length = math.sqrt(sum(x * x for x in self.right))
        self.right = [x / length for x in self.right]

        # Up (cross product of right and direction)
        self.up = [
            self.right[1] * self.direction[2] - self.right[2] * self.direction[1],
            self.right[2] * self.direction[0] - self.right[0] * self.direction[2],
            self.right[0] * self.direction[1] - self.right[1] * self.direction[0]
        ]

    @benchmark
    def handle_input(self, window, dt):
        # Keyboard movement
        if window.is_pressed('w'):
            rad_yaw = math.radians(self.yaw)
            self.pos[0] += math.cos(rad_yaw) * self.speed * dt
            self.pos[2] += math.sin(rad_yaw) * self.speed * dt
        if window.is_pressed('s'):
            rad_yaw = math.radians(self.yaw)
            self.pos[0] -= math.cos(rad_yaw) * self.speed * dt
            self.pos[2] -= math.sin(rad_yaw) * self.speed * dt
        if window.is_pressed('a'):
            rad_yaw = math.radians(self.yaw - 90)
            self.pos[0] += math.cos(rad_yaw) * self.speed * dt
            self.pos[2] += math.sin(rad_yaw) * self.speed * dt
        if window.is_pressed('d'):
            rad_yaw = math.radians(self.yaw + 90)
            self.pos[0] += math.cos(rad_yaw) * self.speed * dt
            self.pos[2] += math.sin(rad_yaw) * self.speed * dt
        if window.is_pressed(ti.ui.SPACE):
            self.pos[1] += self.speed * dt
        if window.is_pressed(ti.ui.SHIFT):
            self.pos[1] -= self.speed * dt

        # Mouse look (right-click drag)
        curr_mouse = window.get_cursor_pos()
        if window.is_pressed(ti.ui.RMB) and self._last_mouse is not None:
            dx = (curr_mouse[0] - self._last_mouse[0]) * 200
            dy = (curr_mouse[1] - self._last_mouse[1]) * 200
            self.yaw += dx * self.sensitivity
            self.pitch = max(-89, min(89, self.pitch + dy * self.sensitivity))
        self._last_mouse = curr_mouse

        self._update_vectors()

    def apply_to_ti_camera(self, ti_camera):
        """Apply position/rotation to Taichi UI camera for rasterization"""
        ti_camera.position(self.pos[0], self.pos[1], self.pos[2])
        ti_camera.lookat(
            self.pos[0] + self.direction[0],
            self.pos[1] + self.direction[1],
            self.pos[2] + self.direction[2]
        )


# Benchmarked wrappers for kernels (ti.sync() needed for accurate GPU timing)
@benchmark
def run_physics(dt):
    """Run one physics step following the pipeline from TODO.md."""
    num_bodies = data.num_bodies[None]
    num_geoms = data.num_geoms[None]

    # Step 1: Apply external forces (gravity)
    apply_gravity(num_bodies, dt)

    # Step 2: Integrate positions/orientations
    integrate_bodies(num_bodies, dt)

    # Step 3: Update geom world transforms and AABBs (AFTER integration)
    update_geom_transforms(num_geoms)

    # Step 4: Broad phase - find candidate collision pairs
    data.num_collision_pairs[None] = 0  # Reset counter
    broad_phase_n_squared(num_geoms)

    # Step 5: Narrow phase - actual collision detection
    data.num_contacts[None] = 0  # Reset contact counter
    num_pairs = data.num_collision_pairs[None]
    if num_pairs > 0:
        narrow_phase(num_pairs)

    # Step 6: Solve contacts - correct velocities to prevent penetration
    num_contacts = data.num_contacts[None]
    if num_contacts > 0:
        solve_contacts(num_contacts, 10, dt)  # 10 iterations for convergence

    # Step 7: Update render mesh vertices
    update_render_vertices(num_bodies)

    # Optional: highlight bodies in contact
    if settings.highlight_contacts and num_contacts > 0:
        highlight_contact_bodies(num_contacts, num_bodies)

    # Optional: build debug geom vertices for rendering
    if settings.render_geoms:
        num_coll_verts = data.num_collision_verts[None]
        num_coll_faces = data.num_collision_faces[None]
        if num_coll_faces > 0:
            build_debug_geom_verts(num_geoms, num_coll_verts, num_coll_faces)

    # Build debug contact visualization
    if settings.debug_contacts and num_contacts > 0:
        build_debug_contacts(num_contacts)

    if is_enabled_benchmark():
        ti.sync()

@benchmark
def run_rasterize(ti_scene, ti_camera, canvas, cam):
    cam.apply_to_ti_camera(ti_camera)
    ti_scene.set_camera(ti_camera)
    ti_scene.ambient_light((0.2, 0.2, 0.2))
    ti_scene.point_light(pos=(10, 10, 10), color=(1, 1, 1))
    ti_scene.mesh(
        data.vertices,
        indices=data.indices,
        per_vertex_color=data.vertex_colors,
        two_sided=True
    )
    # Render collision geom hulls as wireframe mesh
    if settings.render_geoms and data.num_collision_faces[None] > 0:
        ti_scene.mesh(
            data.debug_geom_verts,
            indices=data.debug_geom_indices,
            per_vertex_color=data.debug_geom_colors,
            two_sided=True
        )
        # Render normal arrows as lines
        num_faces = data.num_collision_faces[None]
        ti_scene.lines(
            data.debug_normal_verts,
            per_vertex_color=data.debug_normal_colors,
            width=3.0,
            vertex_count=num_faces * 2
        )
    # Render contact points and normals
    num_contacts = data.num_contacts[None]
    if settings.debug_contacts and num_contacts > 0:
        # Contact points as particles (yellow)
        ti_scene.particles(
            data.debug_contact_points,
            radius=0.02,
            color=(1.0, 1.0, 0.0),
            index_count=num_contacts
        )
        # Contact normals as lines (cyan)
        ti_scene.lines(
            data.debug_contact_normals,
            width=4.0,
            color=(0.0, 1.0, 1.0),
            vertex_count=num_contacts * 2
        )
    canvas.scene(ti_scene)
    if is_enabled_benchmark():
        ti.sync()

# def create_demo_scene():
#     """Create demo scene - dragon inside a room"""
#     room_size = 10
#     wall_thickness = 0.5
#     half = room_size / 2

#     # Floor
#     scene.add_box(
#         center=(0, -wall_thickness / 2, 0),
#         size=(room_size, wall_thickness, room_size),
#         color=(0.4, 0.4, 0.4)
#     )

#     # Ceiling
#     scene.add_box(
#         center=(0, room_size - wall_thickness / 2, 0),
#         size=(room_size, wall_thickness, room_size),
#         color=(0.5, 0.5, 0.5)
#     )

#     # Back wall (far from camera)
#     scene.add_box(
#         center=(0, half, -half - wall_thickness / 2),
#         size=(room_size, room_size, wall_thickness),
#         color=(0.6, 0.6, 0.7)
#     )

#     # Left wall
#     scene.add_box(
#         center=(-half - wall_thickness / 2, half, 0),
#         size=(wall_thickness, room_size, room_size),
#         color=(0.7, 0.5, 0.5)
#     )

#     # Right wall
#     scene.add_box(
#         center=(half + wall_thickness / 2, half, 0),
#         size=(wall_thickness, room_size, room_size),
#         color=(0.5, 0.7, 0.5)
#     )

#     # Front wall is OPEN (no wall) so we can see inside

#     # Dragon inside the room
#     scene.add_mesh_from_obj(
#         "./models/dragon_small.obj",
#         center=(0, 2.5, 0),
#         size=8.0,
#         color=(0.8, 0.3, 0.3),
#         rotation=(0, 180, 0)
#     )

def create_demo_scene():
    """Create demo scene - spheres dropping onto a ground plane"""
    # Large ground plane (static)
    scene.add_box(
        center=(0, -0.25, 0),
        size=(20, 0.5, 20),
        color=(0.3, 0.3, 0.35),
        is_static=True
    )

    # Spheres at various heights (dynamic, will drop with physics)
    scene.add_sphere(center=(-2, 3, 0), radius=0.5, color=(0.9, 0.2, 0.2))   # Red
    scene.add_sphere(center=(0, 5, 0), radius=0.7, color=(0.2, 0.9, 0.2))    # Green
    scene.add_sphere(center=(2, 4, 1), radius=0.5, color=(0.2, 0.2, 0.9))    # Blue
    scene.add_sphere(center=(-1, 6, -1), radius=0.6, color=(0.9, 0.9, 0.2))  # Yellow
    scene.add_sphere(center=(1.5, 7, 0.5), radius=0.4, color=(0.9, 0.2, 0.9)) # Magenta
    scene.add_sphere(center=(0, 8, 0), radius=0.8, color=(0.2, 0.9, 0.9)) 
    
    scene.add_mesh_from_obj(
    "./models/cylinder.obj",
    center=(-5, 2.5, 0),
    size=2.0,
    color=(0.8, 0.3, 0.3),
    rotation=(45, 180, 45)
    )

    # A box to show mixed shapes (dynamic)
    scene.add_box(
        center=(-3, 2, 2),
        size=(1, 1, 1),
        color=(0.8, 0.5, 0.2)
    )


@benchmark
def render_frame(camera, frame, window, canvas, ti_scene, ti_camera, use_raytracing):
    dt = 1.0 / 60.0
    camera.handle_input(window, dt)
    run_physics(dt)

    rays = 0
    if use_raytracing:
        if settings.debug_bvh:
            run_debug_bvh(camera)
            rays = WIDTH * HEIGHT
        else:
            run_build_bvh()
            run_raytrace(camera, frame, settings)
            rays = WIDTH * HEIGHT * settings.samples_per_pixel * settings.max_bounces
        canvas.set_image(data.pixels)
    else:
        run_rasterize(ti_scene, ti_camera, canvas, camera)

    if is_enabled_benchmark():
        ti.sync()

    return rays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raytrace', action='store_true', help='Use ray tracing')
    args = parser.parse_args()

    use_raytracing = args.raytrace

    create_demo_scene()

    # Compute local-space vertices for physics (once, before simulation)
    compute_local_vertices(data.num_bodies[None])

    print(f"Scene: {data.num_vertices[None]} vertices, {data.num_triangles[None]} triangles")
    print(f"Physics: {data.num_bodies[None]} bodies, {data.num_geoms[None]} geoms")
    print(f"Rendering: {'Ray Tracing' if use_raytracing else 'Rasterization'}")

    window = ti.ui.Window("Taichi Scene", (WIDTH, HEIGHT), vsync=False)
    canvas = window.get_canvas()
    ti_scene = window.get_scene()
    ti_camera = ti.ui.Camera()

    camera = Camera(position=(0, 5, 15), yaw=-90, pitch=-15)
    frame = 0
    last_time = time.perf_counter()
    fps_smooth = 0.0
    mrays_smooth = 0.0
    ema_alpha = 0.1  # Smoothing factor (lower = smoother, higher = more responsive)

    while window.running:
        frame_start = time.perf_counter()

        rays = render_frame(camera, frame, window, canvas, ti_scene, ti_camera, use_raytracing)

        now = time.perf_counter()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps = 1.0 / dt
            mrays = rays * fps / 1e6
            # Exponential moving average for smooth display
            fps_smooth = ema_alpha * fps + (1 - ema_alpha) * fps_smooth
            mrays_smooth = ema_alpha * mrays + (1 - ema_alpha) * mrays_smooth

        # GUI panel
        gui = window.get_gui()
        gui.begin("Settings", 0.02, 0.02, 0.3, 0.35)
        gui.text(f"FPS: {fps_smooth:.1f}  Mrays/s: {mrays_smooth:.1f}")
        gui.text(f"Contacts: {data.num_contacts[None]}")
        settings.target_fps = gui.slider_int("Target FPS", settings.target_fps, 0, 120)
        settings.max_bounces = gui.slider_int("Bounces", settings.max_bounces, 1, 16)
        settings.samples_per_pixel = gui.slider_int("Samples", settings.samples_per_pixel, 1, 64)
        settings.sky_intensity = gui.slider_float("Sky", settings.sky_intensity, 0.0, 3.0)
        settings.debug_bvh = gui.checkbox("Debug BVH", settings.debug_bvh)
        settings.highlight_contacts = gui.checkbox("Highlight Contacts", settings.highlight_contacts)
        settings.render_geoms = gui.checkbox("Render Geoms", settings.render_geoms)
        settings.debug_contacts = gui.checkbox("Debug Contacts", settings.debug_contacts)
        gui.end()

        frame += 1
        window.show()

        # FPS limiter
        if settings.target_fps > 0:
            target_frame_time = 1.0 / settings.target_fps
            elapsed = time.perf_counter() - frame_start
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)


if __name__ == '__main__':
    main()
