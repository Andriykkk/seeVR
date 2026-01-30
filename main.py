import taichi as ti
import math
import argparse
from benchmark import benchmark, is_enabled_benchmark

ti.init(arch=ti.gpu)

# Constants
MAX_TRIANGLES = 500000
MAX_VERTICES = 500000
MAX_BVH_NODES = MAX_TRIANGLES * 2  # BVH needs at most 2N-1 nodes for N triangles
WIDTH, HEIGHT = 800, 600

# BVH Node structure (32 bytes, compact layout)
# leftFirst: if triCount==0 -> left child index, else -> first triangle index
BVHNode = ti.types.struct(
    aabb_min=ti.types.vector(3, ti.f32),  # 12 bytes
    aabb_max=ti.types.vector(3, ti.f32),  # 12 bytes
    left_first=ti.u32,                     # 4 bytes
    tri_count=ti.u32,                      # 4 bytes
)

# Path tracing settings (can be changed at runtime via GUI)
class Settings:
    def __init__(self):
        self.max_bounces = 2
        self.samples_per_pixel = 2
        self.sky_intensity = 1.0
        self.debug_bvh = False

settings = Settings()


class Scene:
    """Manages all scene objects as triangle meshes"""

    def __init__(self):
        # All geometry stored as flat arrays (works for both rasterization and ray tracing)
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)
        self.indices = ti.field(dtype=ti.i32, shape=MAX_TRIANGLES * 3)  # Every 3 = one triangle
        self.vertex_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)
        self.num_vertices = ti.field(dtype=ti.i32, shape=())
        self.num_triangles = ti.field(dtype=ti.i32, shape=())

        # For animation - store velocity per vertex
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=MAX_VERTICES)

        # Object tracking for physics (start_vertex, num_vertices, center)
        self.object_starts = []  # List of (start_vert, num_verts) tuples

        # Pixel buffer for ray tracing
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

        # BVH
        self.bvh_nodes = BVHNode.field(shape=MAX_BVH_NODES)
        self.bvh_prim_indices = ti.field(dtype=ti.i32, shape=MAX_TRIANGLES)  # Reordered triangle indices
        self.tri_centroids = ti.Vector.field(3, dtype=ti.f32, shape=MAX_TRIANGLES)  # Triangle centroids
        self.num_bvh_nodes = ti.field(dtype=ti.i32, shape=())
        self.bvh_stack = ti.field(dtype=ti.i32, shape=MAX_BVH_NODES)  # Stack for iterative build
        self.traverse_stack = ti.field(dtype=ti.i32, shape=(WIDTH, HEIGHT, 64))  # Per-pixel traversal stack
        self.bvh_built = False

        # Initialize counts
        self.num_vertices[None] = 0
        self.num_triangles[None] = 0

        self._vertex_count = 0
        self._triangle_count = 0

    @benchmark
    def add_sphere(self, center, radius, color=(1.0, 1.0, 1.0), velocity=(0, 0, 0), segments=16):
        """Add a UV sphere as triangles"""
        cx, cy, cz = center
        start_vertex = self._vertex_count
        verts = []

        # Generate vertices
        for lat in range(segments + 1):
            theta = lat * math.pi / segments
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)

            for lon in range(segments + 1):
                phi = lon * 2 * math.pi / segments
                sin_phi = math.sin(phi)
                cos_phi = math.cos(phi)

                x = cx + radius * cos_phi * sin_theta
                y = cy + radius * cos_theta
                z = cz + radius * sin_phi * sin_theta
                verts.append((x, y, z))

        # Add vertices
        for v in verts:
            self.vertices[self._vertex_count] = v
            self.velocities[self._vertex_count] = velocity
            self._vertex_count += 1

        # Generate triangles
        for lat in range(segments):
            for lon in range(segments):
                current = start_vertex + lat * (segments + 1) + lon
                next_row = current + segments + 1

                idx = self._triangle_count * 3
                self.indices[idx] = current
                self.indices[idx + 1] = next_row
                self.indices[idx + 2] = current + 1
                self._triangle_count += 1

                idx = self._triangle_count * 3
                self.indices[idx] = current + 1
                self.indices[idx + 1] = next_row
                self.indices[idx + 2] = next_row + 1
                self._triangle_count += 1

        # Set vertex colors
        for i in range(start_vertex, self._vertex_count):
            self.vertex_colors[i] = color

        self.num_vertices[None] = self._vertex_count
        self.num_triangles[None] = self._triangle_count

        self.object_starts.append((start_vertex, self._vertex_count - start_vertex))
        return len(self.object_starts) - 1

    @benchmark
    def add_box(self, center, size, color=(1.0, 1.0, 1.0), velocity=(0, 0, 0)):
        """Add a box as 12 triangles"""
        cx, cy, cz = center
        sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
        start_vertex = self._vertex_count

        verts = [
            (cx - sx, cy - sy, cz - sz),
            (cx + sx, cy - sy, cz - sz),
            (cx + sx, cy + sy, cz - sz),
            (cx - sx, cy + sy, cz - sz),
            (cx - sx, cy - sy, cz + sz),
            (cx + sx, cy - sy, cz + sz),
            (cx + sx, cy + sy, cz + sz),
            (cx - sx, cy + sy, cz + sz),
        ]

        faces = [
            (0, 1, 2), (0, 2, 3),
            (5, 4, 7), (5, 7, 6),
            (4, 0, 3), (4, 3, 7),
            (1, 5, 6), (1, 6, 2),
            (3, 2, 6), (3, 6, 7),
            (4, 5, 1), (4, 1, 0),
        ]

        for v in verts:
            self.vertices[self._vertex_count] = v
            self.velocities[self._vertex_count] = velocity
            self._vertex_count += 1

        for f in faces:
            idx = self._triangle_count * 3
            self.indices[idx] = start_vertex + f[0]
            self.indices[idx + 1] = start_vertex + f[1]
            self.indices[idx + 2] = start_vertex + f[2]
            self._triangle_count += 1

        # Set vertex colors
        for i in range(start_vertex, self._vertex_count):
            self.vertex_colors[i] = color

        self.num_vertices[None] = self._vertex_count
        self.num_triangles[None] = self._triangle_count

        self.object_starts.append((start_vertex, self._vertex_count - start_vertex))
        return len(self.object_starts) - 1

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
        start_vertex = self._vertex_count

        verts = [
            (cx - s * bx - s * t2x, cy - s * by - s * t2y, cz - s * bz - s * t2z),
            (cx + s * bx - s * t2x, cy + s * by - s * t2y, cz + s * bz - s * t2z),
            (cx + s * bx + s * t2x, cy + s * by + s * t2y, cz + s * bz + s * t2z),
            (cx - s * bx + s * t2x, cy - s * by + s * t2y, cz - s * bz + s * t2z),
        ]

        for v in verts:
            self.vertices[self._vertex_count] = v
            self.velocities[self._vertex_count] = (0, 0, 0)
            self._vertex_count += 1

        idx = self._triangle_count * 3
        self.indices[idx] = start_vertex
        self.indices[idx + 1] = start_vertex + 1
        self.indices[idx + 2] = start_vertex + 2
        self._triangle_count += 1

        idx = self._triangle_count * 3
        self.indices[idx] = start_vertex
        self.indices[idx + 1] = start_vertex + 2
        self.indices[idx + 2] = start_vertex + 3
        self._triangle_count += 1

        # Set vertex colors
        for i in range(start_vertex, self._vertex_count):
            self.vertex_colors[i] = color

        self.num_vertices[None] = self._vertex_count
        self.num_triangles[None] = self._triangle_count

    @benchmark
    def add_mesh_from_obj(self, filename, center=(0, 0, 0), size=1.0, rotation=(0, 0, 0), color=(1.0, 1.0, 1.0), velocity=(0, 0, 0)):
        """Load mesh from OBJ file, scaled to fit within given size

        Args:
            rotation: (rx, ry, rz) in degrees - applied in Y, X, Z order
        """
        raw_verts = []
        faces = []
        start_vertex = self._vertex_count

        # Read raw vertices and faces
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'v':
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    raw_verts.append((x, y, z))
                elif parts[0] == 'f':
                    indices = []
                    for p in parts[1:]:
                        idx = int(p.split('/')[0]) - 1
                        indices.append(idx)
                    for i in range(1, len(indices) - 1):
                        faces.append((indices[0], indices[i], indices[i + 1]))

        if not raw_verts:
            return -1

        # Calculate bounding box
        min_x = min(v[0] for v in raw_verts)
        max_x = max(v[0] for v in raw_verts)
        min_y = min(v[1] for v in raw_verts)
        max_y = max(v[1] for v in raw_verts)
        min_z = min(v[2] for v in raw_verts)
        max_z = max(v[2] for v in raw_verts)

        # Mesh center and dimensions
        mesh_center = ((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
        max_extent = max(max_x - min_x, max_y - min_y, max_z - min_z)

        # Scale to fit within size
        scale = size / max_extent if max_extent > 0 else 1.0

        # Precompute rotation (Y * X * Z order)
        rx, ry, rz = math.radians(rotation[0]), math.radians(rotation[1]), math.radians(rotation[2])
        cos_x, sin_x = math.cos(rx), math.sin(rx)
        cos_y, sin_y = math.cos(ry), math.sin(ry)
        cos_z, sin_z = math.cos(rz), math.sin(rz)

        # Transform: center at origin, scale, rotate, move to target center
        for x, y, z in raw_verts:
            # Center and scale
            vx = (x - mesh_center[0]) * scale
            vy = (y - mesh_center[1]) * scale
            vz = (z - mesh_center[2]) * scale

            # Rotate Y
            tmp_x = cos_y * vx + sin_y * vz
            tmp_z = -sin_y * vx + cos_y * vz
            vx, vz = tmp_x, tmp_z

            # Rotate X
            tmp_y = cos_x * vy - sin_x * vz
            tmp_z = sin_x * vy + cos_x * vz
            vy, vz = tmp_y, tmp_z

            # Rotate Z
            tmp_x = cos_z * vx - sin_z * vy
            tmp_y = sin_z * vx + cos_z * vy
            vx, vy = tmp_x, tmp_y

            # Translate to center
            vx += center[0]
            vy += center[1]
            vz += center[2]

            self.vertices[self._vertex_count] = (vx, vy, vz)
            self.velocities[self._vertex_count] = velocity
            self._vertex_count += 1

        for f in faces:
            idx = self._triangle_count * 3
            self.indices[idx] = start_vertex + f[0]
            self.indices[idx + 1] = start_vertex + f[1]
            self.indices[idx + 2] = start_vertex + f[2]
            self._triangle_count += 1

        # Set vertex colors
        for i in range(start_vertex, self._vertex_count):
            self.vertex_colors[i] = color

        self.num_vertices[None] = self._vertex_count
        self.num_triangles[None] = self._triangle_count

        self.object_starts.append((start_vertex, len(raw_verts)))
        print(f"Loaded {filename}: {len(raw_verts)} vertices, {len(faces)} triangles")
        return len(self.object_starts) - 1

    def clear(self):
        """Clear all objects"""
        self._vertex_count = 0
        self._triangle_count = 0
        self.num_vertices[None] = 0
        self.num_triangles[None] = 0
        self.object_starts.clear()
        self.bvh_built = False

    @benchmark
    def build_bvh(self):
        """Build BVH acceleration structure"""
        n = self._triangle_count
        if n == 0:
            return

        # GPU: compute centroids and init prim_indices
        bvh_init_centroids(n)
        ti.sync()  # Ensure GPU is done before CPU reads

        # CPU: build tree (deterministic)
        self._build_bvh_cpu(n)
        self.bvh_built = True

    def _build_bvh_cpu(self, n):
        """Build BVH on CPU (deterministic)"""
        import numpy as np

        # Copy data to numpy for CPU processing
        centroids = self.tri_centroids.to_numpy()[:n]
        prim_indices = list(range(n))  # Local list for partitioning

        # Initialize root
        self.bvh_nodes[0].left_first = 0
        self.bvh_nodes[0].tri_count = n
        self.num_bvh_nodes[None] = 1

        # Stack for iterative build: (node_idx, first, count)
        stack = [(0, 0, n)]
        nodes_used = 1

        while stack:
            node_idx, first, count = stack.pop()

            # Calculate bounds
            aabb_min = np.array([1e30, 1e30, 1e30])
            aabb_max = np.array([-1e30, -1e30, -1e30])
            for i in range(first, first + count):
                tri_idx = prim_indices[i]
                c = centroids[tri_idx]
                # Get actual triangle vertices for bounds
                idx = tri_idx * 3
                for vi in range(3):
                    v = self.vertices[self.indices[idx + vi]]
                    aabb_min = np.minimum(aabb_min, [v[0], v[1], v[2]])
                    aabb_max = np.maximum(aabb_max, [v[0], v[1], v[2]])

            self.bvh_nodes[node_idx].aabb_min = aabb_min.tolist()
            self.bvh_nodes[node_idx].aabb_max = aabb_max.tolist()

            # Leaf if <= 10 triangles
            if count <= 10:
                self.bvh_nodes[node_idx].left_first = first
                self.bvh_nodes[node_idx].tri_count = count
                # Write prim_indices to GPU field
                for i in range(count):
                    self.bvh_prim_indices[first + i] = prim_indices[first + i]
                continue

            # Find split axis (longest extent)
            extent = aabb_max - aabb_min
            axis = int(np.argmax(extent))
            split_pos = aabb_min[axis] + extent[axis] * 0.5

            # Partition
            i, j = first, first + count - 1
            while i <= j:
                if centroids[prim_indices[i]][axis] < split_pos:
                    i += 1
                else:
                    prim_indices[i], prim_indices[j] = prim_indices[j], prim_indices[i]
                    j -= 1

            left_count = i - first
            if left_count == 0 or left_count == count:
                # Degenerate - keep as leaf
                self.bvh_nodes[node_idx].left_first = first
                self.bvh_nodes[node_idx].tri_count = count
                for i in range(count):
                    self.bvh_prim_indices[first + i] = prim_indices[first + i]
                continue

            # Create children
            left_idx = nodes_used
            right_idx = nodes_used + 1
            nodes_used += 2

            # Mark parent as interior (tri_count=0 means left_first is left child)
            self.bvh_nodes[node_idx].left_first = left_idx
            self.bvh_nodes[node_idx].tri_count = 0

            # Push children to stack
            stack.append((right_idx, i, count - left_count))
            stack.append((left_idx, first, left_count))

        self.num_bvh_nodes[None] = nodes_used

scene = Scene()


# BVH construction kernels
@ti.kernel
def bvh_init_centroids(n: ti.i32):
    """Initialize triangle centroids and prim indices (parallel)"""
    for i in range(n):
        idx = i * 3
        v0 = scene.vertices[scene.indices[idx]]
        v1 = scene.vertices[scene.indices[idx + 1]]
        v2 = scene.vertices[scene.indices[idx + 2]]
        scene.tri_centroids[i] = (v0 + v1 + v2) / 3.0
        scene.bvh_prim_indices[i] = i


@ti.func
def bvh_update_node_bounds(node_idx: ti.i32):
    """Calculate AABB bounds for a node"""
    aabb_min = ti.Vector([1e30, 1e30, 1e30])
    aabb_max = ti.Vector([-1e30, -1e30, -1e30])

    first = ti.cast(scene.bvh_nodes[node_idx].left_first, ti.i32)
    count = ti.cast(scene.bvh_nodes[node_idx].tri_count, ti.i32)

    for i in range(count):
        tri_idx = scene.bvh_prim_indices[first + i]
        idx = tri_idx * 3
        v0 = scene.vertices[scene.indices[idx]]
        v1 = scene.vertices[scene.indices[idx + 1]]
        v2 = scene.vertices[scene.indices[idx + 2]]

        aabb_min = ti.min(aabb_min, v0)
        aabb_min = ti.min(aabb_min, v1)
        aabb_min = ti.min(aabb_min, v2)
        aabb_max = ti.max(aabb_max, v0)
        aabb_max = ti.max(aabb_max, v1)
        aabb_max = ti.max(aabb_max, v2)

    scene.bvh_nodes[node_idx].aabb_min = aabb_min
    scene.bvh_nodes[node_idx].aabb_max = aabb_max


@ti.kernel
def bvh_build(n: ti.i32):
    """Build BVH iteratively using a stack (runs on GPU)"""
    # Initialize root node
    scene.bvh_nodes[0].left_first = ti.cast(0, ti.u32)
    scene.bvh_nodes[0].tri_count = ti.cast(n, ti.u32)
    scene.num_bvh_nodes[None] = 1

    bvh_update_node_bounds(0)

    # Stack-based iteration (replaces recursion)
    stack_ptr = 1
    scene.bvh_stack[0] = 0

    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = scene.bvh_stack[stack_ptr]

        tri_count = ti.cast(scene.bvh_nodes[node_idx].tri_count, ti.i32)

        # Leaf node - stop subdividing
        if tri_count <= 2:
            continue

        aabb_min = scene.bvh_nodes[node_idx].aabb_min
        aabb_max = scene.bvh_nodes[node_idx].aabb_max
        first_prim = ti.cast(scene.bvh_nodes[node_idx].left_first, ti.i32)

        # Find longest axis
        extent = aabb_max - aabb_min
        axis = 0
        if extent[1] > extent[0]:
            axis = 1
        if extent[2] > extent[axis]:
            axis = 2
        split_pos = aabb_min[axis] + extent[axis] * 0.5

        # In-place partition
        i = first_prim
        j = i + tri_count - 1
        while i <= j:
            if scene.tri_centroids[scene.bvh_prim_indices[i]][axis] < split_pos:
                i += 1
            else:
                tmp = scene.bvh_prim_indices[i]
                scene.bvh_prim_indices[i] = scene.bvh_prim_indices[j]
                scene.bvh_prim_indices[j] = tmp
                j -= 1

        # Check for degenerate split
        left_count = i - first_prim
        if left_count == 0 or left_count == tri_count:
            continue

        # Create child nodes
        left_idx = scene.num_bvh_nodes[None]
        right_idx = left_idx + 1
        scene.num_bvh_nodes[None] += 2

        # Left child
        scene.bvh_nodes[left_idx].left_first = ti.cast(first_prim, ti.u32)
        scene.bvh_nodes[left_idx].tri_count = ti.cast(left_count, ti.u32)

        # Right child
        scene.bvh_nodes[right_idx].left_first = ti.cast(i, ti.u32)
        scene.bvh_nodes[right_idx].tri_count = ti.cast(tri_count - left_count, ti.u32)

        # Mark parent as interior (tri_count=0 means left_first is left child)
        scene.bvh_nodes[node_idx].left_first = ti.cast(left_idx, ti.u32)
        scene.bvh_nodes[node_idx].tri_count = ti.cast(0, ti.u32)

        # Update bounds
        bvh_update_node_bounds(left_idx)
        bvh_update_node_bounds(right_idx)

        # Push children to stack
        scene.bvh_stack[stack_ptr] = left_idx
        stack_ptr += 1
        scene.bvh_stack[stack_ptr] = right_idx
        stack_ptr += 1


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


@ti.kernel
def update_physics(dt: ti.f32):
    for i in range(scene.num_vertices[None]):
        scene.vertices[i] += scene.velocities[i] * dt


@ti.func
def ray_triangle_intersect(ray_o, ray_d, v0, v1, v2):
    """Möller–Trumbore algorithm"""
    e1 = v1 - v0
    e2 = v2 - v0
    h = ray_d.cross(e2)
    a = e1.dot(h)
    t = -1.0

    if ti.abs(a) > 1e-8:
        f = 1.0 / a
        s = ray_o - v0
        u = f * s.dot(h)
        if 0.0 <= u <= 1.0:
            q = s.cross(e1)
            v = f * ray_d.dot(q)
            if v >= 0.0 and u + v <= 1.0:
                t = f * e2.dot(q)
                if t < 0.001:
                    t = -1.0
    return t


@ti.func
def get_triangle_normal(v0, v1, v2):
    e1 = v1 - v0
    e2 = v2 - v0
    return e1.cross(e2).normalized()


@ti.func
def intersect_aabb(ray_o, ray_d, bmin, bmax, closest_t):
    """Ray-AABB intersection test"""
    # Compute inverse direction to avoid division
    inv_d = 1.0 / ray_d

    tx1 = (bmin[0] - ray_o[0]) * inv_d[0]
    tx2 = (bmax[0] - ray_o[0]) * inv_d[0]
    tmin = ti.min(tx1, tx2)
    tmax = ti.max(tx1, tx2)

    ty1 = (bmin[1] - ray_o[1]) * inv_d[1]
    ty2 = (bmax[1] - ray_o[1]) * inv_d[1]
    tmin = ti.max(tmin, ti.min(ty1, ty2))
    tmax = ti.min(tmax, ti.max(ty1, ty2))

    tz1 = (bmin[2] - ray_o[2]) * inv_d[2]
    tz2 = (bmax[2] - ray_o[2]) * inv_d[2]
    tmin = ti.max(tmin, ti.min(tz1, tz2))
    tmax = ti.min(tmax, ti.max(tz1, tz2))

    return tmax >= tmin and tmin < closest_t and tmax > 0.0


@ti.func
def trace_bvh(ray_o, ray_d, px: ti.i32, py: ti.i32):
    """Trace ray through BVH with local stack"""
    closest_t = ti.f32(1e10)
    hit_normal = ti.Vector([0.0, 1.0, 0.0])
    hit_color = ti.Vector([0.0, 0.0, 0.0])
    hit = False

    # Local stack as register values
    stack = ti.Matrix([[0] * 32], dt=ti.i32)
    stack[0, 0] = 0  # Start with root
    stack_ptr = 1

    for _iter in range(1000):
        if stack_ptr <= 0:
            break

        stack_ptr -= 1
        node_idx = stack[0, stack_ptr]

        node_min = scene.bvh_nodes[node_idx].aabb_min
        node_max = scene.bvh_nodes[node_idx].aabb_max

        if not intersect_aabb(ray_o, ray_d, node_min, node_max, closest_t):
            continue

        tri_count = ti.cast(scene.bvh_nodes[node_idx].tri_count, ti.i32)

        if tri_count > 0:
            # Leaf - test triangles
            first_tri = ti.cast(scene.bvh_nodes[node_idx].left_first, ti.i32)
            for i in range(tri_count):
                tri_idx = scene.bvh_prim_indices[first_tri + i]
                idx = tri_idx * 3
                v0 = scene.vertices[scene.indices[idx]]
                v1 = scene.vertices[scene.indices[idx + 1]]
                v2 = scene.vertices[scene.indices[idx + 2]]

                t = ray_triangle_intersect(ray_o, ray_d, v0, v1, v2)
                if 0.001 < t < closest_t:
                    closest_t = t
                    hit_normal = get_triangle_normal(v0, v1, v2)
                    if hit_normal.dot(ray_d) > 0:
                        hit_normal = -hit_normal
                    hit_color = scene.vertex_colors[scene.indices[idx]]
                    hit = True
        else:
            # Interior - push children
            left = ti.cast(scene.bvh_nodes[node_idx].left_first, ti.i32)
            stack[0, stack_ptr] = left
            stack_ptr += 1
            stack[0, stack_ptr] = left + 1
            stack_ptr += 1

    return hit, closest_t, hit_normal, hit_color


@ti.func
def debug_trace_bvh(ray_o, ray_d, px: ti.i32, py: ti.i32):
    """Debug: trace BVH and return info about nodes visited"""
    closest_t = ti.f32(1e10)
    hit_node_idx = -1
    hit_depth = 0

    # Use per-pixel stack from field - store (node_idx, depth)
    stack_ptr = 0
    scene.traverse_stack[px, py, 0] = 0  # Start with root node
    stack_ptr = 1
    depth = 0

    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = scene.traverse_stack[px, py, stack_ptr]

        node_aabb_min = scene.bvh_nodes[node_idx].aabb_min
        node_aabb_max = scene.bvh_nodes[node_idx].aabb_max
        tri_count = ti.cast(scene.bvh_nodes[node_idx].tri_count, ti.i32)
        left_first = ti.cast(scene.bvh_nodes[node_idx].left_first, ti.i32)

        # Test AABB intersection
        if intersect_aabb(ray_o, ray_d, node_aabb_min, node_aabb_max, closest_t):
            if tri_count > 0:
                # Leaf node - record which leaf we hit
                # Use AABB center as hit point estimate
                center = (node_aabb_min + node_aabb_max) * 0.5
                t_approx = (center - ray_o).dot(ray_d.normalized())
                if t_approx > 0 and t_approx < closest_t:
                    closest_t = t_approx
                    hit_node_idx = node_idx
                    hit_depth = depth
            else:
                # Interior node - push children
                scene.traverse_stack[px, py, stack_ptr] = left_first
                stack_ptr += 1
                scene.traverse_stack[px, py, stack_ptr] = left_first + 1
                stack_ptr += 1
                depth += 1

    return hit_node_idx, hit_depth


@ti.kernel
def debug_render_bvh(cam_pos_x: ti.f32, cam_pos_y: ti.f32, cam_pos_z: ti.f32,
                     cam_dir_x: ti.f32, cam_dir_y: ti.f32, cam_dir_z: ti.f32,
                     cam_right_x: ti.f32, cam_right_y: ti.f32, cam_right_z: ti.f32,
                     cam_up_x: ti.f32, cam_up_y: ti.f32, cam_up_z: ti.f32):
    """Debug: render BVH leaf nodes with colors based on node index"""
    cam_pos = ti.Vector([cam_pos_x, cam_pos_y, cam_pos_z])
    cam_dir = ti.Vector([cam_dir_x, cam_dir_y, cam_dir_z])
    cam_right = ti.Vector([cam_right_x, cam_right_y, cam_right_z])
    cam_up = ti.Vector([cam_up_x, cam_up_y, cam_up_z])

    fov = 45.0
    aspect = ti.cast(WIDTH, ti.f32) / ti.cast(HEIGHT, ti.f32)
    fov_scale = ti.tan(fov * 0.5 * 3.14159 / 180.0)

    for i, j in scene.pixels:
        u = (2.0 * (ti.cast(i, ti.f32) + 0.5) / ti.cast(WIDTH, ti.f32) - 1.0) * aspect * fov_scale
        v = (2.0 * (ti.cast(j, ti.f32) + 0.5) / ti.cast(HEIGHT, ti.f32) - 1.0) * fov_scale

        ray_dir = (cam_dir + u * cam_right + v * cam_up).normalized()

        node_idx, depth = debug_trace_bvh(cam_pos, ray_dir, i, j)

        if node_idx >= 0:
            # Color based on node index (pseudo-random color per node)
            r = ti.cast((node_idx * 73) % 256, ti.f32) / 255.0
            g = ti.cast((node_idx * 137) % 256, ti.f32) / 255.0
            b = ti.cast((node_idx * 199) % 256, ti.f32) / 255.0
            scene.pixels[i, j] = ti.Vector([r, g, b])
        else:
            # Sky
            t = 0.5 * (ray_dir[1] + 1.0)
            scene.pixels[i, j] = (1.0 - t) * ti.Vector([1.0, 1.0, 1.0]) + t * ti.Vector([0.5, 0.7, 1.0])


def run_debug_bvh(cam):
    debug_render_bvh(
        cam.pos[0], cam.pos[1], cam.pos[2],
        cam.direction[0], cam.direction[1], cam.direction[2],
        cam.right[0], cam.right[1], cam.right[2],
        cam.up[0], cam.up[1], cam.up[2]
    )


# Random number generation (PCG-based)
@ti.func
def rand_pcg(seed: ti.u32) -> ti.u32:
    state = seed * ti.u32(747796405) + ti.u32(2891336453)
    word = ((state >> ((state >> 28) + 4)) ^ state) * ti.u32(277803737)
    return (word >> 22) ^ word


@ti.func
def rand_float(seed: ti.u32) -> ti.f32:
    return ti.cast(rand_pcg(seed), ti.f32) / 4294967295.0


# Sample random direction on hemisphere (cosine-weighted for diffuse)
@ti.func
def sample_hemisphere_cosine(normal, seed1: ti.u32, seed2: ti.u32):
    r1 = rand_float(seed1)
    r2 = rand_float(seed2)

    # Cosine-weighted hemisphere sampling
    phi = 2.0 * 3.14159 * r1
    cos_theta = ti.sqrt(r2)
    sin_theta = ti.sqrt(1.0 - r2)

    # Create local coordinate system
    up = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(normal[1]) > 0.99:
        up = ti.Vector([1.0, 0.0, 0.0])
    tangent = up.cross(normal).normalized()
    bitangent = normal.cross(tangent)

    # Convert to world space
    dir_local = ti.Vector([
        sin_theta * ti.cos(phi),
        cos_theta,
        sin_theta * ti.sin(phi)
    ])
    return (tangent * dir_local[0] + normal * dir_local[1] + bitangent * dir_local[2]).normalized()

# Sky color (simple gradient)
@ti.func
def sky_color(ray_d, intensity: ti.f32):
    t = 0.5 * (ray_d[1] + 1.0)
    base = (1.0 - t) * ti.Vector([1.0, 1.0, 1.0]) + t * ti.Vector([0.5, 0.7, 1.0])
    return base * intensity


@ti.kernel
def raytrace(cam_pos_x: ti.f32, cam_pos_y: ti.f32, cam_pos_z: ti.f32,
             cam_dir_x: ti.f32, cam_dir_y: ti.f32, cam_dir_z: ti.f32,
             cam_right_x: ti.f32, cam_right_y: ti.f32, cam_right_z: ti.f32,
             cam_up_x: ti.f32, cam_up_y: ti.f32, cam_up_z: ti.f32,
             frame: ti.i32, max_bounces: ti.i32, samples_per_pixel: ti.i32,
             sky_intensity: ti.f32):
    cam_pos = ti.Vector([cam_pos_x, cam_pos_y, cam_pos_z])
    cam_dir = ti.Vector([cam_dir_x, cam_dir_y, cam_dir_z])
    cam_right = ti.Vector([cam_right_x, cam_right_y, cam_right_z])
    cam_up = ti.Vector([cam_up_x, cam_up_y, cam_up_z])

    fov = 45.0
    aspect = ti.cast(WIDTH, ti.f32) / ti.cast(HEIGHT, ti.f32)
    fov_scale = ti.tan(fov * 0.5 * 3.14159 / 180.0)

    for i, j in scene.pixels:
        pixel_color = ti.Vector([0.0, 0.0, 0.0])

        # Multiple samples per pixel
        for sample in range(samples_per_pixel):
            # Random seed based on pixel, frame, and sample
            seed = ti.cast(i + j * WIDTH + frame * WIDTH * HEIGHT + sample * 12345, ti.u32)

            # Jitter ray for anti-aliasing
            jitter_x = rand_float(seed) - 0.5
            jitter_y = rand_float(seed + 1) - 0.5

            u = (2.0 * (ti.cast(i, ti.f32) + 0.5 + jitter_x) / ti.cast(WIDTH, ti.f32) - 1.0) * aspect * fov_scale
            v = (2.0 * (ti.cast(j, ti.f32) + 0.5 + jitter_y) / ti.cast(HEIGHT, ti.f32) - 1.0) * fov_scale

            ray_pos = cam_pos
            ray_dir = (cam_dir + u * cam_right + v * cam_up).normalized()

            # Path tracing with multiple bounces
            throughput = ti.Vector([1.0, 1.0, 1.0])
            color = ti.Vector([0.0, 0.0, 0.0])

            for bounce in range(max_bounces):
                hit, t, normal, albedo = trace_bvh(ray_pos, ray_dir, i, j)

                if not hit:
                    # Hit sky - add sky contribution
                    color += throughput * sky_color(ray_dir, sky_intensity)
                    break

                # Move to hit point
                hit_point = ray_pos + ray_dir * t

                # Simple diffuse BRDF: throughput *= albedo * cos_theta / pdf
                # For cosine-weighted sampling, cos_theta/pdf = 1, so just multiply by albedo
                throughput *= albedo

                # Russian roulette for early termination (after 2 bounces)
                if bounce > 1:
                    p = ti.max(throughput[0], ti.max(throughput[1], throughput[2]))
                    if rand_float(seed + ti.cast(bounce, ti.u32) * 7) > p:
                        break
                    throughput /= p

                # Generate new random direction for next bounce
                seed = rand_pcg(seed)
                ray_dir = sample_hemisphere_cosine(normal, seed, seed + 1)
                ray_pos = hit_point + normal * 0.001  # Offset to avoid self-intersection

            pixel_color += color

        pixel_color /= ti.cast(samples_per_pixel, ti.f32)
        scene.pixels[i, j] = pixel_color


# Benchmarked wrappers for kernels (ti.sync() needed for accurate GPU timing)
@benchmark
def run_physics(dt):
    update_physics(dt)
    if is_enabled_benchmark():
        ti.sync()


@benchmark
def run_raytrace(cam, frame):
    raytrace(
        cam.pos[0], cam.pos[1], cam.pos[2],
        cam.direction[0], cam.direction[1], cam.direction[2],
        cam.right[0], cam.right[1], cam.right[2],
        cam.up[0], cam.up[1], cam.up[2],
        frame,
        settings.max_bounces,
        settings.samples_per_pixel,
        settings.sky_intensity
    )
    if is_enabled_benchmark():
        ti.sync()


@benchmark
def run_rasterize(ti_scene, ti_camera, canvas, cam):
    cam.apply_to_ti_camera(ti_camera)
    ti_scene.set_camera(ti_camera)
    ti_scene.ambient_light((0.2, 0.2, 0.2))
    ti_scene.point_light(pos=(10, 10, 10), color=(1, 1, 1))
    ti_scene.mesh(
        scene.vertices,
        indices=scene.indices,
        per_vertex_color=scene.vertex_colors,
        two_sided=True
    )
    canvas.scene(ti_scene)
    if is_enabled_benchmark():
        ti.sync()


def create_demo_scene():
    """Create demo scene - dragon inside a room"""
    room_size = 10
    wall_thickness = 0.5
    half = room_size / 2

    # Floor
    scene.add_box(
        center=(0, -wall_thickness / 2, 0),
        size=(room_size, wall_thickness, room_size),
        color=(0.4, 0.4, 0.4)
    )

    # Ceiling
    scene.add_box(
        center=(0, room_size - wall_thickness / 2, 0),
        size=(room_size, wall_thickness, room_size),
        color=(0.5, 0.5, 0.5)
    )

    # Back wall (far from camera)
    scene.add_box(
        center=(0, half, -half - wall_thickness / 2),
        size=(room_size, room_size, wall_thickness),
        color=(0.6, 0.6, 0.7)
    )

    # Left wall
    scene.add_box(
        center=(-half - wall_thickness / 2, half, 0),
        size=(wall_thickness, room_size, room_size),
        color=(0.7, 0.5, 0.5)
    )

    # Right wall
    scene.add_box(
        center=(half + wall_thickness / 2, half, 0),
        size=(wall_thickness, room_size, room_size),
        color=(0.5, 0.7, 0.5)
    )

    # Front wall is OPEN (no wall) so we can see inside

    # Dragon inside the room
    scene.add_mesh_from_obj(
        "./models/dragon_smallest.obj",
        center=(0, 2.5, 0),
        size=8.0,
        color=(0.8, 0.3, 0.3),
        rotation=(0, 180, 0)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raytrace', action='store_true', help='Use ray tracing')
    args = parser.parse_args()

    use_raytracing = args.raytrace

    create_demo_scene()

    print(f"Scene: {scene._vertex_count} vertices, {scene._triangle_count} triangles")
    print(f"Rendering: {'Ray Tracing' if use_raytracing else 'Rasterization'}")

    window = ti.ui.Window("Taichi Scene", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()
    ti_scene = window.get_scene()
    ti_camera = ti.ui.Camera()

    camera = Camera(position=(0, 5, 15), yaw=-90, pitch=-15)
    frame = 0

    scene.build_bvh()
    while window.running:
        dt = 1.0 / 60.0

        camera.handle_input(window, dt)
        run_physics(dt)

        if use_raytracing:
            if settings.debug_bvh:
                run_debug_bvh(camera)
            else:
                run_raytrace(camera, frame)
            canvas.set_image(scene.pixels)
        else:
            run_rasterize(ti_scene, ti_camera, canvas, camera)

        # GUI panel
        gui = window.get_gui()
        gui.begin("Settings", 0.02, 0.02, 0.3, 0.3)
        settings.max_bounces = gui.slider_int("Bounces", settings.max_bounces, 1, 16)
        settings.samples_per_pixel = gui.slider_int("Samples", settings.samples_per_pixel, 1, 64)
        settings.sky_intensity = gui.slider_float("Sky", settings.sky_intensity, 0.0, 3.0)
        settings.debug_bvh = gui.checkbox("Debug BVH", settings.debug_bvh)
        gui.end()

        frame += 1
        window.show()


if __name__ == '__main__':
    main()
