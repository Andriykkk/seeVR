import numpy as np
import time

# --- Create sphere hull vertices ---
def make_sphere(n_segments=32):
    """Generate sphere hull vertices like the engine does."""
    verts = []
    for i in range(n_segments):
        theta = 2 * np.pi * i / n_segments
        for j in range(n_segments):
            phi = np.pi * j / (n_segments - 1)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            verts.append([x, y, z])
    return np.array(verts)

# --- Brute force support ---
def brute_force_support(verts, direction):
    dots = verts @ direction
    idx = np.argmax(dots)
    return idx, verts[idx]

# --- 1D Lookup ---
class Lookup1D:
    def __init__(self, verts, n_bins, n_axes):
        self.verts = verts
        self.n_bins = n_bins
        self.n_axes = n_axes
        self.lookups = []

        # Create n_axes evenly spaced axes
        axes = []
        if n_axes == 1:
            axes = [np.array([0, 0, 1])]  # Z axis
        elif n_axes == 2:
            axes = [np.array([0, 0, 1]), np.array([1, 0, 0])]
        elif n_axes == 3:
            axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        elif n_axes == 4:
            axes = [
                np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                np.array([1, 1, 1]) / np.sqrt(3)
            ]
        elif n_axes == 6:
            axes = [
                np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                np.array([1, 1, 0]) / np.sqrt(2),
                np.array([1, 0, 1]) / np.sqrt(2),
                np.array([0, 1, 1]) / np.sqrt(2),
            ]
        else:
            # Random axes
            rng = np.random.RandomState(42)
            for _ in range(n_axes):
                a = rng.randn(3)
                axes.append(a / np.linalg.norm(a))

        self.axes = axes

        for axis in axes:
            # Create two perpendicular vectors to this axis
            axis = axis.astype(float)
            if abs(axis[0]) < 0.9:
                perp1 = np.cross(axis, np.array([1.0, 0, 0]))
            else:
                perp1 = np.cross(axis, np.array([0, 1.0, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(axis, perp1)

            lookup = []
            for i in range(n_bins):
                angle = 2 * np.pi * i / n_bins
                d = np.cos(angle) * perp1 + np.sin(angle) * perp2
                dots = verts @ d
                best_idx = np.argmax(dots)
                lookup.append(best_idx)
            self.lookups.append(lookup)

        self.perps = []
        for axis in axes:
            axis = axis.astype(float)
            if abs(axis[0]) < 0.9:
                perp1 = np.cross(axis, np.array([1.0, 0, 0]))
            else:
                perp1 = np.cross(axis, np.array([0, 1.0, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(axis, perp1)
            self.perps.append((perp1, perp2))

    def query(self, direction):
        best_dot = -1e10
        best_vert = None
        reads = 0

        for axis_idx in range(self.n_axes):
            perp1, perp2 = self.perps[axis_idx]
            # Project direction onto the plane perpendicular to this axis
            d1 = np.dot(direction, perp1)
            d2 = np.dot(direction, perp2)
            angle = np.arctan2(d2, d1)
            if angle < 0:
                angle += 2 * np.pi
            bin_f = angle / (2 * np.pi) * self.n_bins
            bin_lo = int(bin_f) % self.n_bins
            bin_hi = (bin_lo + 1) % self.n_bins

            for b in [bin_lo, bin_hi]:
                idx = self.lookups[axis_idx][b]
                d = np.dot(self.verts[idx], direction)
                reads += 1
                if d > best_dot:
                    best_dot = d
                    best_vert = self.verts[idx]

        return best_vert, best_dot, reads

    def memory_bytes(self):
        return self.n_axes * self.n_bins * 4  # uint32 per bin


# --- 2D Grid Lookup ---
class Lookup2D:
    def __init__(self, verts, grid_size):
        self.verts = verts
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)

        for i in range(grid_size):
            theta = (i + 0.5) / grid_size * 2 * np.pi - np.pi
            for j in range(grid_size):
                phi = (j + 0.5) / grid_size * np.pi
                d = np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ])
                dots = verts @ d
                self.grid[i, j] = np.argmax(dots)

    def query(self, direction):
        d = direction / np.linalg.norm(direction)
        theta = np.arctan2(d[1], d[0])  # -pi to pi
        phi = np.arccos(np.clip(d[2], -1, 1))  # 0 to pi

        gi = (theta + np.pi) / (2 * np.pi) * self.grid_size
        gj = phi / np.pi * self.grid_size

        i_lo = int(gi) % self.grid_size
        i_hi = (i_lo + 1) % self.grid_size
        j_lo = int(np.clip(gj, 0, self.grid_size - 1))
        j_hi = min(j_lo + 1, self.grid_size - 1)

        best_dot = -1e10
        best_vert = None
        reads = 0

        for i in [i_lo, i_hi]:
            for j in [j_lo, j_hi]:
                idx = self.grid[i, j]
                dot_val = np.dot(self.verts[idx], direction)
                reads += 1
                if dot_val > best_dot:
                    best_dot = dot_val
                    best_vert = self.verts[idx]

        return best_vert, best_dot, reads

    def memory_bytes(self):
        return self.grid_size * self.grid_size * 4  # uint32 per cell


# --- Test ---
def test_accuracy(verts, lookup, name, test_dirs):
    errors = []
    total_reads = 0
    for d in test_dirs:
        _, true_dot = brute_force_support(verts, d)
        _, lookup_dot, reads = lookup.query(d)
        total_reads += reads
        errors.append(true_dot - lookup_dot)

    errors = np.array(errors)
    avg_reads = total_reads / len(test_dirs)
    n_exact = np.sum(errors < 1e-10)
    print(f"{name:<35} mem={lookup.memory_bytes():>8} bytes  "
          f"reads={avg_reads:.1f}  "
          f"exact={n_exact}/{len(test_dirs)}  "
          f"avg_err={np.mean(errors):.6f}  "
          f"max_err={np.max(errors):.6f}")


def main():
    print("=== Support Function Lookup Comparison ===\n")

    # Create sphere with 512 verts (like 128*4 segments → 32*32 grid = 1024 actually)
    for n_seg in [16, 32]:
        verts = make_sphere(n_seg)
        print(f"Sphere: {len(verts)} hull vertices (segments={n_seg})")

        # Random test directions
        rng = np.random.RandomState(123)
        test_dirs = rng.randn(10000, 3)
        test_dirs /= np.linalg.norm(test_dirs, axis=1, keepdims=True)

        print(f"Testing {len(test_dirs)} random directions\n")

        # 1D lookups with varying axes
        for n_axes in [1, 2, 3, 4, 6]:
            for n_bins in [32, 64, 128]:
                lk = Lookup1D(verts, n_bins, n_axes)
                test_accuracy(verts, lk, f"1D  axes={n_axes}  bins={n_bins}", test_dirs)
            print()

        # 2D grids with varying sizes
        for grid_size in [8, 16, 32, 64]:
            lk = Lookup2D(verts, grid_size)
            test_accuracy(verts, lk, f"2D  grid={grid_size}x{grid_size}", test_dirs)
        print()

        # Speed comparison
        print("Speed comparison (100k queries):")
        speed_dirs = rng.randn(100000, 3)
        speed_dirs /= np.linalg.norm(speed_dirs, axis=1, keepdims=True)

        # Brute force
        t0 = time.time()
        for d in speed_dirs:
            brute_force_support(verts, d)
        t_brute = time.time() - t0
        print(f"  Brute force ({len(verts)} verts):  {t_brute:.3f}s")

        # Best 1D
        lk1d = Lookup1D(verts, 64, 3)
        t0 = time.time()
        for d in speed_dirs:
            lk1d.query(d)
        t_1d = time.time() - t0
        print(f"  1D (3 axes, 64 bins):      {t_1d:.3f}s  ({t_brute/t_1d:.1f}x faster)")

        # Best 2D
        lk2d = Lookup2D(verts, 32)
        t0 = time.time()
        for d in speed_dirs:
            lk2d.query(d)
        t_2d = time.time() - t0
        print(f"  2D (32x32 grid):           {t_2d:.3f}s  ({t_brute/t_2d:.1f}x faster)")

        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
