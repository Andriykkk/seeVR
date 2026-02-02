"""Benchmark script for LBVH build stages."""
import taichi as ti
import time

ti.init(arch=ti.gpu)

import kernels.data as data
data.init_scene()

from kernels.radix_sort import init_radix_sort
init_radix_sort()

from kernels.bvh import (
    bvh_init_centroids, compute_scene_bounds, compute_morton_codes,
    build_lbvh_hierarchy, compute_leaf_aabbs, clear_aabb_flags,
    propagate_aabbs_atomic
)
from kernels.radix_sort import radix_sort_morton
from main import scene


def benchmark_bvh_stages():
    """Benchmark each stage of BVH construction."""
    # Load mesh
    scene.add_mesh_from_obj('./models/dragon_smallest.obj', center=(0, 2.5, 0), size=8.0)
    n = data.num_triangles[None]
    print(f"Triangles: {n}")
    print()

    # Warm up ALL kernels (first kernel launch is slow due to JIT compilation)
    print("Warming up kernels (JIT compilation)...")
    bvh_init_centroids(n)
    data.scene_aabb_min[None] = [1e30, 1e30, 1e30]
    data.scene_aabb_max[None] = [-1e30, -1e30, -1e30]
    compute_scene_bounds(n)
    ti.sync()
    compute_morton_codes(n)
    radix_sort_morton(n)
    build_lbvh_hierarchy(n)
    compute_leaf_aabbs(n)
    clear_aabb_flags(n)
    propagate_aabbs_atomic(n)
    ti.sync()
    print("Warm-up complete.\n")

    NUM_RUNS = 10
    results = {
        "1. Centroids": [],
        "2. Scene bounds": [],
        "3. Morton codes": [],
        "4. Radix sort": [],
        "5. Build hierarchy": [],
        "6. Leaf AABBs": [],
        "7. Clear flags": [],
        "8. Propagate AABBs": [],
    }

    for run in range(NUM_RUNS):
        # 1. Compute centroids
        ti.sync()
        start = time.perf_counter()
        bvh_init_centroids(n)
        ti.sync()
        results["1. Centroids"].append(time.perf_counter() - start)

        # 2. Scene bounds
        data.scene_aabb_min[None] = [1e30, 1e30, 1e30]
        data.scene_aabb_max[None] = [-1e30, -1e30, -1e30]
        ti.sync()
        start = time.perf_counter()
        compute_scene_bounds(n)
        ti.sync()
        results["2. Scene bounds"].append(time.perf_counter() - start)

        # 3. Morton codes
        ti.sync()
        start = time.perf_counter()
        compute_morton_codes(n)
        ti.sync()
        results["3. Morton codes"].append(time.perf_counter() - start)

        # 4. Radix sort
        ti.sync()
        start = time.perf_counter()
        radix_sort_morton(n)
        ti.sync()
        results["4. Radix sort"].append(time.perf_counter() - start)

        # 5. Build hierarchy
        ti.sync()
        start = time.perf_counter()
        build_lbvh_hierarchy(n)
        ti.sync()
        results["5. Build hierarchy"].append(time.perf_counter() - start)

        # 6. Leaf AABBs
        ti.sync()
        start = time.perf_counter()
        compute_leaf_aabbs(n)
        ti.sync()
        results["6. Leaf AABBs"].append(time.perf_counter() - start)

        # 7. Clear flags
        ti.sync()
        start = time.perf_counter()
        clear_aabb_flags(n)
        ti.sync()
        results["7. Clear flags"].append(time.perf_counter() - start)

        # 8. Propagate AABBs
        ti.sync()
        start = time.perf_counter()
        propagate_aabbs_atomic(n)
        ti.sync()
        results["8. Propagate AABBs"].append(time.perf_counter() - start)

    # Print results
    print("=" * 70)
    print(f"BVH Build Stage Timings ({NUM_RUNS} runs)")
    print("=" * 70)
    print(f"{'Stage':25s} {'Min':>10s} {'Avg':>10s} {'Max':>10s}")
    print("-" * 70)
    total_avg = 0
    for name in results:
        times = results[name]
        min_t = min(times) * 1000
        avg_t = sum(times) / len(times) * 1000
        max_t = max(times) * 1000
        print(f"{name:25s} {min_t:8.3f} ms {avg_t:8.3f} ms {max_t:8.3f} ms")
        total_avg += avg_t
    print("-" * 70)
    print(f"{'Total (avg)':25s} {total_avg:30.3f} ms")
    print("=" * 70)

    # Verify result
    node0 = data.bvh_nodes[0]
    print(f"\nRoot AABB: min={node0.aabb_min}, max={node0.aabb_max}")


if __name__ == "__main__":
    benchmark_bvh_stages()
