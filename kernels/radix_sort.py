"""GPU Radix Sort for Morton codes.

Optimized approach using per-chunk histograms to avoid atomic contention:
1. Divide data into chunks (e.g., 256 elements each)
2. Each chunk builds local histogram sequentially (no atomics)
3. Global prefix sum across all chunk histograms
4. Scatter using pre-computed chunk offsets
"""
import taichi as ti
import kernels.data as data

# Chunk size for local histograms (tune for GPU - larger = less overhead, smaller = better parallelism)
CHUNK_SIZE = 256
NUM_BUCKETS = 256  # 8-bit radix
MAX_CHUNKS = (data.MAX_TRIANGLES + CHUNK_SIZE - 1) // CHUNK_SIZE

# Per-chunk histograms: [chunk_id, bucket] -> count
chunk_histograms = None
# Global prefix sums per chunk: [chunk_id, bucket] -> starting position
chunk_offsets = None


def init_radix_sort():
    """Initialize radix sort fields. Call after ti.init()"""
    global chunk_histograms, chunk_offsets
    chunk_histograms = ti.field(dtype=ti.i32, shape=(MAX_CHUNKS, NUM_BUCKETS))
    chunk_offsets = ti.field(dtype=ti.i32, shape=(MAX_CHUNKS, NUM_BUCKETS))
    init_prefix_sum_fields()


@ti.kernel
def init_sort_indices(n: ti.i32):
    """Initialize sort indices to identity."""
    for i in range(n):
        data.sort_indices[i] = i


@ti.kernel
def clear_chunk_histograms(num_chunks: ti.i32):
    """Clear all chunk histograms."""
    for chunk in range(num_chunks):
        for bucket in range(NUM_BUCKETS):
            chunk_histograms[chunk, bucket] = 0


@ti.kernel
def build_chunk_histograms_pass(n: ti.i32, shift: ti.i32, src_is_main: ti.i32):
    """Build histogram for each chunk - no atomics within chunk.

    Each chunk processes CHUNK_SIZE elements sequentially,
    building a local histogram without atomic operations.
    """
    num_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk in range(num_chunks):
        start = chunk * CHUNK_SIZE
        end = ti.min(start + CHUNK_SIZE, n)

        # Local histogram for this chunk (in registers/local memory)
        local_hist = ti.Matrix([[0] * NUM_BUCKETS], dt=ti.i32)

        for i in range(start, end):
            key = ti.u32(0)
            if src_is_main:
                key = data.morton_codes[i]
            else:
                key = data.morton_codes_temp[i]
            digit = ti.cast((key >> shift) & 0xFF, ti.i32)
            local_hist[0, digit] += 1

        # Write local histogram to global memory (one write per bucket)
        for bucket in range(NUM_BUCKETS):
            chunk_histograms[chunk, bucket] = local_hist[0, bucket]


# Single field for prefix sum (256 values)
bucket_starts = None


def init_prefix_sum_fields():
    """Initialize prefix sum field. Call after ti.init()"""
    global bucket_starts
    bucket_starts = ti.field(dtype=ti.i32, shape=NUM_BUCKETS)


@ti.kernel
def compute_chunk_offsets_single_kernel(num_chunks: ti.i32):
    """Compute bucket totals, prefix sum, and chunk offsets in one kernel.

    Uses a single thread for the prefix sum (256 values is tiny for GPU).
    This avoids the overhead of multiple kernel launches.
    """
    # Single thread does everything sequentially (faster than kernel launch overhead)
    ti.loop_config(serialize=True)
    for _ in range(1):
        # Step 1: Sum chunks per bucket
        for bucket in range(NUM_BUCKETS):
            total = 0
            for chunk in range(num_chunks):
                total += chunk_histograms[chunk, bucket]
            bucket_starts[bucket] = total

        # Step 2: Exclusive prefix sum
        running = 0
        for bucket in range(NUM_BUCKETS):
            old_val = bucket_starts[bucket]
            bucket_starts[bucket] = running
            running += old_val

        # Step 3: Compute chunk offsets
        for bucket in range(NUM_BUCKETS):
            chunk_running = 0
            for chunk in range(num_chunks):
                chunk_offsets[chunk, bucket] = bucket_starts[bucket] + chunk_running
                chunk_running += chunk_histograms[chunk, bucket]


def compute_chunk_offsets(n: int, num_chunks: int):
    """Single kernel for all prefix sum operations."""
    compute_chunk_offsets_single_kernel(num_chunks)


@ti.kernel
def scatter_pass(n: ti.i32, shift: ti.i32, src_is_main: ti.i32):
    """Scatter elements to sorted positions.

    Each chunk scatters its elements using pre-computed offsets.
    Elements within a chunk are processed sequentially to maintain
    stable sort order without atomics.
    """
    num_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk in range(num_chunks):
        start = chunk * CHUNK_SIZE
        end = ti.min(start + CHUNK_SIZE, n)

        # Local offset counters for this chunk
        local_offsets = ti.Matrix([[0] * NUM_BUCKETS], dt=ti.i32)
        for bucket in range(NUM_BUCKETS):
            local_offsets[0, bucket] = chunk_offsets[chunk, bucket]

        for i in range(start, end):
            key = ti.u32(0)
            idx = 0
            if src_is_main:
                key = data.morton_codes[i]
                idx = data.sort_indices[i]
            else:
                key = data.morton_codes_temp[i]
                idx = data.sort_indices_temp[i]

            digit = ti.cast((key >> shift) & 0xFF, ti.i32)
            dest = local_offsets[0, digit]
            local_offsets[0, digit] += 1

            if src_is_main:
                data.morton_codes_temp[dest] = key
                data.sort_indices_temp[dest] = idx
            else:
                data.morton_codes[dest] = key
                data.sort_indices[dest] = idx


def radix_sort_morton(n: int):
    """Sort morton_codes in-place, tracking permutation in sort_indices.

    Uses 4 passes for 32-bit keys (8 bits per pass).
    After sorting:
    - morton_codes contains sorted values
    - sort_indices[i] = original index of element now at position i
    """
    if n == 0:
        return

    num_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Initialize indices and run all 4 passes
    init_sort_indices(n)

    for pass_idx in range(4):
        shift = pass_idx * 8
        src_is_main = 1 if pass_idx % 2 == 0 else 0

        clear_chunk_histograms(num_chunks)
        build_chunk_histograms_pass(n, shift, src_is_main)
        ti.sync()  # Histogram must complete before prefix sum
        compute_chunk_offsets(n, num_chunks)
        ti.sync()  # Offsets must complete before scatter
        scatter_pass(n, shift, src_is_main)
