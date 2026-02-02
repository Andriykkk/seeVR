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


@ti.kernel
def compute_chunk_offsets(n: ti.i32, num_chunks: ti.i32):
    """Compute global offsets for each (chunk, bucket) pair.

    For bucket b in chunk c, the offset is:
      sum of all counts for buckets < b across ALL chunks
      + sum of counts for bucket b in chunks < c

    This is computed sequentially (small data - num_chunks * 256 elements).
    """
    # First pass: compute total count per bucket across all chunks
    # and prefix sum across buckets
    bucket_totals = ti.Matrix([[0] * NUM_BUCKETS], dt=ti.i32)

    for bucket in range(NUM_BUCKETS):
        total = 0
        for chunk in range(num_chunks):
            total += chunk_histograms[chunk, bucket]
        bucket_totals[0, bucket] = total

    # Prefix sum of bucket totals (where each bucket's data starts globally)
    bucket_starts = ti.Matrix([[0] * NUM_BUCKETS], dt=ti.i32)
    running = 0
    for bucket in range(NUM_BUCKETS):
        bucket_starts[0, bucket] = running
        running += bucket_totals[0, bucket]

    # For each chunk and bucket, compute the offset
    # offset[chunk, bucket] = bucket_starts[bucket] + sum of histogram[c, bucket] for c < chunk
    for bucket in range(NUM_BUCKETS):
        running_in_bucket = 0
        for chunk in range(num_chunks):
            chunk_offsets[chunk, bucket] = bucket_starts[0, bucket] + running_in_bucket
            running_in_bucket += chunk_histograms[chunk, bucket]


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

    # Initialize indices
    init_sort_indices(n)
    ti.sync()

    # Debug: verify init
    print(f"Radix sort: n={n}, chunks={num_chunks}")
    print(f"  After init: sort_indices[0:5] = {[data.sort_indices[i] for i in range(min(5, n))]}")
    print(f"  Morton codes[0:5] = {[hex(data.morton_codes[i]) for i in range(min(5, n))]}")

    # Pass 0: bits 0-7 (main -> temp)
    clear_chunk_histograms(num_chunks)
    build_chunk_histograms_pass(n, 0, 1)  # src_is_main=1
    compute_chunk_offsets(n, num_chunks)
    scatter_pass(n, 0, 1)
    ti.sync()
    print(f"  After pass 0: sort_indices_temp[0:5] = {[data.sort_indices_temp[i] for i in range(min(5, n))]}")
    print(f"  After pass 0: morton_codes_temp[0:5] = {[hex(data.morton_codes_temp[i]) for i in range(min(5, n))]}")

    # Pass 1: bits 8-15 (temp -> main)
    clear_chunk_histograms(num_chunks)
    ti.sync()
    build_chunk_histograms_pass(n, 8, 0)  # src_is_main=0
    ti.sync()
    # Debug: check histogram
    print(f"  Pass 1 histogram check - chunk_histograms[0, 0:5] = {[chunk_histograms[0, i] for i in range(5)]}")
    compute_chunk_offsets(n, num_chunks)
    ti.sync()
    print(f"  Pass 1 offsets check - chunk_offsets[0, 0:5] = {[chunk_offsets[0, i] for i in range(5)]}")
    scatter_pass(n, 8, 0)
    ti.sync()
    print(f"  After pass 1: sort_indices[0:5] = {[data.sort_indices[i] for i in range(min(5, n))]}")
    print(f"  After pass 1: morton_codes[0:5] = {[hex(data.morton_codes[i]) for i in range(min(5, n))]}")

    # Pass 2: bits 16-23 (main -> temp)
    clear_chunk_histograms(num_chunks)
    build_chunk_histograms_pass(n, 16, 1)
    compute_chunk_offsets(n, num_chunks)
    scatter_pass(n, 16, 1)
    ti.sync()
    print(f"  After pass 2: sort_indices_temp[0:5] = {[data.sort_indices_temp[i] for i in range(min(5, n))]}")

    # Pass 3: bits 24-31 (temp -> main)
    clear_chunk_histograms(num_chunks)
    build_chunk_histograms_pass(n, 24, 0)
    compute_chunk_offsets(n, num_chunks)
    scatter_pass(n, 24, 0)
    ti.sync()
    print(f"  After pass 3 (final): sort_indices[0:5] = {[data.sort_indices[i] for i in range(min(5, n))]}")
