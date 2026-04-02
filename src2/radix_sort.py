import taichi as ti
from data import data, MAX_TRIANGLES

CHUNK_SIZE = 256
NUM_BUCKETS = 256
MAX_CHUNKS = (MAX_TRIANGLES + CHUNK_SIZE - 1) // CHUNK_SIZE

chunk_histograms = ti.field(dtype=ti.i32, shape=(MAX_CHUNKS, NUM_BUCKETS))
chunk_offsets = ti.field(dtype=ti.i32, shape=(MAX_CHUNKS, NUM_BUCKETS))
bucket_starts = ti.field(dtype=ti.i32, shape=NUM_BUCKETS)


@ti.kernel
def init_sort_indices(n: ti.i32):
    for i in range(n):
        data.sort_indices[i] = i


@ti.kernel
def clear_chunk_histograms(num_chunks: ti.i32):
    for chunk in range(num_chunks):
        for bucket in range(NUM_BUCKETS):
            chunk_histograms[chunk, bucket] = 0


@ti.kernel
def build_chunk_histograms_pass(n: ti.i32, shift: ti.i32, src_is_main: ti.i32):
    num_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk in range(num_chunks):
        start = chunk * CHUNK_SIZE
        end = ti.min(start + CHUNK_SIZE, n)

        local_hist = ti.Matrix([[0] * NUM_BUCKETS], dt=ti.i32)

        for i in range(start, end):
            key = ti.u32(0)
            if src_is_main:
                key = data.morton_codes[i]
            else:
                key = data.morton_codes_temp[i]
            digit = ti.cast((key >> shift) & 0xFF, ti.i32)
            local_hist[0, digit] += 1

        for bucket in range(NUM_BUCKETS):
            chunk_histograms[chunk, bucket] = local_hist[0, bucket]


@ti.kernel
def compute_chunk_offsets_kernel(num_chunks: ti.i32):
    ti.loop_config(serialize=True)
    for _ in range(1):
        for bucket in range(NUM_BUCKETS):
            total = 0
            for chunk in range(num_chunks):
                total += chunk_histograms[chunk, bucket]
            bucket_starts[bucket] = total

        running = 0
        for bucket in range(NUM_BUCKETS):
            old_val = bucket_starts[bucket]
            bucket_starts[bucket] = running
            running += old_val

        for bucket in range(NUM_BUCKETS):
            chunk_running = 0
            for chunk in range(num_chunks):
                chunk_offsets[chunk, bucket] = bucket_starts[bucket] + chunk_running
                chunk_running += chunk_histograms[chunk, bucket]


@ti.kernel
def scatter_pass(n: ti.i32, shift: ti.i32, src_is_main: ti.i32):
    num_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk in range(num_chunks):
        start = chunk * CHUNK_SIZE
        end = ti.min(start + CHUNK_SIZE, n)

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
    if n == 0:
        return

    num_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE

    init_sort_indices(n)

    for pass_idx in range(4):
        shift = pass_idx * 8
        src_is_main = 1 if pass_idx % 2 == 0 else 0

        clear_chunk_histograms(num_chunks)
        build_chunk_histograms_pass(n, shift, src_is_main)
        ti.sync()
        compute_chunk_offsets_kernel(num_chunks)
        ti.sync()
        scatter_pass(n, shift, src_is_main)
