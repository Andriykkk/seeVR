Optimization plan
1. Scene bounds (73ms → <0.1ms)
Currently: 1 thread loops over 1M centroids.

Fix: Parallel reduction in two passes:

Pass 1: Each workgroup (256 threads) reduces 256 centroids to 1 min/max using shared memory. Write per-group results to a temp buffer. Dispatch num_tris/256 workgroups.
Pass 2: 1 workgroup reduces the ~4000 group results to the final bounds.
2. Radix sort (948ms → <2ms)
Currently: 1 thread does counting sort over 1M entries, 4 times.

Fix: Parallel radix sort — standard GPU approach, each pass has 3 dispatches:

Local histogram: Each workgroup counts 256 bins for its chunk of data using shared memory. Output: [num_groups × 256] histogram table.
Prefix sum: Exclusive scan over the histogram table to get global offsets. Can be done in a few workgroups with shared memory.
Scatter: Each thread reads its key, looks up the global offset from the prefix sum, writes to destination.
This needs one extra buffer: the histogram table (num_groups × 256 × sizeof(uint)). About 4MB for 4000 workgroups.

3. Physics AABB (487ms → <1ms)
Currently: 1 thread per body, loops over 263k vertices.

Fix: Parallel reduction per body. Two options:

Option A: Dispatch 1 thread per vertex. Each thread writes its vertex's min/max to a per-body atomicMin/atomicMax. GLSL doesn't have float atomics natively, but you can use atomicMin on int with float-to-sortable-int conversion.
Option B: Two-pass — first pass: each workgroup reduces a chunk of vertices to local min/max. Second pass: reduce per-body group results. Cleaner but needs a temp buffer.
4. Render verts (49.5ms → <1ms)
Currently: 1 thread per body, loops over 263k vertices.

Fix: Dispatch 1 thread per vertex. Each thread looks up its body via a vertex-to-body mapping (or just iterate: for body b, if vert_start <= thread_id < vert_start+vert_count). Or simpler — change the dispatch from num_bodies workgroups to num_vertices/256 workgroups, and in the shader each thread handles one vertex, looks up which body owns it.

5. Narrow phase (915ms → ???)
91.5ms per substep dispatching MAX_COLLISION_PAIRS/256 = 39 workgroups. But most pairs are empty — only the actual broad phase output matters. The shader early-exits with if (i >= actual_pairs) return; but the threads are still launched and the dispatch still costs GPU scheduling.

Fix: Read back actual_pairs count before narrow phase and dispatch only (actual_pairs+255)/256 workgroups. Or use indirect dispatch — write the workgroup count from the broad phase shader into an indirect buffer, then vkCmdDispatchIndirect.

Priority order
Radix sort — biggest win (948ms), but most complex to implement
AABB + render verts — second biggest (537ms combined), simpler fix (dispatch per vertex)
Scene bounds — easy parallel reduction (73ms)
Narrow phase — indirect dispatch or readback (915ms but needs API change)
The first three are pure shader changes. Narrow phase needs either a GPU readback (adds latency) or indirect dispatch (needs a new Vulkan buffer + vkCmdDispatchIndirect).