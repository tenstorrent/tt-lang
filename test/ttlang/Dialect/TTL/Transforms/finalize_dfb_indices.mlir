// Tests for ttl-finalize-dfb-indices pass.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=CHECK
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=REUSE
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=OVERLAP
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=FOUR
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' -debug-only=ttl-finalize-dfb-indices 2>&1 | FileCheck %s --check-prefix=DEBUG
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=MIXED
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=NOPOP
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=THREE
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=SINGLE
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=USER
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=XTHREAD
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=EPOCH

// -----

// User DFBs at indices 0, 1, 2 (unused: keep their indices) and a
// compiler-allocated DFB at index 3. The pass should update base_cta_index
// to 4 and emit ttl.compiler_allocated_dfbs.

// CHECK: module attributes {ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 3 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}]}

// CHECK-LABEL: func.func @reader
// CHECK-SAME: ttl.base_cta_index = 4 : i32
func.func @reader()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = [0 : i32, 1 : i32]} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// CHECK-LABEL: func.func @compute
// CHECK-SAME: ttl.base_cta_index = 4 : i32
func.func @compute()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc = ttl.bind_cb {cb_index = 3, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// CHECK-LABEL: func.func @writer
// CHECK-SAME: ttl.base_cta_index = 4 : i32
func.func @writer()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = [0 : i32, 1 : i32]} {
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// No compiler-allocated DFBs: pass should not add ttl.compiler_allocated_dfbs,
// but should still update base_cta_index to the true DFB count (3).

// CHECK-NOT: ttl.compiler_allocated_dfbs

// CHECK-LABEL: func.func @compute_only
// CHECK-SAME: ttl.base_cta_index = 3 : i32
func.func @compute_only()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 2 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Non-overlapping compiler-allocated DFBs: DFB #3 is released (cb_pop)
// before DFB #4 is allocated (bind_cb). Both should be assigned index 3.

// DEBUG: DFB reuse: cb4 -> cb3
// DEBUG: Total DFB count: 4

// REUSE: module attributes {ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 3 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}]}

// REUSE-LABEL: func.func @non_overlapping_reuse
// REUSE-SAME: ttl.base_cta_index = 4 : i32
// REUSE-COUNT-2: ttl.bind_cb{cb_index = 3,
// REUSE-NOT: cb_index = 4
// REUSE: return
func.func @non_overlapping_reuse()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc3 = ttl.bind_cb {cb_index = 3, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc4 = ttl.bind_cb {cb_index = 4, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Overlapping compiler-allocated DFBs: DFB #4 is allocated while DFB #3
// is still live. They must keep separate indices.

// DEBUG: Total DFB count: 5

// OVERLAP: module attributes {ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 3 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}, {block_count = 2 : i32, dfb_index = 4 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}]}

// OVERLAP-LABEL: func.func @overlapping_no_reuse
// OVERLAP-SAME: ttl.base_cta_index = 5 : i32
// OVERLAP: ttl.bind_cb{cb_index = 3,
// OVERLAP: ttl.bind_cb{cb_index = 4,
// OVERLAP-NOT: cb_index = 5
// OVERLAP: return
func.func @overlapping_no_reuse()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc3 = ttl.bind_cb {cb_index = 3, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc4 = ttl.bind_cb {cb_index = 4, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Four compiler-allocated DFBs with nested lifetimes (softmax pattern).
// DFB-A [bind, pop]: spans past DFB-B
// DFB-B [bind, pop]: nested within A, dies before A
// DFB-C [bind, pop]: starts after A dies, spans past DFB-D
// DFB-D [bind, pop]: nested within C, dies before C
// Result: A and C share slot 0 (index 3), B and D share slot 1 (index 4).

// DEBUG: DFB reuse: cb5 -> cb3
// DEBUG: DFB reuse: cb6 -> cb4
// DEBUG: Total DFB count: 5

// FOUR: module attributes {ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 3 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}, {block_count = 2 : i32, dfb_index = 4 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}]}

// FOUR-LABEL: func.func @four_dfbs_nested_reuse
// FOUR-SAME: ttl.base_cta_index = 5 : i32
//
// DFB-A -> slot 0 (index 3)
// FOUR: ttl.bind_cb{cb_index = 3, {{.*}}} {ttl.compiler_allocated}
// DFB-B -> slot 1 (index 4)
// FOUR: ttl.bind_cb{cb_index = 4, {{.*}}} {ttl.compiler_allocated}
// FOUR: ttl.cb_pop
// FOUR: ttl.cb_pop
// DFB-C -> slot 0 (index 3, reused from A)
// FOUR: ttl.bind_cb{cb_index = 3, {{.*}}} {ttl.compiler_allocated}
// DFB-D -> slot 1 (index 4, reused from B)
// FOUR: ttl.bind_cb{cb_index = 4, {{.*}}} {ttl.compiler_allocated}
// FOUR: ttl.cb_pop
// FOUR: ttl.cb_pop
// FOUR-NOT: cb_index = 5
// FOUR-NOT: cb_index = 6
// FOUR: return
func.func @four_dfbs_nested_reuse()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %allocA = ttl.bind_cb {cb_index = 3, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %allocB = ttl.bind_cb {cb_index = 4, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %allocB : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %allocA : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %allocC = ttl.bind_cb {cb_index = 5, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %allocD = ttl.bind_cb {cb_index = 6, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %allocD : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %allocC : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Compiler DFBs preserve the established CircularBufferType partitioning.
// The two [1,1] buffers share one slot, while the [2,4] buffer remains in a
// separate slot even though all three have the same element type.

// DEBUG: DFB reuse: cb5 -> cb3
// DEBUG: Total DFB count: 5

// MIXED: module attributes {ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 3 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}, {block_count = 2 : i32, dfb_index = 4 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 8 : i32}]}

// MIXED-LABEL: func.func @mixed_shapes_share_slot
// MIXED-SAME: ttl.base_cta_index = 5 : i32
// MIXED: ttl.bind_cb{cb_index = 3, {{.*}}} {ttl.compiler_allocated} : <[1, 1],
// MIXED: ttl.bind_cb{cb_index = 4, {{.*}}} {ttl.compiler_allocated} : <[2, 4],
// MIXED: ttl.bind_cb{cb_index = 3, {{.*}}} {ttl.compiler_allocated} : <[1, 1],
// MIXED-NOT: cb_index = 5
// MIXED: return
func.func @mixed_shapes_share_slot()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc3 = ttl.bind_cb {cb_index = 3, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc4 = ttl.bind_cb {cb_index = 4, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc4 : <[2, 4], !ttcore.tile<32x32, bf16>, 2>
  %alloc5 = ttl.bind_cb {cb_index = 5, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc5 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// No cb_pop on either compiler-allocated DFB. No acquires or releases at
// all: both keep their indices.

// DEBUG: Total DFB count: 5

// NOPOP: module attributes {ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 3 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}, {block_count = 2 : i32, dfb_index = 4 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}]}

// NOPOP-LABEL: func.func @no_cb_pop_conservative
// NOPOP-SAME: ttl.base_cta_index = 5 : i32
// NOPOP: ttl.bind_cb{cb_index = 3,
// NOPOP: ttl.bind_cb{cb_index = 4,
// NOPOP-NOT: cb_index = 5
// NOPOP: return
func.func @no_cb_pop_conservative()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc3 = ttl.bind_cb {cb_index = 3, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc4 = ttl.bind_cb {cb_index = 4, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Three sequential non-overlapping DFBs. All should map to a single
// physical slot (multi-round slot recycling).

// DEBUG: DFB reuse: cb4 -> cb3
// DEBUG: DFB reuse: cb5 -> cb3
// DEBUG: Total DFB count: 4

// THREE: module attributes {ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 3 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}]}

// THREE-LABEL: func.func @three_sequential_one_slot
// THREE-SAME: ttl.base_cta_index = 4 : i32
// THREE-COUNT-3: ttl.bind_cb{cb_index = 3,
// THREE-NOT: cb_index = 4
// THREE: return
func.func @three_sequential_one_slot()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc3 = ttl.bind_cb {cb_index = 3, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc4 = ttl.bind_cb {cb_index = 4, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc5 = ttl.bind_cb {cb_index = 5, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc5 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Single compiler-allocated DFB. Index and module attribute unchanged.

// SINGLE: module attributes {ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 3 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}]}

// SINGLE-LABEL: func.func @single_dfb_no_reuse
// SINGLE-SAME: ttl.base_cta_index = 4 : i32
// SINGLE: ttl.bind_cb{cb_index = 3, {{.*}}} {ttl.compiler_allocated}
// SINGLE-NOT: cb_index = 4
// SINGLE: return
func.func @single_dfb_no_reuse()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc3 = ttl.bind_cb {cb_index = 3, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Fully balanced user DFBs local to one kernel thread and with disjoint
// lifetimes share one index. The pass emits ttl.dfb_index_map for the runtime.

// DEBUG: DFB reuse: cb1 -> cb0
// DEBUG: Total DFB count: 1

// USER: module attributes {ttl.dfb_index_map = [{new_index = 0 : i32, old_index = 1 : i32}]}

// USER-LABEL: func.func @thread_local_user
// USER-SAME: ttl.base_cta_index = 1 : i32
// USER-COUNT-2: ttl.bind_cb{cb_index = 0,
// USER-NOT: cb_index = 1
// USER: return
func.func @thread_local_user()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.base_cta_index = 2 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %r0 = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %r1 = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w1 = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// User DFBs with different producer threads never share an index, even with
// disjoint lifetimes: the physical CB counters and per-RISC pointers do not
// survive a producer change.

// XTHREAD-LABEL: func.func @xt_reader
// XTHREAD: ttl.bind_cb{cb_index = 0,
// XTHREAD-LABEL: func.func @xt_writer
// XTHREAD: ttl.bind_cb{cb_index = 1,
// XTHREAD-LABEL: func.func @xt_compute
// XTHREAD: ttl.bind_cb{cb_index = 0,
// XTHREAD: ttl.bind_cb{cb_index = 1,
func.func @xt_reader()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.base_cta_index = 2 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %r0 = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

func.func @xt_writer()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.base_cta_index = 2 : i32,
                ttl.crta_indices = []} {
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %r1 = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

func.func @xt_compute()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 2 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w1 = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// An explicit epoch boundary restarts physical indices at zero even when the
// next epoch changes data type and page size.

// EPOCH: ttl.dfb_epoch_physical_configs = [{dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, tile_height = 32 : i32, tile_width = 32 : i32, total_size = 8192 : i64}]
// EPOCH: ttl.dfb_index_map = [{new_index = 0 : i32, old_index = 1 : i32}]
// EPOCH-LABEL: func.func @epoch_restart
// EPOCH-COUNT-2: ttl.bind_cb{cb_index = 0,
// EPOCH: ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [0 : i64, 1 : i64, 0 : i64, 4096 : i64, 2 : i64, 2048 : i64, 5 : i64, 32 : i64, 32 : i64, 16 : i64, 4 : i64, 5 : i64, 5 : i64]}
// EPOCH: ttl.cb_reserve
// EPOCH: ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [0 : i64, 1 : i64, 0 : i64, 8192 : i64, 2 : i64, 4096 : i64, 0 : i64, 32 : i64, 32 : i64, 16 : i64, 4 : i64, 5 : i64, 5 : i64]}
// EPOCH: ttl.cb_reserve
// EPOCH-NOT: cb_index = 1
// EPOCH: return
func.func @epoch_restart()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 2 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %r0 = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
  %r1 = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.cb_push %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  %w1 = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  return
}
