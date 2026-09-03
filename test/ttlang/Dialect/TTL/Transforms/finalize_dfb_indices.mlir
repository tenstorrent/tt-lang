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
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=CYCLIC1
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=CYCLIC2
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=CYCLIC3
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=UNPACK
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=PRESERVE

// -----

// User DFBs at indices 0, 1, 2 (unused: keep their indices) and a
// compiler-allocated DFB at index 3. The pass should update base_cta_index
// to 4 and emit ttl.compiler_allocated_dfbs.

// CHECK: ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 3 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}]

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

// REUSE: ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 3 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}]

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

// OVERLAP: ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 3 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}, {block_count = 2 : i32, dfb_index = 4 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}]

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

// FOUR: ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 3 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}, {block_count = 2 : i32, dfb_index = 4 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}]

// FOUR-LABEL: func.func @four_dfbs_nested_reuse
// FOUR-SAME: ttl.base_cta_index = 5 : i32
//
// DFB-A -> slot 0 (index 3)
// FOUR: ttl.bind_cb{cb_index = 3, block_count = 2} {ttl.compiler_allocated, ttl.dfb_logical_index = 3 : i64}
// DFB-B -> slot 1 (index 4)
// FOUR: ttl.bind_cb{cb_index = 4, block_count = 2} {ttl.compiler_allocated, ttl.dfb_logical_index = 4 : i64}
// FOUR: ttl.cb_pop
// FOUR: ttl.cb_pop
// DFB-C -> slot 0 (index 3, reused from A)
// FOUR: ttl.bind_cb{cb_index = 3, block_count = 2} {ttl.compiler_allocated, ttl.dfb_logical_index = 5 : i64}
// DFB-D -> slot 1 (index 4, reused from B)
// FOUR: ttl.bind_cb{cb_index = 4, block_count = 2} {ttl.compiler_allocated, ttl.dfb_logical_index = 6 : i64}
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

// MIXED: ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 3 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}, {block_count = 2 : i32, dfb_index = 4 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 8 : i32}]

// MIXED-LABEL: func.func @mixed_shapes_share_slot
// MIXED-SAME: ttl.base_cta_index = 5 : i32
// MIXED: ttl.bind_cb{cb_index = 3, {{.*}}} {ttl.compiler_allocated{{.*}}} : <[1, 1],
// MIXED: ttl.bind_cb{cb_index = 4, {{.*}}} {ttl.compiler_allocated{{.*}}} : <[2, 4],
// MIXED: ttl.bind_cb{cb_index = 3, {{.*}}} {ttl.compiler_allocated{{.*}}} : <[1, 1],
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

// NOPOP: ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 3 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}, {block_count = 2 : i32, dfb_index = 4 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}]

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

// THREE: ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 3 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}]

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

// SINGLE: ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 3 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}]

// SINGLE-LABEL: func.func @single_dfb_no_reuse
// SINGLE-SAME: ttl.base_cta_index = 4 : i32
// SINGLE: ttl.bind_cb{cb_index = 3, {{.*}}} {ttl.compiler_allocated{{.*}}}
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

// USER: ttl.dfb_index_map = [{new_index = 0 : i32, old_index = 1 : i32}]
// USER: ttl.logical_dfb_configs = [
// USER-SAME: {block_count = 2 : i32, compiler_allocated = false, element_type = !ttcore.tile<32x32, bf16>, elems_per_block = 1 : i32, epoch = 0 : i32, logical_index = 0 : i32, num_pages = 2 : i32, physical_index = 0 : i32, unpack_to_dest_fp32 = false}
// USER-SAME: , {block_count = 2 : i32, compiler_allocated = false, element_type = !ttcore.tile<32x32, bf16>, elems_per_block = 1 : i32, epoch = 0 : i32, logical_index = 1 : i32, num_pages = 2 : i32, physical_index = 0 : i32, unpack_to_dest_fp32 = false}]

// USER-LABEL: func.func @thread_local_user
// USER-SAME: ttl.base_cta_index = 1 : i32
// USER: ttl.bind_cb{cb_index = 0, block_count = 2} {ttl.dfb_logical_index = 0 : i64}
// USER: ttl.bind_cb{cb_index = 0, block_count = 2} {ttl.dfb_logical_index = 1 : i64}
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
// EPOCH: ttl.logical_dfb_configs = [
// EPOCH-SAME: {block_count = 2 : i32, compiler_allocated = false, element_type = !ttcore.tile<32x32, bf16>, elems_per_block = 1 : i32, epoch = 0 : i32, logical_index = 0 : i32, num_pages = 2 : i32, physical_index = 0 : i32, unpack_to_dest_fp32 = false}
// EPOCH-SAME: , {block_count = 2 : i32, compiler_allocated = false, element_type = !ttcore.tile<32x32, f32>, elems_per_block = 1 : i32, epoch = 1 : i32, logical_index = 1 : i32, num_pages = 2 : i32, physical_index = 0 : i32, unpack_to_dest_fp32 = false}]
// EPOCH-LABEL: func.func @epoch_restart
// EPOCH: ttl.bind_cb{cb_index = 0, block_count = 2} {ttl.dfb_logical_index = 0 : i64}
// EPOCH: ttl.bind_cb{cb_index = 0, block_count = 2} {ttl.dfb_logical_index = 1 : i64}
// EPOCH: ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [0, 1, 0, 4096, 2, 2048, 5, 32, 32, 16, 4, 5, 5], ttl.dfb_reset_epoch = 0 : i32, ttl.dfb_reset_preserved_indices = []}
// EPOCH: ttl.cb_reserve
// EPOCH: ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [0, 1, 0, 8192, 2, 4096, 0, 32, 32, 16, 4, 5, 5], ttl.dfb_reset_epoch = 1 : i32, ttl.dfb_reset_preserved_indices = []}
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

// -----

// A single cyclic cut restores phase zero on every loop backedge.

// CYCLIC1: ttl.dfb_epoch_physical_configs = []
// CYCLIC1: ttl.logical_dfb_configs = []
// CYCLIC1-LABEL: func.func @one_cyclic_cut
// CYCLIC1: ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [0, 0], ttl.dfb_reset_epoch = 0 : i32, ttl.dfb_reset_preserved_indices = []}
// CYCLIC1: scf.for
// CYCLIC1: ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [0, 0], ttl.dfb_reset_epoch = 0 : i32, ttl.dfb_reset_preserved_indices = []}
// CYCLIC1: return
func.func @one_cyclic_cut()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
  }
  return
}

// -----

// Two resets directly in a resident loop alternate between phase A and phase
// B. The prologue configures A, the first reset configures B, and the second
// reset restores A before the loop backedge.

// CYCLIC2: ttl.dfb_epoch_physical_configs = [{dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, tile_height = 32 : i32, tile_width = 32 : i32, total_size = 8192 : i64}]
// CYCLIC2: ttl.dfb_index_map = [{new_index = 0 : i32, old_index = 1 : i32}]
// CYCLIC2-LABEL: func.func @two_phase_resident_loop
// CYCLIC2-COUNT-2: ttl.bind_cb{cb_index = 0,
// CYCLIC2: ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [0, 1, 0, 4096, 2, 2048, 5, 32, 32, 16, 4, 5, 5], ttl.dfb_reset_epoch = 0 : i32, ttl.dfb_reset_preserved_indices = []}
// CYCLIC2: scf.for
// CYCLIC2: ttl.cb_reserve
// CYCLIC2: ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [0, 1, 0, 8192, 2, 4096, 0, 32, 32, 16, 4, 5, 5], ttl.dfb_reset_epoch = 1 : i32, ttl.dfb_reset_preserved_indices = []}
// CYCLIC2: ttl.cb_reserve
// CYCLIC2: ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [0, 1, 0, 4096, 2, 2048, 5, 32, 32, 16, 4, 5, 5], ttl.dfb_reset_epoch = 0 : i32, ttl.dfb_reset_preserved_indices = []}
// CYCLIC2: return
func.func @two_phase_resident_loop()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 2 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %r0 = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %w0 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
    %r1 = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    %w1 = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
  }
  return
}

// -----

// Three resets support an additional terminal phase while retaining the same
// cyclic contract. The emitted order is A prologue, B, terminal, then A.

// CYCLIC3: ttl.dfb_epoch_physical_configs = [{dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, tile_height = 32 : i32, tile_width = 32 : i32, total_size = 8192 : i64}]
// CYCLIC3: ttl.dfb_index_map = [{new_index = 0 : i32, old_index = 1 : i32}, {new_index = 0 : i32, old_index = 2 : i32}]
// CYCLIC3-LABEL: func.func @three_phase_resident_loop
// CYCLIC3-COUNT-3: ttl.bind_cb{cb_index = 0,
// CYCLIC3: ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [0, 1, 0, 4096, 2, 2048, 5, 32, 32, 16, 4, 5, 5], ttl.dfb_reset_epoch = 0 : i32, ttl.dfb_reset_preserved_indices = []}
// CYCLIC3: scf.for
// CYCLIC3: ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [0, 1, 0, 8192, 2, 4096, 0, 32, 32, 16, 4, 5, 5], ttl.dfb_reset_epoch = 1 : i32, ttl.dfb_reset_preserved_indices = []}
// CYCLIC3: ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [0, 1, 0, 2048, 2, 1024, 5, 16, 32, 16, 2, 5, 5], ttl.dfb_reset_epoch = 2 : i32, ttl.dfb_reset_preserved_indices = []}
// CYCLIC3: ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [0, 1, 0, 4096, 2, 2048, 5, 32, 32, 16, 4, 5, 5], ttl.dfb_reset_epoch = 0 : i32, ttl.dfb_reset_preserved_indices = []}
// CYCLIC3: return
func.func @three_phase_resident_loop()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<16x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %r0 = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %w0 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
    %r1 = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_push %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    %w1 = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
    %r2 = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<16x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<16x32, bf16>>
    ttl.cb_push %cb2 : <[1, 1], !ttcore.tile<16x32, bf16>, 2>
    %w2 = ttl.cb_wait %cb2 : <[1, 1], !ttcore.tile<16x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<16x32, bf16>>
    ttl.cb_pop %cb2 : <[1, 1], !ttcore.tile<16x32, bf16>, 2>
    ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
  }
  return
}

// -----

// The logical configuration records FP32 unpack routing even without a reset
// epoch, so later per-core analysis does not lose the compute-kernel contract.

// UNPACK: ttl.logical_dfb_configs = [{block_count = 2 : i32, compiler_allocated = false, element_type = !ttcore.tile<32x32, f32>, elems_per_block = 1 : i32, epoch = 0 : i32, logical_index = 0 : i32, num_pages = 2 : i32, physical_index = 0 : i32, unpack_to_dest_fp32 = true}]
// UNPACK-LABEL: func.func @no_reset_unpack_to_dest_fp32
func.func @no_reset_unpack_to_dest_fp32()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 1 : i32,
                ttl.unpack_to_dest_fp32 = array<i32: 0>} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %r0 = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.cb_push %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  %w0 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  return
}

// -----

// A DFB preserved across the first reset is pinned above the epoch-local
// index space. The preserving reset omits it; the prologue and next reset
// configure it normally.

// PRESERVE: ttl.dfb_epoch_physical_configs = [{dfb_index = 0 : i32, {{[^}]*}}}, {dfb_index = 1 : i32, {{[^}]*}}}]
// PRESERVE: ttl.logical_dfb_configs = [
// PRESERVE-SAME: {block_count = 2 : i32, compiler_allocated = false, element_type = !ttcore.tile<32x32, bf16>, elems_per_block = 1 : i32, epoch = 0 : i32, logical_index = 0 : i32, num_pages = 2 : i32, physical_index = 0 : i32, unpack_to_dest_fp32 = false}
// PRESERVE-SAME: , {block_count = 2 : i32, compiler_allocated = false, element_type = !ttcore.tile<32x32, f32>, elems_per_block = 1 : i32, epoch = 1 : i32, logical_index = 1 : i32, num_pages = 2 : i32, physical_index = 0 : i32, unpack_to_dest_fp32 = false}
// PRESERVE-SAME: , {block_count = 2 : i32, compiler_allocated = false, element_type = !ttcore.tile<32x32, bf16>, elems_per_block = 1 : i32, epoch = 0 : i32, logical_index = 2 : i32, num_pages = 2 : i32, physical_index = 1 : i32, unpack_to_dest_fp32 = false}
// PRESERVE-SAME: , {block_count = 2 : i32, compiler_allocated = false, element_type = !ttcore.tile<16x32, bf16>, elems_per_block = 1 : i32, epoch = 2 : i32, logical_index = 3 : i32, num_pages = 2 : i32, physical_index = 0 : i32, unpack_to_dest_fp32 = false}]
// PRESERVE-LABEL: func.func @preserve_across_one_boundary
// PRESERVE: ttl.bind_cb{cb_index = 0, block_count = 2} {ttl.dfb_logical_index = 0 : i64}
// PRESERVE: ttl.bind_cb{cb_index = 0, block_count = 2} {ttl.dfb_logical_index = 1 : i64}
// PRESERVE: ttl.bind_cb{cb_index = 1, block_count = 2} {ttl.dfb_logical_index = 2 : i64}
// PRESERVE: ttl.bind_cb{cb_index = 0, block_count = 2} {ttl.dfb_logical_index = 3 : i64}
// Prologue: configure the local DFB and pinned DFB.
// PRESERVE: ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [0, 2, 0, {{.*}}, 1, {{.*}}], ttl.dfb_reset_epoch = 0 : i32, ttl.dfb_reset_preserved_indices = []}
// First boundary: preserve physical DFB 1 and configure only local DFB 0.
// PRESERVE: ttl.opaque_call "ttlang::reset_dataflow_buffers"(%{{.*}}) {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [0, 1, 0, {{.*}}], ttl.dfb_reset_epoch = 1 : i32, ttl.dfb_reset_preserved_indices = [1]}
// Second boundary: reset and configure both physical DFBs.
// PRESERVE: ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h", template_args = [0, 2, 0, {{.*}}, 1, {{.*}}], ttl.dfb_reset_epoch = 2 : i32, ttl.dfb_reset_preserved_indices = []}
func.func @preserve_across_one_boundary()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 4 : i32,
                ttl.crta_indices = []} {
  %local0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %local1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %live = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %local2 = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<16x32, bf16>, 2>
  %r0 = ttl.cb_reserve %local0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %local0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %live_r = ttl.cb_reserve %live : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %live : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.opaque_call "ttlang::reset_dataflow_buffers"(%live) {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
  %live_w = ttl.cb_wait %live : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %live : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %r1 = ttl.cb_reserve %local1 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.cb_push %local1 : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.opaque_call "ttlang::reset_dataflow_buffers"() {header = "ttlang/Target/TTKernel/LLKs/reset_dataflow_buffers.h"} : () -> ()
  %r2 = ttl.cb_reserve %local2 : <[1, 1], !ttcore.tile<16x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<16x32, bf16>>
  ttl.cb_push %local2 : <[1, 1], !ttcore.tile<16x32, bf16>, 2>
  return
}
