// Tests for ttl-finalize-dfb-indices pass.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=CHECK
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=REUSE
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=OVERLAP
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=FOUR
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' -debug-only=ttl-finalize-dfb-indices 2>&1 | FileCheck %s --check-prefix=DEBUG
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=MIXED
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=UNUSED
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=THREE
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=SINGLE
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=GLOBAL
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s --check-prefix=NESTED

// -----

// User DFBs at indices 0, 1, 2 and a compiler-allocated DFB at index 3.
// The pass should update base_cta_index to 4 and emit ttl.compiler_allocated_dfbs.

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

// Non-overlapping compiler-allocated DFB lifecycles share index 3.

// DEBUG: DFB reuse: 2 compiler-allocated DFBs -> 1 physical slot(s)
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
  %reserve3 = ttl.cb_reserve %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %wait3 = ttl.cb_wait %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc4 = ttl.bind_cb {cb_index = 4, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve4 = ttl.cb_reserve %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %wait4 = ttl.cb_wait %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Overlapping compiler-allocated DFBs: DFB #4 is allocated while DFB #3
// is still live. They must keep separate indices.

// DEBUG: DFB reuse: 2 compiler-allocated DFBs -> 2 physical slot(s)
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
  %reserve3 = ttl.cb_reserve %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %alloc4 = ttl.bind_cb {cb_index = 4, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve4 = ttl.cb_reserve %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %wait4 = ttl.cb_wait %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_push %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %wait3 = ttl.cb_wait %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Four compiler-allocated DFBs with nested lifetimes (softmax pattern).
// DFB-A [reserve, pop]: spans past DFB-B
// DFB-B [reserve, pop]: nested within A, dies before A
// DFB-C [reserve, pop]: starts after A dies, spans past DFB-D
// DFB-D [reserve, pop]: nested within C, dies before C
// Result: A and C share slot 0 (index 3), B and D share slot 1 (index 4).

// DEBUG: DFB reuse: 4 compiler-allocated DFBs -> 2 physical slot(s)
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
  %reserveA = ttl.cb_reserve %allocA : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %allocB = ttl.bind_cb {cb_index = 4, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserveB = ttl.cb_reserve %allocB : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %allocB : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %waitB = ttl.cb_wait %allocB : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %allocB : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_push %allocA : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %waitA = ttl.cb_wait %allocA : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %allocA : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %allocC = ttl.bind_cb {cb_index = 5, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserveC = ttl.cb_reserve %allocC : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %allocD = ttl.bind_cb {cb_index = 6, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserveD = ttl.cb_reserve %allocD : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %allocD : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %waitD = ttl.cb_wait %allocD : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %allocD : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_push %allocC : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %waitC = ttl.cb_wait %allocC : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %allocC : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Mixed CircularBufferTypes: DFBs #3 and #5 are [1,1] bf16, DFB #4 is
// [2,4] bf16. Type partitioning prevents #3 and #4 from sharing a slot
// even though their lifetimes do not overlap. #3 and #5 share within
// the [1,1] partition. The [2,4] partition gets the next contiguous index.
//
// [1,1] partition: #3 non-overlapping with #5 -> 1 slot (index 3)
// [2,4] partition: #4 alone -> 1 slot (index 4)
// Total: 2 physical compiler-allocated slots.

// DEBUG: DFB reuse: 3 compiler-allocated DFBs -> 2 physical slot(s)
// DEBUG: Total DFB count: 5

// MIXED: module attributes {ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 3 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}, {block_count = 2 : i32, dfb_index = 4 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 8 : i32}]}

// MIXED-LABEL: func.func @mixed_types_no_cross_reuse
// MIXED-SAME: ttl.base_cta_index = 5 : i32
// [1,1] partition slot
// MIXED: ttl.bind_cb{cb_index = 3, {{.*}}} {ttl.compiler_allocated} : <[1, 1],
// [2,4] partition slot
// MIXED: ttl.bind_cb{cb_index = 4, {{.*}}} {ttl.compiler_allocated} : <[2, 4],
// [1,1] partition slot reused
// MIXED: ttl.bind_cb{cb_index = 3, {{.*}}} {ttl.compiler_allocated} : <[1, 1],
// MIXED-NOT: cb_index = 5
// MIXED: return
func.func @mixed_types_no_cross_reuse()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  // [1,1] DFB
  %alloc3 = ttl.bind_cb {cb_index = 3, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve3 = ttl.cb_reserve %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %wait3 = ttl.cb_wait %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  // [2,4] DFB -- different type, cannot reuse index 3
  %alloc4 = ttl.bind_cb {cb_index = 4, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 2>
  %reserve4 = ttl.cb_reserve %alloc4 : <[2, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %alloc4 : <[2, 4], !ttcore.tile<32x32, bf16>, 2>
  %wait4 = ttl.cb_wait %alloc4 : <[2, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %alloc4 : <[2, 4], !ttcore.tile<32x32, bf16>, 2>
  // [1,1] DFB -- same type as #3, reuses its slot
  %alloc5 = ttl.bind_cb {cb_index = 5, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve5 = ttl.cb_reserve %alloc5 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %alloc5 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %wait5 = ttl.cb_wait %alloc5 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %alloc5 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Unused compiler-allocated DFB declarations remain live for the entire
// kernel. No reuse is possible.

// DEBUG: DFB reuse: 2 compiler-allocated DFBs -> 2 physical slot(s)
// DEBUG: Total DFB count: 5

// UNUSED: module attributes {ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 3 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}, {block_count = 2 : i32, dfb_index = 4 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}]}

// UNUSED-LABEL: func.func @unused_declarations_conservative
// UNUSED-SAME: ttl.base_cta_index = 5 : i32
// UNUSED: ttl.bind_cb{cb_index = 3,
// UNUSED: ttl.bind_cb{cb_index = 4,
// UNUSED-NOT: cb_index = 5
// UNUSED: return
func.func @unused_declarations_conservative()
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

// DEBUG: DFB reuse: 3 compiler-allocated DFBs -> 1 physical slot(s)
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
  %reserve3 = ttl.cb_reserve %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %wait3 = ttl.cb_wait %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc4 = ttl.bind_cb {cb_index = 4, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve4 = ttl.cb_reserve %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %wait4 = ttl.cb_wait %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc5 = ttl.bind_cb {cb_index = 5, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve5 = ttl.cb_reserve %alloc5 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %alloc5 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %wait5 = ttl.cb_wait %alloc5 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %alloc5 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// A single compiler-allocated DFB is assigned the first physical index after
// the user-declared range.

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
  %reserve3 = ttl.cb_reserve %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %wait3 = ttl.cb_wait %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Passes operating on individual kernels may assign the same provisional
// compiler DFB index in sibling kernels. Finalization must assign disjoint
// physical indices after the highest user-declared index, including user
// indices in other kernels.

// GLOBAL: module attributes {ttl.compiler_allocated_dfbs = [{block_count = 2 : i32, dfb_index = 5 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}, {block_count = 2 : i32, dfb_index = 6 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}]}

// GLOBAL-LABEL: func.func @global_user_index
// GLOBAL-SAME: ttl.base_cta_index = 7 : i32
// GLOBAL: ttl.bind_cb{cb_index = 4,
func.func @global_user_index()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.base_cta_index = 5 : i32,
                ttl.crta_indices = []} {
  %user4 = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// GLOBAL-LABEL: func.func @first_provisional_index
// GLOBAL-SAME: ttl.base_cta_index = 7 : i32
// GLOBAL: ttl.bind_cb{cb_index = 5, {{.*}}} {ttl.compiler_allocated}
func.func @first_provisional_index()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 1 : i32,
                ttl.crta_indices = []} {
  %user0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %compiler1 = ttl.bind_cb {cb_index = 1, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve1 = ttl.cb_reserve %compiler1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %compiler1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %wait1 = ttl.cb_wait %compiler1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %compiler1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// GLOBAL-LABEL: func.func @second_provisional_index
// GLOBAL-SAME: ttl.base_cta_index = 7 : i32
// GLOBAL: ttl.bind_cb{cb_index = 6, {{.*}}} {ttl.compiler_allocated}
func.func @second_provisional_index()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 1 : i32,
                ttl.crta_indices = []} {
  %user0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %compiler1 = ttl.bind_cb {cb_index = 1, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve1 = ttl.cb_reserve %compiler1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %compiler1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %wait1 = ttl.cb_wait %compiler1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %compiler1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// A reserve before another DFB's pop inside the same region-bearing kernel-body
// operation makes both DFBs live concurrently. Projection gives both events
// the same ordinal, so the pop endpoint must include that ordinal.

// NESTED: module attributes {ttl.compiler_allocated_dfbs = [{block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}, {block_count = 1 : i32, dfb_index = 1 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32}]}
// NESTED-LABEL: func.func @nested_lifecycles_overlap
// NESTED-SAME: ttl.base_cta_index = 2 : i32
// NESTED: ttl.bind_cb{cb_index = 0,
// NESTED: ttl.bind_cb{cb_index = 1,
// NESTED: return
func.func @nested_lifecycles_overlap()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 0 : i32} {
  %lower = arith.constant 0 : index
  %upper = arith.constant 1 : index
  %step = arith.constant 1 : index
  %first = ttl.bind_cb {cb_index = 0, block_count = 1}
      {ttl.compiler_allocated}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %second = ttl.bind_cb {cb_index = 1, block_count = 1}
      {ttl.compiler_allocated}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %first_reserve = ttl.cb_reserve %first
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %first_wait = ttl.cb_wait %first
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  scf.for %iteration = %lower to %upper step %step {
    %second_reserve = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second_wait = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  }
  ttl.cb_pop %second : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  return
}
