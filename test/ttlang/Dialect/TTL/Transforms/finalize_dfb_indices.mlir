// Tests for ttl-finalize-dfb-indices pass.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})' | FileCheck %s --check-prefix=CHECK
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})' | FileCheck %s --check-prefix=REUSE
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})' | FileCheck %s --check-prefix=OVERLAP
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})' | FileCheck %s --check-prefix=FOUR
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})' | FileCheck %s --check-prefix=MIXED
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})' | FileCheck %s --check-prefix=NOPOP
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})' | FileCheck %s --check-prefix=THREE
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})' | FileCheck %s --check-prefix=SINGLE
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})' | FileCheck %s --check-prefix=GLOBAL

// -----

// User DFBs at indices 0, 1, 2 and a compiler-allocated DFB at index 3.
// The pass should update base_cta_index to 4 and emit the complete allocation
// table.

// CHECK: module attributes {ttl.dfb_allocations = {{.*}}}

// CHECK-LABEL: func.func @reader
// CHECK-SAME: ttl.base_cta_index = 4 : i32
func.func @reader()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = [0 : i32, 1 : i32]} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// CHECK-LABEL: func.func @compute
// CHECK-SAME: ttl.base_cta_index = 4 : i32
func.func @compute()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc = ttl.bind_cb {cb_index = 3, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// CHECK-LABEL: func.func @writer
// CHECK-SAME: ttl.base_cta_index = 4 : i32
func.func @writer()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = [0 : i32, 1 : i32]} {
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// With no compiler-allocated DFBs, the pass still updates base_cta_index to
// the true DFB count (3).

// CHECK: module attributes {ttl.dfb_allocations = {{.*}}}

// CHECK-LABEL: func.func @compute_only
// CHECK-SAME: ttl.base_cta_index = 3 : i32
func.func @compute_only()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 2 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Non-overlapping compiler-allocated DFBs: DFB #3 is released (cb_pop)
// before DFB #4 is allocated (bind_cb). Both should be assigned index 3.

// REUSE: module attributes {ttl.dfb_allocations = {{.*}}}

// REUSE-LABEL: func.func @non_overlapping_reuse
// REUSE-SAME: ttl.base_cta_index = 4 : i32
// REUSE-COUNT-2: ttl.bind_cb{cb_index = 3,
// REUSE-NOT: cb_index = 4
// REUSE: return
func.func @non_overlapping_reuse()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc3 = ttl.bind_cb {cb_index = 3, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc4 = ttl.bind_cb {cb_index = 4, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Overlapping compiler-allocated DFBs: DFB #4 is allocated while DFB #3
// is still live. They must keep separate indices.

// OVERLAP: module attributes {ttl.dfb_allocations = {{.*}}}

// OVERLAP-LABEL: func.func @overlapping_no_reuse
// OVERLAP-SAME: ttl.base_cta_index = 5 : i32
// OVERLAP: ttl.bind_cb{cb_index = 3,
// OVERLAP: ttl.bind_cb{cb_index = 4,
// OVERLAP-NOT: cb_index = 5
// OVERLAP: return
func.func @overlapping_no_reuse()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
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

// FOUR: module attributes {ttl.dfb_allocations = {{.*}}}

// FOUR-LABEL: func.func @four_dfbs_nested_reuse
// FOUR-SAME: ttl.base_cta_index = 5 : i32
//
// DFB-A -> slot 0 (index 3)
// FOUR: ttl.bind_cb{cb_index = 3, {{.*}}} {dfb_id = 3 : index, ttl.compiler_allocated}
// DFB-B -> slot 1 (index 4)
// FOUR: ttl.bind_cb{cb_index = 4, {{.*}}} {dfb_id = 4 : index, ttl.compiler_allocated}
// FOUR: ttl.cb_pop
// FOUR: ttl.cb_pop
// DFB-C -> slot 0 (index 3, reused from A)
// FOUR: ttl.bind_cb{cb_index = 3, {{.*}}} {dfb_id = 5 : index, ttl.compiler_allocated}
// DFB-D -> slot 1 (index 4, reused from B)
// FOUR: ttl.bind_cb{cb_index = 4, {{.*}}} {dfb_id = 6 : index, ttl.compiler_allocated}
// FOUR: ttl.cb_pop
// FOUR: ttl.cb_pop
// FOUR-NOT: cb_index = 5
// FOUR-NOT: cb_index = 6
// FOUR: return
func.func @four_dfbs_nested_reuse()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
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

// Mixed CircularBufferTypes: DFBs #3 and #5 are [1,1] bf16, DFB #4 is
// [2,4] bf16. Type partitioning prevents #3 and #4 from sharing a slot
// even though their lifetimes do not overlap. #3 and #5 share within
// the [1,1] partition. The [2,4] partition gets the next contiguous index.
//
// [1,1] partition: #3 non-overlapping with #5 -> 1 slot (index 3)
// [2,4] partition: #4 alone -> 1 slot (index 4)
// Total: 2 physical compiler-allocated slots.

// MIXED: module attributes {ttl.dfb_allocations = {{.*}}}

// MIXED-LABEL: func.func @mixed_types_no_cross_reuse
// MIXED-SAME: ttl.base_cta_index = 5 : i32
// [1,1] partition slot
// MIXED: ttl.bind_cb{cb_index = 3, {{.*}}} {dfb_id = 3 : index, ttl.compiler_allocated} : <[1, 1],
// [2,4] partition slot
// MIXED: ttl.bind_cb{cb_index = 4, {{.*}}} {dfb_id = 4 : index, ttl.compiler_allocated} : <[2, 4],
// [1,1] partition slot reused
// MIXED: ttl.bind_cb{cb_index = 3, {{.*}}} {dfb_id = 5 : index, ttl.compiler_allocated} : <[1, 1],
// MIXED-NOT: cb_index = 5
// MIXED: return
func.func @mixed_types_no_cross_reuse()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  // [1,1] DFB
  %alloc3 = ttl.bind_cb {cb_index = 3, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  // [2,4] DFB -- different type, cannot reuse index 3
  %alloc4 = ttl.bind_cb {cb_index = 4, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc4 : <[2, 4], !ttcore.tile<32x32, bf16>, 2>
  // [1,1] DFB -- same type as #3, reuses its slot
  %alloc5 = ttl.bind_cb {cb_index = 5, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc5 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// No cb_pop on either compiler-allocated DFB. Conservative fallback
// treats both as live for the entire function. No reuse possible.

// NOPOP: module attributes {ttl.dfb_allocations = {{.*}}}

// NOPOP-LABEL: func.func @no_cb_pop_conservative
// NOPOP-SAME: ttl.base_cta_index = 5 : i32
// NOPOP: ttl.bind_cb{cb_index = 3,
// NOPOP: ttl.bind_cb{cb_index = 4,
// NOPOP-NOT: cb_index = 5
// NOPOP: return
func.func @no_cb_pop_conservative()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc3 = ttl.bind_cb {cb_index = 3, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc4 = ttl.bind_cb {cb_index = 4, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Three sequential non-overlapping DFBs. All should map to a single
// physical slot (multi-round slot recycling).

// THREE: module attributes {ttl.dfb_allocations = {{.*}}}

// THREE-LABEL: func.func @three_sequential_one_slot
// THREE-SAME: ttl.base_cta_index = 4 : i32
// THREE-COUNT-3: ttl.bind_cb{cb_index = 3,
// THREE-NOT: cb_index = 4
// THREE: return
func.func @three_sequential_one_slot()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc3 = ttl.bind_cb {cb_index = 3, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc4 = ttl.bind_cb {cb_index = 4, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc5 = ttl.bind_cb {cb_index = 5, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc5 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// A single compiler-allocated DFB is assigned the first physical index after
// the user-declared range.

// SINGLE: module attributes {ttl.dfb_allocations = {{.*}}}

// SINGLE-LABEL: func.func @single_dfb_no_reuse
// SINGLE-SAME: ttl.base_cta_index = 4 : i32
// SINGLE: ttl.bind_cb{cb_index = 3, {{.*}}} {dfb_id = 3 : index, ttl.compiler_allocated}
// SINGLE-NOT: cb_index = 4
// SINGLE: return
func.func @single_dfb_no_reuse()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 3 : i32,
                ttl.crta_indices = []} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %alloc3 = ttl.bind_cb {cb_index = 3, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Function passes may assign the same provisional compiler DFB index in
// sibling functions. Finalization must assign disjoint physical indices after
// the highest user-declared index, including user indices in other functions.

// GLOBAL: module attributes {ttl.dfb_allocations = {{.*}}}

// GLOBAL-LABEL: func.func @global_user_index
// GLOBAL-SAME: ttl.base_cta_index = 7 : i32
// GLOBAL: ttl.bind_cb{cb_index = 4,
func.func @global_user_index()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.base_cta_index = 5 : i32,
                ttl.crta_indices = []} {
  %user4 = ttl.bind_cb {cb_index = 4, block_count = 2} {dfb_id = 4 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// GLOBAL-LABEL: func.func @first_provisional_index
// GLOBAL-SAME: ttl.base_cta_index = 7 : i32
// GLOBAL: ttl.bind_cb{cb_index = 5, {{.*}}} {dfb_id = 5 : index, ttl.compiler_allocated}
func.func @first_provisional_index()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 1 : i32,
                ttl.crta_indices = []} {
  %user0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %compiler1 = ttl.bind_cb {cb_index = 1, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %compiler1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

// GLOBAL-LABEL: func.func @second_provisional_index
// GLOBAL-SAME: ttl.base_cta_index = 7 : i32
// GLOBAL: ttl.bind_cb{cb_index = 6, {{.*}}} {dfb_id = 6 : index, ttl.compiler_allocated}
func.func @second_provisional_index()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 1 : i32,
                ttl.crta_indices = []} {
  %user0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %compiler1 = ttl.bind_cb {cb_index = 1, block_count = 2} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %compiler1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}
