// Tests debug reporting of the complete logical-to-physical DFB assignment.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false})' -debug-only=ttl-finalize-dfb-indices 2>&1 | FileCheck %s

// CHECK: Total DFB count: 4
// CHECK-NEXT: DFB assignment: logical DFB 0 -> physical index 0
// CHECK-NEXT: DFB assignment: logical DFB 1 -> physical index 1
// CHECK-NEXT: DFB assignment: logical DFB 2 -> physical index 2
// CHECK-NEXT: DFB assignment: logical DFB 3 -> physical index 3
// CHECK-NEXT: DFB assignment: logical DFB 4 -> physical index 3

module {
  func.func @non_overlapping_reuse()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
    %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %alloc3 = ttl.bind_cb {cb_index = 3, block_count = 2}
        {ttl.compiler_allocated}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserve3 = ttl.cb_reserve %alloc3
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %wait3 = ttl.cb_wait %alloc3
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %alloc3 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %alloc4 = ttl.bind_cb {cb_index = 4, block_count = 2}
        {ttl.compiler_allocated}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserve4 = ttl.cb_reserve %alloc4
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %wait4 = ttl.cb_wait %alloc4
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %alloc4 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
