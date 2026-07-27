// Tests SPSC verification after compiler-only DFB index finalization.
//
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=false},ttl-verify-dfb-spsc)' | FileCheck %s

// Distinct compiler-created logical DFBs may reuse one physical index without
// becoming one SPSC participant set.

// CHECK: module attributes {ttl.dfb_allocations = [
// CHECK-SAME: {block_count = 2 : i32, dfb_index = 0 : i32,
// CHECK-LABEL: func.func @sequential_compiler_dfbs
// CHECK-SAME: ttl.base_cta_index = 1 : i32
// CHECK: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, {{.*}} {dfb_id = 0 : index, ttl.compiler_allocated}
// CHECK: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 0, {{.*}} {dfb_id = 1 : index, ttl.compiler_allocated}

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @sequential_compiler_dfbs()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {ttl.compiler_allocated}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {ttl.compiler_allocated}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

    %first_reserved = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_waited = ttl.cb_wait %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>

    %second_reserved = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_waited = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
