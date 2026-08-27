// Tests that finalization separates conflicting provisional aliases before
// SPSC verification.
//
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true},ttl-verify-dfb-spsc)' | FileCheck %s

// The two producer functions execute concurrently, so their distinct logical
// DFBs cannot retain the same provisional physical index.

// CHECK: module attributes {ttl.dfb_allocations = [
// CHECK-SAME: {allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 2 : i32, dfb_index = 0 : i32,
// CHECK-LABEL: func.func @producer_a
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, {{.*}} {dfb_id = 0 : index}
// CHECK-LABEL: func.func @producer_b
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 1, {{.*}} {dfb_id = 1 : index}

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @producer_a()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_reserved = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    return
  }

  func.func @producer_b()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %second = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_reserved = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    return
  }
}
