// Tests ttl-finalize-dfb-indices with conditionally acquired dataflow buffers.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s

// A guarded acquire that yields an inactive value on the else branch uses the
// parent scf.if as its ordering point. A sibling release under the same
// condition must prove concrete lifecycle order during physical allocation.

// CHECK: module attributes {ttl.dfb_allocations = [{{.*}}dfb_index = 0 : i32
// CHECK-LABEL: func.func @guarded_producer
// CHECK: scf.if %[[CONDITION:.*]] -> (tensor
// CHECK: ttl.cb_reserve
// CHECK: scf.if %[[CONDITION]] {
// CHECK-NEXT: ttl.cb_push
// CHECK-LABEL: func.func @guarded_consumer
// CHECK: scf.if %[[CONDITION:.*]] -> (tensor
// CHECK: ttl.cb_wait
// CHECK: scf.if %[[CONDITION]] {
// CHECK-NEXT: ttl.cb_pop
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @guarded_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 1 : i32,
                  ttl.crta_indices = []} {
    %condition = arith.constant true
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %view = scf.if %condition -> (tensor<1x1x!ttcore.tile<1x16, bf16>>) {
      %reserved = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      scf.yield %reserved : tensor<1x1x!ttcore.tile<1x16, bf16>>
    } else {
      %inactive = "builtin.unrealized_conversion_cast"()
          {ttl.inactive_guarded_dfb}
          : () -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      scf.yield %inactive : tensor<1x1x!ttcore.tile<1x16, bf16>>
    }
    scf.if %condition {
      ttl.cb_push %dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    }
    return
  }

  func.func @guarded_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 1 : i32,
                  ttl.crta_indices = []} {
    %condition = arith.constant true
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %view = scf.if %condition -> (tensor<1x1x!ttcore.tile<1x16, bf16>>) {
      %waited = ttl.cb_wait %dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      scf.yield %waited : tensor<1x1x!ttcore.tile<1x16, bf16>>
    } else {
      %inactive = "builtin.unrealized_conversion_cast"()
          {ttl.inactive_guarded_dfb}
          : () -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      scf.yield %inactive : tensor<1x1x!ttcore.tile<1x16, bf16>>
    }
    scf.if %condition {
      ttl.cb_pop %dfb
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    }
    return
  }
}
