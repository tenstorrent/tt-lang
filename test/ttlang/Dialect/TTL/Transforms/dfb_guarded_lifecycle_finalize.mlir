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

// -----

// A guarded local release may complete a lifecycle after all storage uses in
// the acquiring branch have completed. Propagation-only yields do not extend
// the storage lifetime when their result is unused.

// CHECK-LABEL: func.func @guarded_local_release_after_use
// CHECK-SAME: ttl.base_cta_index = 1 : i32
// CHECK: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 1 : index}
module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @guarded_local_release_after_use(
      %arg0: tensor<1x1x!ttcore.tile<1x16, bf16>>)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %condition = arith.constant true
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>
    %view = scf.if %condition -> (tensor<1x1x!ttcore.tile<1x16, bf16>>) {
      %reserved = ttl.cb_reserve %first
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.store %arg0, %reserved
          : tensor<1x1x!ttcore.tile<1x16, bf16>>,
            tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.cb_push %first
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
      scf.yield %reserved : tensor<1x1x!ttcore.tile<1x16, bf16>>
    } else {
      %inactive = "builtin.unrealized_conversion_cast"()
          {ttl.inactive_guarded_dfb}
          : () -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      scf.yield %inactive : tensor<1x1x!ttcore.tile<1x16, bf16>>
    }
    scf.if %condition {
      %waited = ttl.cb_wait %first
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      %sum = ttl.add %waited, %arg0
          : tensor<1x1x!ttcore.tile<1x16, bf16>>,
            tensor<1x1x!ttcore.tile<1x16, bf16>>
          -> tensor<1x1x!ttcore.tile<1x16, bf16>>
      ttl.cb_pop %first
          : <[1, 1], !ttcore.tile<1x16, bf16>, 2>
    }
    ttl.opaque_call "second"
        dfb_dependencies(%second
            : !ttl.cb<[1, 1], !ttcore.tile<1x16, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>,
                     #ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
    return
  }
}
