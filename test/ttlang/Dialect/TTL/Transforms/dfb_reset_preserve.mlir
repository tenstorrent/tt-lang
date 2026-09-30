// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s

// CHECK-LABEL: func.func @reader
// CHECK: %[[PRESERVED:.*]] = ttl.bind_cb{cb_index = 0,
// CHECK: %[[STALE:.*]] = ttl.bind_cb{cb_index = 1,
// CHECK: %[[CURRENT:.*]] = ttl.bind_cb{cb_index = 1,
// CHECK: ttl.reset_all_dfbs {{.*}} preserve %[[PRESERVED]]

module attributes {
  ttl.launch_grid = [1, 1],
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "preserved_reset">,
                  ttl.noc_index = 0 : i32} {
    %preserved = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %stale = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %current = ttl.bind_cb {cb_index = 2, block_count = 1} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %preserved_slot = ttl.cb_reserve %preserved : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %preserved : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %stale_slot = ttl.cb_reserve %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %stale : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "preserved_reset">, <kind = data_movement, identity = "reader", operation = "preserved_reset">, <kind = data_movement, identity = "writer", operation = "preserved_reset">]> preserve %preserved : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %current_slot = ttl.cb_reserve %current : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %current : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }

  func.func @compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "preserved_reset">} {
    %preserved = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %current = ttl.bind_cb {cb_index = 2, block_count = 1} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "preserved_reset">, <kind = data_movement, identity = "reader", operation = "preserved_reset">, <kind = data_movement, identity = "writer", operation = "preserved_reset">]> preserve %preserved : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %preserved_slot = ttl.cb_wait %preserved : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %preserved : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %current_slot = ttl.cb_wait %current : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %current : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }

  func.func @writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "preserved_reset">,
                  ttl.noc_index = 1 : i32} {
    %preserved = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "preserved_reset">, <kind = data_movement, identity = "reader", operation = "preserved_reset">, <kind = data_movement, identity = "writer", operation = "preserved_reset">]> preserve %preserved : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
