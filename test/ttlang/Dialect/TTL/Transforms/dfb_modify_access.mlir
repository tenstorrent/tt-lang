// Tests allocation-group reuse for ordered queue lifecycles whose external
// calls modify producer-owned storage synchronously.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s

// Both logical dataflow buffers use physical index zero because each
// reserve/modify/push/wait/pop lifecycle completes before the next begins.
// CHECK-LABEL: func.func @sequential_modify_lifecycles
// CHECK: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0,
// CHECK: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 0,
// CHECK: ttl.opaque_call "modify_first" dfb_dependencies(%[[FIRST]]
// CHECK: ttl.opaque_call "modify_second" dfb_dependencies(%[[SECOND]]

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @sequential_modify_lifecycles()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 1}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>
    %first_produced = ttl.cb_reserve %first
        : <[1, 1], !ttcore.tile<8x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<8x32, bf16>>
    ttl.opaque_call "modify_first"
        dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>)
        dfb_accesses [#ttl.dfb_non_transactional_access<modify, 0>]
        () {header = "modify.hpp"} : () -> ()
    ttl.cb_push %first : <[1, 1], !ttcore.tile<8x32, bf16>, 1>
    %first_consumed = ttl.cb_wait %first
        : <[1, 1], !ttcore.tile<8x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<8x32, bf16>>
    ttl.cb_pop %first : <[1, 1], !ttcore.tile<8x32, bf16>, 1>

    %second_produced = ttl.cb_reserve %second
        : <[1, 1], !ttcore.tile<8x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<8x32, bf16>>
    ttl.opaque_call "modify_second"
        dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>)
        dfb_accesses [#ttl.dfb_non_transactional_access<modify, 0>]
        () {header = "modify.hpp"} : () -> ()
    ttl.cb_push %second : <[1, 1], !ttcore.tile<8x32, bf16>, 1>
    %second_consumed = ttl.cb_wait %second
        : <[1, 1], !ttcore.tile<8x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<8x32, bf16>>
    ttl.cb_pop %second : <[1, 1], !ttcore.tile<8x32, bf16>, 1>
    return
  }
}
