// Tests allocation-group reuse for ordered nested accesses that execute once.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s

// Both logical dataflow buffers use physical index zero because the static
// one-iteration loop preserves the order of its nested synchronous accesses.
// CHECK-LABEL: func.func @sequential_conditional_inspect
// CHECK: %[[FIRST:.*]] = ttl.bind_cb{cb_index = 0,
// CHECK-NEXT: %[[SECOND:.*]] = ttl.bind_cb{cb_index = 0,
// CHECK: ttl.opaque_call "inspect_first" dfb_dependencies(%[[FIRST]]
// CHECK: ttl.opaque_call "inspect_second" dfb_dependencies(%[[SECOND]]

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @sequential_conditional_inspect()
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
    %core_x = ttl.core_x : index
    %one = arith.constant 1 : index
    %active = arith.cmpi slt, %core_x, %one : index
    %zero = arith.constant 0 : index
    scf.for %iteration = %zero to %one step %one {
      scf.if %active {
        ttl.opaque_call "inspect_first"
            dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>)
            dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 0>]
            () {header = "inspect.hpp"} : () -> ()
      }
      scf.if %active {
        ttl.opaque_call "inspect_second"
            dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>)
            dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 0>]
            () {header = "inspect.hpp"} : () -> ()
      }
    }
    return
  }
}
