// Tests rejection when repeated nested access runs overlap.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)'

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @repeated_conditional_inspect()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>
    // expected-error @below {{cannot alias logical DFBs 0 and 1: concurrent-lifetime}}
    %second = ttl.bind_cb {cb_index = 1, block_count = 1}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<8x32, bf16>, 1>
    %core_x = ttl.core_x : index
    %one = arith.constant 1 : index
    %active = arith.cmpi slt, %core_x, %one : index
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    scf.for %iteration = %zero to %two step %one {
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
