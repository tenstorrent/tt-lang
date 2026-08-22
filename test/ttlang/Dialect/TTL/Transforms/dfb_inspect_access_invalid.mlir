// Tests that an external DFB dependency remains conservative without an
// explicit non-transactional access contract.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})'

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @missing_inspect_access()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %descriptor = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{'ttl.bind_cb' op DFB allocation group #ttl.dfb_allocation_group<0> members=[0, 1] cannot alias logical DFBs 0 and 1: access-completion-not-proven}}
    %queue = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "inspect"
        dfb_dependencies(%descriptor : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        () {header = "inspect.hpp"} : () -> ()
    %produced = ttl.cb_reserve %queue
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %queue : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %consumed = ttl.cb_wait %queue
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %queue : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// A state-preserving inspection does not make concurrent storage access safe.
// The writer may begin waiting on the queue while compute uses the descriptor.
module attributes {ttl.launch_grid = [1, 1]} {
  func.func @concurrent_descriptor_compute()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %descriptor = ttl.bind_cb {cb_index = 0, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{'ttl.bind_cb' op DFB allocation group #ttl.dfb_allocation_group<1> members=[2, 3] cannot alias logical DFBs 2 and 3: concurrent-lifetime}}
    %queue = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "inspect"
        dfb_dependencies(%descriptor : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 0>]
        () {header = "inspect.hpp"} : () -> ()
    %produced = ttl.cb_reserve %queue
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %queue : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @concurrent_queue_writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 2 : i32,
                  ttl.crta_indices = []} {
    %queue = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<1>, dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %consumed = ttl.cb_wait %queue
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %queue : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
