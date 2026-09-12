// Tests that unknown external access does not apply to compiler-managed DFBs.
//
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-verify-dfb-spsc)'
// RUN: env TTL_RELAX_DFB_SPSC=1 ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-verify-dfb-spsc)'

module attributes {
  ttl.dfb_allocations = [{allocation_nodes = [[0, 0]], block_count = 2 : i32,
                          dfb_index = 0 : i32,
                          element_type = !ttcore.tile<32x32, bf16>,
                          num_tiles = 1 : i32, page_size = 2048 : i32}],
  ttl.launch_grid = [1, 1]
} {
  func.func @unknown_user_dfb_access()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    ttl.opaque_call "possible_user_dfb_producer" ()
        {header = "opaque.hpp", unknown_dfb_access} : () -> ()
    func.return
  }

  func.func @compiler_dfb_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-note @+1 {{dataflow buffer declared here}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 4 : index, ttl.compiler_allocated}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{logical DFB 4 is waited on but no kernel thread pushes it}}
    // expected-note @below {{a DFB wait blocks until a matching push publishes data}}
    %block = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}
