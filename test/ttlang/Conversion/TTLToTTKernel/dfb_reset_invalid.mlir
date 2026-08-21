// Verifies reset lowering rejects unsupported targets and invalid allocation.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --convert-ttl-to-ttkernel

module {
  func.func @missing_target()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{'ttl.reset_dfbs' op requires a resolved target architecture; synchronized DFB reset is supported only for Blackhole}}
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    return
  }
}

// -----

module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  func.func @unsupported_target()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{'ttl.reset_dfbs' op is supported only for Blackhole; selected target is #ttcore.arch<wormhole_b0>}}
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    return
  }
}

// -----

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @out_of_range_index()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{'ttl.bind_cb' op finalized DFB index 64 is outside [0, 63] for the 64-DFB-index Blackhole target capacity}}
    %dfb = ttl.bind_cb {cb_index = 64, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    return
  }
}
