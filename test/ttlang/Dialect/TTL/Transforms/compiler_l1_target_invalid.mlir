// Verifies that Wormhole retains compiler-managed SRAM support while rejecting Blackhole-only lifecycle boundaries.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1})'

module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  func.func @unsupported_reset_target() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{'ttl.reset_dfbs' op is supported only for Blackhole; selected target is #ttcore.arch<wormhole_b0>}}
    ttl.reset_dfbs <0, participants[<kind = compute, identity = "compute", operation = "reset_test">, <kind = data_movement, identity = "reader", operation = "reset_test">, <kind = data_movement, identity = "writer", operation = "reset_test">]>(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reconfiguration_test">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reconfiguration_test">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reconfiguration_test">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  func.func @unsupported_reconfiguration_target() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #compute} {
    // expected-error @below {{'ttl.dfb_reconfiguration' op is supported only for Blackhole; selected target is #ttcore.arch<wormhole_b0>}}
    ttl.dfb_reconfiguration #boundary
    return
  }
}
