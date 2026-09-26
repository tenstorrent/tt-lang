// Verifies the compiler-managed backend rejects synchronization before allocation.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-sram})'

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">
#reset = #ttl.synchronized_dfb_reset<0, participants[#compute, #reader, #writer]>

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @reset() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #compute} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{'ttl.reset_dfbs' op compiler-sram does not support synchronized DFB reset or reconfiguration}}
    ttl.reset_dfbs #reset(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    return
  }
}

// -----

#all_compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_all_test">
#all_reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_all_test">
#all_writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_all_test">
#all_reset = #ttl.synchronized_dfb_reset<0, participants[#all_compute, #all_reader, #all_writer]>

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @reset_all() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #all_compute} {
    // expected-error @below {{'ttl.reset_all_dfbs' op compiler-sram does not support synchronized DFB reset or reconfiguration}}
    ttl.reset_all_dfbs #all_reset
    return
  }
}

// -----

#reconfig_compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reconfiguration_test">
#reconfig_reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reconfiguration_test">
#reconfig_writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reconfiguration_test">
#boundary = #ttl.dfb_reconfiguration<0, participants[#reconfig_compute, #reconfig_reader, #reconfig_writer]>

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @reconfiguration() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #reconfig_compute} {
    // expected-error @below {{'ttl.dfb_reconfiguration' op compiler-sram does not support synchronized DFB reset or reconfiguration}}
    ttl.dfb_reconfiguration #boundary
    return
  }
}
