// Verifies direct conversion also rejects unsupported compiler-managed synchronization.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=false})'

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">
#reset = #ttl.synchronized_dfb_reset<0, participants[#compute, #reader, #writer]>

module attributes {ttl.memory_model = "compiler-sram", ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @reset() attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #compute} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // expected-error @below {{'ttl.reset_dfbs' op compiler-sram does not support synchronized DFB reset or reconfiguration}}
    ttl.reset_dfbs #reset(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    return
  }
}
