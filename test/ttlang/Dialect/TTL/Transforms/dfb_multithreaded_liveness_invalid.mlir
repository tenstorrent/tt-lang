// Tests invalid logical DFB declarations for multithreaded liveness analysis.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})'

// One logical DFB must have one exact type across all thread functions.

func.func @type_mismatch_producer()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  return
}

func.func @type_mismatch_consumer()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  // expected-error @below {{logical DFB 0 has inconsistent types across thread functions}}
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  return
}
