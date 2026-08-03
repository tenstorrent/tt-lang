// Verify that final DFB allocation rejects incomplete compiler-created
// lifecycles before rewriting physical indices.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)'

// A producer lifecycle requires a reserve before its push.
func.func @missing_reserve()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  // expected-error @below {{'ttl.bind_cb' op compiler-allocated DFB has a partial lifecycle: missing ttl.cb_reserve}}
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  return
}

// -----

// A producer lifecycle requires a push after its reserve.
func.func @missing_push()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  // expected-error @below {{'ttl.bind_cb' op compiler-allocated DFB has a partial lifecycle: missing ttl.cb_push}}
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %reserve = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %wait = ttl.cb_wait %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  return
}

// -----

// A consumer lifecycle requires a wait before its pop.
func.func @missing_wait()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  // expected-error @below {{'ttl.bind_cb' op compiler-allocated DFB has a partial lifecycle: missing ttl.cb_wait}}
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %reserve = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  return
}

// -----

// A consumer lifecycle requires a pop after its wait.
func.func @missing_pop()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  // expected-error @below {{'ttl.bind_cb' op compiler-allocated DFB has a partial lifecycle: missing ttl.cb_pop}}
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {ttl.compiler_allocated} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %reserve = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %wait = ttl.cb_wait %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}
