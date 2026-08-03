// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -ttl-verify-pipenet-guards

// Summary: Negative tests for logical DFB identity preconditions.

// A user-declared DFB must contain its module-wide logical ID before physical
// allocation.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @missing_logical_id()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    // expected-error @below {{`ttl-verify-pipenet-guards` requires valid logical DFB identities: user-declared DFB requires dfb_id before physical allocation}}
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.return
  }
}

// -----

// A DFB lifecycle operand must resolve to a DFB declaration.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unresolved_dfb_operand(
      %dfb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    // expected-error @below {{requires a DFB operand defined by `ttl.bind_cb`}}
    %block = ttl.cb_wait %dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}
