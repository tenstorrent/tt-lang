// Summary: Verifies PipeNet endpoint checks remain enabled in relaxed mode.

// RUN: env TTL_RELAX_DFB_SPSC=1 ttlang-opt %s --split-input-file --verify-diagnostics -ttl-verify-pipenet-guards

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unguarded_pipe_source()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @below {{PipeNet net_0 declared here}}
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{this `ttl.copy(buffer, pipe)` sends data on PipeNet net_0 from a node that is not a source}}
    // expected-note @below {{example node where the guard does not hold: core_x=1, core_y=0}}
    %send = ttl.copy %dfb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    func.return
  }
}
