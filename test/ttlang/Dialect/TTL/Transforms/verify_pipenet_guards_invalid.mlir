// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -ttl-verify-pipenet-guards

// Summary: Negative tests for ttl-verify-pipenet-guards diagnostics.

// A DFB-to-pipe copy must execute only on the pipe source node.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unguarded_source_copy() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{may copy to a pipe outside that pipe's source role}}
    %send = ttl.copy %cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    func.return
  }
}

// -----

// A pipe-to-DFB copy must execute only on pipe destination nodes.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unguarded_destination_copy() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{may copy from a pipe outside that pipe's destination role}}
    %recv = ttl.copy %pipe, %cb
        : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> !ttl.transfer_handle
    func.return
  }
}

// -----

// The scope predicate must be contained in the declared role domain.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unguarded_scope() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    // expected-error @below {{PipeNet scope may execute outside its declared role domain}}
    ttl.pipenet_scope attributes {ttl.pipe_net_ids = [0 : i64], ttl.pipe_net_roles = [0 : i64]} {
      ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      }
    }
    func.return
  }
}

// -----

// Unsupported predicates are rejected instead of treated as valid guards.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unsupported_predicate(%flag: i1) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    scf.if %flag {
      // expected-error @below {{cannot prove PipeNet guard condition}}
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Waiting on a DFB with no producer domain is rejected.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @wait_without_producer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{has no producer domain for DFB index 0}}
    %view = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// A wait whose execution domain is broader than the producer domain is rejected.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @producer() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 7, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    func.return
  }

  func.func @consumer() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 7, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{may wait on a DFB from nodes where no producer pushes it}}
    %view = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}
