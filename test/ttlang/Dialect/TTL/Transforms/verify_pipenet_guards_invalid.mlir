// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -ttl-verify-pipenet-guards

// Summary: Negative tests for ttl-verify-pipenet-guards diagnostics.

// A DFB-to-pipe copy must execute only on the pipe source node.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unguarded_source_copy() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @below {{PipeNet 0 declared here}}
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{may copy to a pipe outside that pipe's source role}}
    // expected-note @below {{example node where the guard does not hold: core_x=1}}
    // expected-note @below {{suggested guard: `net_0.is_src()`}}
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
    // expected-note @below {{PipeNet 0 declared here}}
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{may copy from a pipe outside that pipe's destination role}}
    // expected-note @below {{example node where the guard does not hold: core_x=0}}
    // expected-note @below {{suggested guard: `net_0.is_dst()`}}
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
    // expected-note @below {{PipeNet 0 declared here}}
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    // expected-error @below {{PipeNet scope may execute outside its declared role domain}}
    // expected-note @below {{example node where the guard does not hold: core_x=1}}
    // expected-note @below {{suggested guard: `net_0.is_src()`}}
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
  func.func @unsupported_predicate(%runtime: index) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %scaled = arith.muli %core_x, %runtime : index
    %zero = arith.constant 0 : index
    // expected-note @below {{this expression is not statically analyzable}}
    %cond = arith.cmpi eq, %scaled, %zero : index
    scf.if %cond {
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
    // expected-error @below {{no producer pushes to DFB index 0}}
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
    // expected-note @below {{example node where the guard does not hold: core_x=1}}
    %view = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Cross-PipeNet: copy targets net_a from a region whose only guard is for
// net_b. Diagnostic must name net_a, not "the active set".

module attributes {ttl.launch_grid = [4 : i64, 4 : i64]} {
  func.func @cross_net_guard() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @below {{PipeNet 0 declared here}}
    %pa = ttl.create_pipe src(0, 0) dst(0, 1) to(0, 3) net 0
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 3) net 0>
    %pb = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 1
        : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 1>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cond = ttl.is_dst {pipe_net_id = 1 : i64}
    scf.if %cond {
      // expected-error @below {{may copy from a pipe outside that pipe's destination role}}
      // expected-note @below {{example node where the guard does not hold:}}
      // expected-note @below {{suggested guard: `net_0.is_dst()`}}
      %r = ttl.copy %pa, %cb
          : (!ttl.pipe<src(0, 0) dst(0, 1) to(0, 3) net 0>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> !ttl.transfer_handle
    }
    func.return
  }
}

// -----

// Missing ttl.launch_grid attribute is a hard error.

// expected-error @below {{ttl-verify-pipenet-guards requires a `ttl.launch_grid` module attribute}}
module {
  func.func @no_launch_grid() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    func.return
  }
}

// -----

// affine.if guard insufficient: the IntegerSet covers nodes outside the
// pipe's source role.

// Nested `is_active`s narrow the domain by intersection. The intersection
// of net_a (col 0) and net_b (row 0) is {(0, 0)}, which is net_a's source.
// A copy from a pipe (i.e. expecting a destination role) is rejected
// because (0, 0) is outside net_a's destination range.

module attributes {ttl.launch_grid = [4 : i64, 4 : i64]} {
  func.func @nested_is_active_misses_role() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @below {{PipeNet 0 declared here}}
    %pa = ttl.create_pipe src(0, 0) dst(0, 1) to(0, 3) net 0
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 3) net 0>
    %pb = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 1
        : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 1>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %a_active = ttl.is_active {pipe_net_id = 0 : i64}
    scf.if %a_active {
      %b_active = ttl.is_active {pipe_net_id = 1 : i64}
      scf.if %b_active {
        // expected-error @below {{may copy from a pipe outside that pipe's destination role}}
        // expected-note @below {{example node where the guard does not hold: core_x=0}}
        // expected-note @below {{suggested guard: `net_0.is_dst()`}}
        %recv = ttl.copy %pa, %cb
            : (!ttl.pipe<src(0, 0) dst(0, 1) to(0, 3) net 0>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            -> !ttl.transfer_handle
      }
    }
    func.return
  }
}

// -----

#wideSet = affine_set<(d0) : (d0 - 4 >= 0)>
module attributes {ttl.launch_grid = [8 : i64, 1 : i64]} {
  func.func @affine_if_too_wide() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @below {{PipeNet 0 declared here}}
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    affine.if #wideSet(%x) {
      // expected-error @below {{may copy to a pipe outside that pipe's source role}}
      // expected-note @below {{example node where the guard does not hold: core_x=4}}
      // expected-note @below {{suggested guard: `net_0.is_src()`}}
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}
