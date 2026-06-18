// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -ttl-verify-pipenet-guards

// Summary: Negative tests for ttl-verify-pipenet-guards diagnostics.

// A DFB-to-pipe copy must execute only on the pipe source node.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unguarded_source_copy() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @below {{PipeNet net_0 declared here}}
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{this `ttl.copy(buffer, pipe)` sends data on PipeNet net_0 from a node that is not a source}}
    // expected-note @below {{example node where the guard does not hold: core_x=1}}
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
    // expected-note @below {{PipeNet net_0 declared here}}
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_reserve = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %recv_view = ttl.attach_cb %recv_reserve, %cb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-error @below {{this `ttl.copy(pipe, buffer)` receives data from PipeNet net_0 on a node that is not a destination}}
    // expected-note @below {{example node where the guard does not hold: core_x=0}}
    %recv = ttl.copy %pipe, %recv_view
        : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.transfer_handle
    func.return
  }
}

// -----

// The scope predicate must be contained in the declared role domain.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unguarded_scope() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @below {{PipeNet net_0 declared here}}
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    // expected-error @below {{this region exchanges data on PipeNet}}
    // expected-note @below {{example node where the guard does not hold: core_x=1}}
    ttl.pipenet_scope attributes {ttl.pipe_net_ids = [0 : i64], ttl.pipe_net_roles = [0 : i64]} {
      ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      }
    }
    func.return
  }
}

// -----

// Soundness: a uniform-unknown predicate (a runtime flag, not coord-dependent)
// must not let the else-branch's domain collapse to empty. Without conservative
// branch handling the verifier would silently accept the pipe op below.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @uniform_unknown_else(%flag: i1) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @below {{PipeNet net_0 declared here}}
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    scf.if %flag {
    } else {
      // expected-error @below {{this `ttl.copy(buffer, pipe)` sends data on PipeNet net_0 from a node that is not a source}}
      // expected-note @below {{example node where the guard does not hold: core_x=1, core_y=0}}
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
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
      // expected-error @below {{could not statically analyze the PipeNet guard}}
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Pipe receive waits are schedule-relevant. A receive wait under an
// unanalyzable coordinate-dependent predicate is rejected instead of being
// omitted from the wait-for graph.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @pipe_receive_wait_unanalyzable_guard(%runtime: index) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %reserve = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %view = ttl.attach_cb %reserve, %cb
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %view
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      %core_x = ttl.core_x : index
      %scaled = arith.muli %core_x, %runtime : index
      %zero = arith.constant 0 : index
      // expected-note @below {{this expression is not statically analyzable}}
      %cond = arith.cmpi eq, %scaled, %zero : index
      scf.if %cond {
        // expected-error @below {{could not statically analyze the PipeNet guard}}
        ttl.wait %recv : !ttl.transfer_handle
      }
      ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    func.return
  }
}

// -----

// `arith.andi` of two unanalyzable predicates: the verifier attaches the
// "not statically analyzable" note to the source-earliest predicate, so the
// diagnostic is the same regardless of dataflow visit order.

module attributes {ttl.launch_grid = [2 : i64, 2 : i64]} {
  func.func @two_unanalyzable_predicates_andi(%runtime: index) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %core_y = ttl.core_y : index
    %scaled_x = arith.muli %core_x, %runtime : index
    %scaled_y = arith.muli %core_y, %runtime : index
    %zero = arith.constant 0 : index
    // expected-note @below {{this expression is not statically analyzable}}
    %cond_x = arith.cmpi eq, %scaled_x, %zero : index
    %cond_y = arith.cmpi eq, %scaled_y, %zero : index
    %cond = arith.andi %cond_x, %cond_y : i1
    scf.if %cond {
      // expected-error @below {{could not statically analyze the PipeNet guard}}
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Source order, not operand position, determines which predicate the note
// attaches to. The source-earliest predicate (`%cond_x`) is the *rhs* of
// the `arith.andi`; a "pick lhs" implementation would attach the note to
// `%cond_y` instead and the test would fail.

module attributes {ttl.launch_grid = [2 : i64, 2 : i64]} {
  func.func @source_order_beats_operand_position(%runtime: index) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %core_y = ttl.core_y : index
    %scaled_x = arith.muli %core_x, %runtime : index
    %scaled_y = arith.muli %core_y, %runtime : index
    %zero = arith.constant 0 : index
    // expected-note @below {{this expression is not statically analyzable}}
    %cond_x = arith.cmpi eq, %scaled_x, %zero : index
    %cond_y = arith.cmpi eq, %scaled_y, %zero : index
    %cond = arith.andi %cond_y, %cond_x : i1
    scf.if %cond {
      // expected-error @below {{could not statically analyze the PipeNet guard}}
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Same as above, but with `arith.ori`. The note still attaches to the
// source-earliest unanalyzable predicate.

module attributes {ttl.launch_grid = [2 : i64, 2 : i64]} {
  func.func @two_unanalyzable_predicates_ori(%runtime: index) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %core_y = ttl.core_y : index
    %scaled_x = arith.muli %core_x, %runtime : index
    %scaled_y = arith.muli %core_y, %runtime : index
    %zero = arith.constant 0 : index
    // expected-note @below {{this expression is not statically analyzable}}
    %cond_x = arith.cmpi eq, %scaled_x, %zero : index
    %cond_y = arith.cmpi eq, %scaled_y, %zero : index
    %cond = arith.ori %cond_x, %cond_y : i1
    scf.if %cond {
      // expected-error @below {{could not statically analyze the PipeNet guard}}
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
    // expected-error @below {{this `cb_wait` reads from a dataflow buffer that no other thread fills}}
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
    // expected-error @below {{this `cb_wait` runs on launched nodes where no thread pushes data to the buffer}}
    // expected-note @below {{example node where the guard does not hold: core_x=1}}
    %view = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Cross-PipeNet: copy targets `net_a` from a region whose only guard is
// for `net_b`. Diagnostic names `net_a` specifically. The
// `pipeNetName` attribute on each `ttl.create_pipe` exercises the
// named-PipeNet diagnostic form: the verifier surfaces the user's
// Python variable names (`net_a`, `net_b`) instead of synthesizing
// `net_<id>`.

module attributes {ttl.launch_grid = [4 : i64, 4 : i64]} {
  func.func @cross_net_guard() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @below {{PipeNet net_a declared here}}
    %pa = ttl.create_pipe src(0, 0) dst(0, 1) to(0, 3) net 0
        {pipeNetName = "net_a"}
        : !ttl.pipe<src(0, 0) dst(0, 1) to(0, 3) net 0>
    %pb = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 1
        {pipeNetName = "net_b"}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 1>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cond = ttl.is_dst {pipe_net_id = 1 : i64}
    scf.if %cond {
      %recv_reserve = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv_view = ttl.attach_cb %recv_reserve, %cb
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-error @below {{this `ttl.copy(pipe, buffer)` receives data from PipeNet net_a on a node that is not a destination}}
      // expected-note @below {{example node where the guard does not hold:}}
      %r = ttl.copy %pa, %recv_view
          : (!ttl.pipe<src(0, 0) dst(0, 1) to(0, 3) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
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
    // expected-note @below {{PipeNet net_0 declared here}}
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
        %recv_reserve = ttl.cb_reserve %cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %recv_view = ttl.attach_cb %recv_reserve, %cb
            : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
               !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        // expected-error @below {{this `ttl.copy(pipe, buffer)` receives data from PipeNet net_0 on a node that is not a destination}}
        // expected-note @below {{example node where the guard does not hold: core_x=0}}
        %recv = ttl.copy %pa, %recv_view
            : (!ttl.pipe<src(0, 0) dst(0, 1) to(0, 3) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.transfer_handle
      }
    }
    func.return
  }
}

// -----

// `affine.if` IntegerSet with two AND'd constraints that together still
// admit nodes outside the source role: `(d0 >= 0, 3 - d0 >= 0)` narrows to
// {0..3}, but the pipe source is only {0}. Exercises the multi-constraint
// path with a domain that the verifier must reject.

#multiWide = affine_set<(d0) : (d0 >= 0, 3 - d0 >= 0)>
module attributes {ttl.launch_grid = [8 : i64, 1 : i64]} {
  func.func @affine_if_multi_constraint_too_wide() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @below {{PipeNet net_0 declared here}}
    %pipe = ttl.create_pipe src(0, 0) dst(4, 0) to(7, 0) net 0
        : !ttl.pipe<src(0, 0) dst(4, 0) to(7, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    affine.if #multiWide(%x) {
      // expected-error @below {{this `ttl.copy(buffer, pipe)` sends data on PipeNet net_0 from a node that is not a source}}
      // expected-note @below {{example node where the guard does not hold: core_x=1}}
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(4, 0) to(7, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

#wideSet = affine_set<(d0) : (d0 - 4 >= 0)>
module attributes {ttl.launch_grid = [8 : i64, 1 : i64]} {
  func.func @affine_if_too_wide() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @below {{PipeNet net_0 declared here}}
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    affine.if #wideSet(%x) {
      // expected-error @below {{this `ttl.copy(buffer, pipe)` sends data on PipeNet net_0 from a node that is not a source}}
      // expected-note @below {{example node where the guard does not hold: core_x=4}}
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(3, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// `is_src` referencing a PipeNet id that no `ttl.create_pipe` declares is
// rejected. Without this check, the empty role domain would silently accept
// any pipe-coupled op nested under the bogus guard.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unknown_pipenet_id() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{references unknown PipeNet net_7}}
    %cond = ttl.is_src {pipe_net_id = 7 : i64}
    scf.if %cond {
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// `affine.if` over `floordiv(d0, 0)` must be rejected as `⊥` rather than
// silently substituting a value. A pipe-coupled op nested inside cannot be
// verified, so the verifier emits the "cannot prove" diagnostic.

#divByZero = affine_set<(d0) : (d0 floordiv 0 - 1 >= 0)>
module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @affine_if_div_by_zero() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    // expected-note @below {{this expression is not statically analyzable}}
    affine.if #divByZero(%x) {
      // expected-error @below {{could not statically analyze the PipeNet guard}}
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// `is_dst` referencing an unknown PipeNet id is rejected, mirroring the
// `is_src` check.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unknown_pipenet_id_dst() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{references unknown PipeNet net_9}}
    %cond = ttl.is_dst {pipe_net_id = 9 : i64}
    scf.if %cond {
      %recv_reserve = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv_view = ttl.attach_cb %recv_reserve, %cb
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_view
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
    }
    func.return
  }
}

// -----

// `is_active` referencing an unknown PipeNet id is rejected.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unknown_pipenet_id_active() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-error @below {{references unknown PipeNet net_5}}
    %cond = ttl.is_active {pipe_net_id = 5 : i64}
    scf.if %cond {
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// `pipenet_scope` referencing an unknown PipeNet id is rejected before
// downstream role-domain analysis runs.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unknown_pipenet_id_scope() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    // expected-error @below {{references unknown PipeNet net_4}}
    ttl.pipenet_scope attributes {ttl.pipe_net_ids = [4 : i64], ttl.pipe_net_roles = [0 : i64]} {
    }
    func.return
  }
}

// -----

// `pipenet_scope` with mismatched-length id and role arrays is rejected.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @scope_length_mismatch() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    // expected-error @below {{requires equal-length PipeNet id and role arrays}}
    ttl.pipenet_scope attributes {ttl.pipe_net_ids = [0 : i64, 0 : i64], ttl.pipe_net_roles = [0 : i64]} {
    }
    func.return
  }
}

// -----

// `pipenet_scope` with a role value outside {0, 1} is rejected.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @scope_role_out_of_range() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    // expected-error @below {{has invalid PipeNet role 7}}
    ttl.pipenet_scope attributes {ttl.pipe_net_ids = [0 : i64], ttl.pipe_net_roles = [7 : i64]} {
    }
    func.return
  }
}

// -----

// `arith.cmpi ne`: `x != 0` covers the pipe destination range only on coord 1,
// but the pipe destination is at coord 1 so a dst-side copy is fine here.
// However, a SRC-side copy from inside this guard is not — the guard does not
// imply src.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @cmpi_ne_insufficient_for_src() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @below {{PipeNet net_0 declared here}}
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %x = ttl.core_x : index
    %c0 = arith.constant 0 : index
    %cond = arith.cmpi ne, %x, %c0 : index
    scf.if %cond {
      // expected-error @below {{this `ttl.copy(buffer, pipe)` sends data on PipeNet net_0 from a node that is not a source}}
      // expected-note @below {{example node where the guard does not hold: core_x=1}}
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Pipe-coupled op inside `scf.for` with no surrounding guard. The loop adds
// no narrowing, so the body's domain is the full launch grid.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @scf_for_unguarded() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @below {{PipeNet net_0 declared here}}
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      // expected-error @below {{this `ttl.copy(buffer, pipe)` sends data on PipeNet net_0 from a node that is not a source}}
      // expected-note @below {{example node where the guard does not hold: core_x=1}}
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Pipe-coupled op inside `scf.execute_region` with no surrounding guard.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @scf_execute_region_unguarded() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @below {{PipeNet net_0 declared here}}
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    scf.execute_region {
      // expected-error @below {{this `ttl.copy(buffer, pipe)` sends data on PipeNet net_0 from a node that is not a source}}
      // expected-note @below {{example node where the guard does not hold: core_x=1}}
      %send = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      scf.yield
    }
    func.return
  }
}

// -----

// Helper has an unguarded `ttl.copy(cb, pipe)` and the caller invokes it
// without any role guard. The helper's entry domain is the caller's
// lattice (the full launch grid here), so the copy is rejected.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func private @send_helper(%cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
                                  %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    // expected-error @below {{this `ttl.copy(buffer, pipe)` sends data on PipeNet net_0 from a node that is not a source}}
    // expected-note @below {{example node where the guard does not hold: core_x=1}}
    %x = ttl.copy %cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    func.return
  }

  func.func @kernel_caller_unguarded() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-note @below {{PipeNet net_0 declared here}}
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    func.call @send_helper(%cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.return
  }
}
