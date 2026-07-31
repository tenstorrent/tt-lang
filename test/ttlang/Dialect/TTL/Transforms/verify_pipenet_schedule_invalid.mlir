// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -ttl-verify-pipenet-schedule

// Summary: Negative tests for invalid or unprovable PipeNet synchronization
// schedules.

// Schedule verification requires the launch grid used to specialize events to
// individual kernel instances.

// expected-error @below {{ttl-verify-pipenet-schedule requires a `ttl.launch_grid` module attribute}}
module {
  func.func @missing_launch_grid()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    func.return
  }
}

// -----

// A helper argument can preserve a coordinate-dependent loop bound across a
// call. The source executes once, while the destination executes twice.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func private @coordinate_dependent_count_helper(
      %count: index,
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %recv_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %iteration = %c0 to %count step %c1 {
      ttl.if_src %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        // expected-error @below {{cannot prove a one-to-one synchronization schedule}}
        %send = ttl.copy %send_cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
      ttl.if_dst %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %reserve = ttl.cb_reserve %recv_cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        // expected-note @below {{matching receiver post occurrence is here}}
        %receive = ttl.copy %pipe, %reserve
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.transfer_handle
        ttl.wait %receive : !ttl.transfer_handle
      }
    }
    func.return
  }

  func.func @coordinate_dependent_count_through_call()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %c1 = arith.constant 1 : index
    %count = arith.addi %core_x, %c1 : index
    func.call @coordinate_dependent_count_helper(
        %count, %send_cb, %recv_cb, %pipe)
        : (index, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.return
  }
}

// -----

// Pipe events must be reachable from a kernel-thread function so their launch
// nodes and call sites are defined.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unreachable_pipe_event() {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // expected-error @below {{cannot verify PipeNet synchronization in @unreachable_pipe_event because it is not reachable from a kernel-thread function through direct calls}}
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// PipeNet references are validated even when the module has no declaration.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @undeclared_pipe_predicate()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{references unknown PipeNet net_7}}
    %is_src = ttl.is_src {pipe_net_id = 7 : i64}
    scf.if %is_src {
    }
    func.return
  }
}

// -----

// A same-thread loopback receive wait before the matching send creates a
// cycle: the wait needs the send to complete, but program order places the
// send after the wait.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @loopback_wait_before_send() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {pipeNetName = "net"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv_view = ttl.attach_cb %recv_reserve, %recv_cb
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_view
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      // expected-error @below {{receive wait occurs before the send that completes it on PipeNet net}}
      // expected-note @below {{this wait blocks until the sender transfers into the posted destination dataflow buffer slot}}
      // expected-note @below {{move the receive wait after the send, or place send and receive in separate data-movement threads}}
      ttl.wait %recv : !ttl.transfer_handle
      // expected-note @below {{this send is ordered after the wait in the same data-movement thread}}
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// A send cannot rendezvous when no receiver posts a destination DFB slot.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @missing_receiver_post()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // expected-error @below {{PipeNet net_0 requires one receiver post operation for each send operation at receiver core_x=1, core_y=0; found 0 receiver post operation(s) and 1 send operation(s)}}
      // expected-note @below {{this send operation has no corresponding receiver post operation}}
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// A receiver post and wait cannot complete when the corresponding send is
// absent.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @missing_send()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      // expected-error @below {{PipeNet net_0 requires one send operation for each receive wait operation at receiver core_x=1, core_y=0; found 0 send operation(s) and 1 receive wait operation(s)}}
      // expected-note @below {{this receive wait operation has no corresponding send operation}}
      ttl.wait %recv : !ttl.transfer_handle
    }
    func.return
  }
}

// -----

// Recursive helper calls containing pipe events have no finite static event
// sequence for correspondence and program-order analysis.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func private @recursive_send(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    // expected-error @below {{cannot verify PipeNet synchronization through a recursive call to @recursive_send}}
    func.call @recursive_send(%send_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.return
  }

  func.func @recursive_pipe_events()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      func.call @recursive_send(%send_cb, %pipe)
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    }
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %receive = ttl.copy %pipe, %reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      ttl.wait %receive : !ttl.transfer_handle
    }
    func.return
  }
}

// -----

// Two calls to a send helper are two static send occurrences. One receiver
// post cannot satisfy both calls.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func private @send_twice_helper(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    // expected-error @below {{PipeNet net_0 requires one receiver post operation for each send operation at receiver core_x=1, core_y=0; found 1 receiver post operation(s) and 2 send operation(s)}}
    // expected-note @below {{this send operation has no corresponding receiver post operation}}
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    func.return
  }

  func.func @helper_call_count_mismatch()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      func.call @send_twice_helper(%send_cb, %pipe)
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
      func.call @send_twice_helper(%send_cb, %pipe)
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    }
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      ttl.wait %recv : !ttl.transfer_handle
    }
    func.return
  }
}

// -----

// Expanding a helper at its call site exposes a loopback send that executes
// before the caller posts the receiver reservation.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func private @loopback_send_helper(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>) {
    // expected-error @below {{pipe send occurs before the receiver posts a dataflow buffer reservation on PipeNet loopback}}
    // expected-note @below {{this send waits for each destination to post `ttl.copy(pipe, dst)`}}
    // expected-note @below {{move `ttl.copy(pipe, dst)` before the dependent send, or place send and receive in separate data-movement threads}}
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    func.return
  }

  func.func @helper_send_before_caller_post()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {pipeNetName = "loopback"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      func.call @loopback_send_helper(%send_cb, %pipe)
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>) -> ()
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-note @below {{this receiver post is ordered after the send in the same data-movement thread}}
      %recv = ttl.copy %pipe, %recv_reserve
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      ttl.wait %recv : !ttl.transfer_handle
    }
    func.return
  }
}

// -----

// Straight-line post and send counts must agree for one logical pipe.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @different_static_post_send_counts()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      ttl.wait %recv : !ttl.transfer_handle
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send0 = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send0 : !ttl.transfer_handle<write>
      // expected-error @below {{PipeNet net_0 requires one receiver post operation for each send operation at receiver core_x=1, core_y=0; found 1 receiver post operation(s) and 2 send operation(s)}}
      // expected-note @below {{this send operation has no corresponding receiver post operation}}
      %send1 = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send1 : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Reusing one receive handle for two completion waits requires two sends, but
// the pipe has only one send occurrence.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @different_send_receive_wait_counts()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      ttl.wait %recv : !ttl.transfer_handle
      // expected-error @below {{PipeNet net_0 requires one send operation for each receive wait operation at receiver core_x=1, core_y=0; found 1 send operation(s) and 2 receive wait operation(s)}}
      // expected-note @below {{this receive wait operation has no corresponding send operation}}
      ttl.wait %recv : !ttl.transfer_handle
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// A send operation inside a loop does not correspond to a receiver-post
// operation outside the loop because their dynamic execution counts differ.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @different_rendezvous_loop_contexts()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-note @below {{matching receiver post occurrence is here}}
      %recv = ttl.copy %pipe, %recv_reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      ttl.wait %recv : !ttl.transfer_handle
    }
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        // expected-error @below {{cannot prove a one-to-one synchronization schedule on PipeNet net_0 for receiver core_x=1, core_y=0; receiver post and send occurrences do not have matching proven execution counts and conditions}}
        %send = ttl.copy %send_cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
    }
    func.return
  }
}

// -----

// The source and destination evaluate the shared scf.for upper bound at their
// own coordinates. One send cannot match two receiver posts.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @node_dependent_rendezvous_count()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %upper = arith.addi %core_x, %c1 : index
    scf.for %iteration = %c0 to %upper step %c1 {
      ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        // expected-error @below {{cannot prove a one-to-one synchronization schedule on PipeNet net_0 for receiver core_x=1, core_y=0; receiver post and send occurrences do not have matching proven execution counts and conditions}}
        %send = ttl.copy %send_cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
      ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %recv_reserve = ttl.cb_reserve %recv_cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        // expected-note @below {{matching receiver post occurrence is here}}
        %recv = ttl.copy %pipe, %recv_reserve
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.transfer_handle
        ttl.wait %recv : !ttl.transfer_handle
      }
    }
    func.return
  }
}

// -----

// Equal unresolved loop syntax does not prove equal multiplicity when the
// runtime upper bound also depends on the launch node.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @unresolved_node_dependent_rendezvous_count(%runtime: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %upper = arith.addi %runtime, %core_x : index
    scf.for %iteration = %c0 to %upper step %c1 {
      ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        // expected-error @below {{cannot prove a one-to-one synchronization schedule on PipeNet net_0 for receiver core_x=1, core_y=0; receiver post and send occurrences do not have matching proven execution counts and conditions}}
        %send = ttl.copy %send_cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
      ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %recv_reserve = ttl.cb_reserve %recv_cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        // expected-note @below {{matching receiver post occurrence is here}}
        %recv = ttl.copy %pipe, %recv_reserve
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.transfer_handle
        ttl.wait %recv : !ttl.transfer_handle
      }
    }
    func.return
  }
}

// -----

// A loop-IV condition restricts the send to the first iteration while the
// receiver posts on both iterations.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @loop_conditional_rendezvous_count()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    scf.for %iteration = %c0 to %c2 step %c1 {
      %is_first = arith.cmpi eq, %iteration, %c0 : index
      scf.if %is_first {
        ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
          // expected-error @below {{cannot prove a one-to-one synchronization schedule on PipeNet net_0 for receiver core_x=1, core_y=0; receiver post and send occurrences do not have matching proven execution counts and conditions}}
          %send = ttl.copy %send_cb, %pipe
              : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
                 !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
              -> !ttl.transfer_handle<write>
          ttl.wait %send : !ttl.transfer_handle<write>
        }
      }
      ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        %recv_reserve = ttl.cb_reserve %recv_cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        // expected-note @below {{matching receiver post occurrence is here}}
        %recv = ttl.copy %pipe, %recv_reserve
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.transfer_handle
        ttl.wait %recv : !ttl.transfer_handle
      }
    }
    func.return
  }
}

// -----

// A send can be separated from the matching receiver post by other pipe
// events. The verifier should still report the real protocol violation instead
// of the intermediate program-order edge.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @loopback_send_before_receive_post_with_intervening_send() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %loopback_pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {pipeNetName = "loopback_net"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %other_pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
        {pipeNetName = "other_net"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %loopback_pipe : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      // expected-error @below {{pipe send occurs before the receiver posts a dataflow buffer reservation on PipeNet loopback_net}}
      // expected-note @below {{this send waits for each destination to post `ttl.copy(pipe, dst)`}}
      // expected-note @below {{move `ttl.copy(pipe, dst)` before the dependent send, or place send and receive in separate data-movement threads}}
      %send = ttl.copy %send_cb, %loopback_pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      %other_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %other_recv = ttl.copy %other_pipe, %other_reserve
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      %other_send = ttl.copy %send_cb, %other_pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>)
          -> !ttl.transfer_handle<write>
      ttl.wait %other_send : !ttl.transfer_handle<write>
      ttl.wait %other_recv : !ttl.transfer_handle
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv_view = ttl.attach_cb %recv_reserve, %recv_cb
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-note @below {{this receiver post is ordered after the send in the same data-movement thread}}
      %recv = ttl.copy %loopback_pipe, %recv_view
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      ttl.wait %recv : !ttl.transfer_handle
    }
    func.return
  }
}

// -----

// A same-thread loopback send before the receive copy creates a cycle: the
// send waits for the receiver post, but program order executes that post after
// the send.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @loopback_send_before_receive_post() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {pipeNetName = "net"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0> {
      // expected-error @below {{pipe send occurs before the receiver posts a dataflow buffer reservation on PipeNet net}}
      // expected-note @below {{this send waits for each destination to post `ttl.copy(pipe, dst)`}}
      // expected-note @below {{move `ttl.copy(pipe, dst)` before the dependent send, or place send and receive in separate data-movement threads}}
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv_view = ttl.attach_cb %recv_reserve, %recv_cb
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-note @below {{this receiver post is ordered after the send in the same data-movement thread}}
      %recv = ttl.copy %pipe, %recv_view
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      ttl.wait %recv : !ttl.transfer_handle
    }
    func.return
  }
}

// -----

// A correspondence error in one PipeNet must not suppress an independent
// wait-for-cycle diagnostic in another PipeNet.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @independent_correspondence_error()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      ttl.wait %recv : !ttl.transfer_handle
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send0 = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send0 : !ttl.transfer_handle<write>
      // expected-error @below {{PipeNet net_0 requires one receiver post operation for each send operation at receiver core_x=1, core_y=0; found 1 receiver post operation(s) and 2 send operation(s)}}
      // expected-note @below {{this send operation has no corresponding receiver post operation}}
      %send1 = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send1 : !ttl.transfer_handle<write>
    }
    func.return
  }

  func.func @independent_wait_for_cycle()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1
        {pipeNetName = "loopback"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
    %send_cb = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 3 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1> {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv_view = ttl.attach_cb %recv_reserve, %recv_cb
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_view
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      // expected-error @below {{receive wait occurs before the send that completes it on PipeNet loopback}}
      // expected-note @below {{this wait blocks until the sender transfers into the posted destination dataflow buffer slot}}
      // expected-note @below {{move the receive wait after the send, or place send and receive in separate data-movement threads}}
      ttl.wait %recv : !ttl.transfer_handle
      // expected-note @below {{this send is ordered after the wait in the same data-movement thread}}
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}

// -----

// Repeated helper calls can create exponentially many pipe events. Schedule
// construction must diagnose its bound instead of exhausting memory or the
// cycle checker's call stack.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func private @expansion_leaf(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    // expected-error @below {{cannot verify PipeNet synchronization because the schedule contains more than 4096 pipe events after specializing launch nodes and expanding helper calls}}
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    func.return
  }

  func.func private @expansion_level_1(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    func.call @expansion_leaf(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_leaf(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_leaf(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_leaf(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_leaf(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_leaf(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_leaf(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_leaf(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.return
  }

  func.func private @expansion_level_2(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    func.call @expansion_level_1(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_1(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_1(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_1(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_1(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_1(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_1(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_1(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.return
  }

  func.func private @expansion_level_3(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    func.call @expansion_level_2(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_2(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_2(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_2(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_2(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_2(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_2(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_2(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.return
  }

  func.func private @expansion_level_4(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    func.call @expansion_level_3(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_3(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_3(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_3(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_3(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_3(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_3(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_3(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.return
  }

  func.func private @expansion_level_5(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    func.call @expansion_level_4(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_4(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_4(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_4(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_4(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_4(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_4(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.call @expansion_level_4(%send_cb, %pipe) : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    func.return
  }

  func.func @bounded_schedule_expansion()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      func.call @expansion_level_5(%send_cb, %pipe)
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    }
    func.return
  }
}
