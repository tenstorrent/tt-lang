// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -ttl-verify-pipenet-guards

// Summary: Negative tests for pipe schedules that would deadlock at runtime.

// A same-thread loopback receive wait before the matching send creates a
// cycle: the wait needs the send to complete, but program order places the
// send after the wait.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @loopback_wait_before_send() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(1, 0) net 0
        {pipeNetName = "net"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0> {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv_view = ttl.attach_cb %recv_reserve, %recv_cb
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_view
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      // expected-error @below {{receive wait occurs before the send that completes it on PipeNet net}}
      // expected-note @below {{this wait blocks until the sender transfers into the posted destination dataflow buffer slot}}
      // expected-note @below {{move the receive wait after the send, or place send and receive in separate data-movement threads}}
      ttl.wait %recv : !ttl.transfer_handle
      // expected-note @below {{this send is ordered after the wait in the same data-movement thread}}
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
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
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
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
      // expected-error @below {{cannot prove a one-to-one synchronization schedule on PipeNet net_0 for receiver core_x=1, core_y=0; found 1 receiver post occurrence(s) and 2 send occurrence(s)}}
      %send0 = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send0 : !ttl.transfer_handle<write>
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
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv = ttl.copy %pipe, %recv_reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      // expected-error @below {{cannot prove a one-to-one synchronization schedule on PipeNet net_0 for receiver core_x=1, core_y=0; found 1 send occurrence(s) and 2 receive wait occurrence(s)}}
      ttl.wait %recv : !ttl.transfer_handle
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

// One static send inside a loop does not correspond to one static receiver post
// outside the loop because their dynamic execution counts differ.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @different_rendezvous_loop_contexts()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
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
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
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

// A loop-IV condition restricts the send to the first iteration while the
// receiver posts on both iterations.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @loop_conditional_rendezvous_count()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
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
    %loopback_pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(1, 0) net 0
        {pipeNetName = "loopback_net"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>
    %other_pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(2, 0) net 1
        {pipeNetName = "other_net"}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 1>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %loopback_pipe : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0> {
      // expected-error @below {{pipe send occurs before the receiver posts a dataflow buffer reservation on PipeNet loopback_net}}
      // expected-note @below {{this send waits for each destination to post `ttl.copy(pipe, dst)`}}
      // expected-note @below {{move `ttl.copy(pipe, dst)` before the dependent send, or place send and receive in separate data-movement threads}}
      %send = ttl.copy %send_cb, %loopback_pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      %other_send = ttl.copy %send_cb, %other_pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(2, 0) net 1>)
          -> !ttl.transfer_handle<write>
      ttl.wait %other_send : !ttl.transfer_handle<write>
      %recv_reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %recv_view = ttl.attach_cb %recv_reserve, %recv_cb
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // expected-note @below {{this receiver post is ordered after the send in the same data-movement thread}}
      %recv = ttl.copy %loopback_pipe, %recv_view
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>,
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
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(1, 0) net 0
        {pipeNetName = "net"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0> {
      // expected-error @below {{pipe send occurs before the receiver posts a dataflow buffer reservation on PipeNet net}}
      // expected-note @below {{this send waits for each destination to post `ttl.copy(pipe, dst)`}}
      // expected-note @below {{move `ttl.copy(pipe, dst)` before the dependent send, or place send and receive in separate data-movement threads}}
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>)
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
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      ttl.wait %recv : !ttl.transfer_handle
    }
    func.return
  }
}
