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

// A send can be separated from the matching receive address publication by
// other pipe events. The verifier should still report the real protocol
// violation instead of the intermediate program-order edge.

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
      // expected-error @below {{pipe send occurs before the receiver publishes a destination address on PipeNet loopback_net}}
      // expected-note @below {{this send waits for each destination to execute `ttl.copy(pipe, dst)`}}
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
      // expected-note @below {{this receive address publication is ordered after the send in the same data-movement thread}}
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
// send waits for the destination address, but program order publishes that
// address after the send.

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
      // expected-error @below {{pipe send occurs before the receiver publishes a destination address on PipeNet net}}
      // expected-note @below {{this send waits for each destination to execute `ttl.copy(pipe, dst)`}}
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
      // expected-note @below {{this receive address publication is ordered after the send in the same data-movement thread}}
      %recv = ttl.copy %pipe, %recv_view
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      ttl.wait %recv : !ttl.transfer_handle
    }
    func.return
  }
}
