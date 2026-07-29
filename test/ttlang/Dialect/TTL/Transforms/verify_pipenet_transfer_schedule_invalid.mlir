// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -ttl-verify-pipenet-guards

// Summary: Verifies explicit PipeTransfer events preserve PipeNet guard and
// schedule diagnostics after frontend pipe copies have been expanded.

// A receive completion cannot precede the send that completes the posted
// destination slot in the same data-movement thread.

module attributes {ttl.dfb_allocations = [], ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @receive_wait_before_send()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {pipeNetName = "net"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %transfer = ttl.pipe_transfer.create %pipe
        {expectedReceivers = 1 : i64,
         kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
        -> !ttl.pipe_transfer
    %recv = ttl.cb_reserve %recv_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %token = ttl.pipe_transfer.post %transfer, %recv
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    // expected-error @below {{receive wait occurs before the send that completes it on PipeNet net}}
    // expected-note @below {{this wait blocks until the sender transfers into the posted destination dataflow buffer slot}}
    // expected-note @below {{move the receive wait after the send, or place send and receive in separate data-movement threads}}
    ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
    // expected-note @below {{this send is ordered after the wait in the same data-movement thread}}
    %send = ttl.pipe_transfer.send %transfer, %send_dfb
        : (!ttl.pipe_transfer,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> !ttl.transfer_handle<write>
    func.return
  }
}

// -----

// A loopback send cannot precede the receive post that makes its destination
// slot available.

module attributes {ttl.dfb_allocations = [], ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @send_before_receive_post()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        {pipeNetName = "net"}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %transfer = ttl.pipe_transfer.create %pipe
        {expectedReceivers = 1 : i64,
         kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
        -> !ttl.pipe_transfer
    // expected-error @below {{pipe send occurs before the receiver posts a dataflow buffer reservation on PipeNet net}}
    // expected-note @below {{this send waits for each destination to post `ttl.copy(pipe, dst)`}}
    // expected-note @below {{move `ttl.copy(pipe, dst)` before the dependent send, or place send and receive in separate data-movement threads}}
    %send = ttl.pipe_transfer.send %transfer, %send_dfb
        : (!ttl.pipe_transfer,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> !ttl.transfer_handle<write>
    %recv = ttl.cb_reserve %recv_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-note @below {{this receiver post is ordered after the send in the same data-movement thread}}
    %token = ttl.pipe_transfer.post %transfer, %recv
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.pipe_token<net 0>
    ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
    func.return
  }
}

// -----

// A receive wait under an unanalyzable coordinate-dependent predicate cannot
// be omitted from the wait-for graph.

module attributes {ttl.dfb_allocations = [], ttl.launch_grid = [2 : i64, 1 : i64]} {
  func.func @receive_wait_unanalyzable_guard(%runtime: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %transfer = ttl.pipe_transfer.create %pipe
        {expectedReceivers = 1 : i64,
         kind = #ttl.pipe_transfer_kind<point_to_point>}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
        -> !ttl.pipe_transfer
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.pipe_transfer.send %transfer, %send_dfb
          : (!ttl.pipe_transfer,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> !ttl.transfer_handle<write>
    }
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %recv = ttl.cb_reserve %recv_dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %token = ttl.pipe_transfer.post %transfer, %recv
          : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.pipe_token<net 0>
      %core_x = ttl.core_x : index
      %scaled = arith.muli %core_x, %runtime : index
      %zero = arith.constant 0 : index
      // expected-note @below {{this expression is not statically analyzable}}
      %condition = arith.cmpi eq, %scaled, %zero : index
      scf.if %condition {
        // expected-error @below {{could not statically analyze the PipeNet guard}}
        ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
      }
    }
    func.return
  }
}
