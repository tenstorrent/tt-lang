// RUN: ttlang-opt %s -ttl-verify-pipenet-schedule | FileCheck %s

// Summary: Verifies that the schedule bound applies independently at each
// launch node rather than to the coordinate-specialized graph as a whole.

// Thirty-two sends and thirty-two post/wait pairs at each of sixty-four
// receivers produce 4128 graph nodes in total but at most 64 at one launch
// node. Helper expansion preserves each static occurrence.

module attributes {ttl.launch_grid = [9 : i64, 8 : i64]} {
  // The receive leaf posts one reservation, waits for its payload, and makes
  // the completed DFB block available to its consumer.
  // CHECK-LABEL: func.func private @receive_once(
  // CHECK-SAME: %[[RECV_CB:.*]]: !ttl.cb
  // CHECK-SAME: %[[RECV_PIPE:.*]]: !ttl.pipe
  func.func private @receive_once(
      %recv_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) {
    // CHECK-NEXT: %[[RESERVE:.*]] = ttl.cb_reserve %[[RECV_CB]]
    %reserve = ttl.cb_reserve %recv_cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // CHECK-NEXT: %[[RECEIVE:.*]] = ttl.copy %[[RECV_PIPE]], %[[RESERVE]]
    %receive = ttl.copy %pipe, %reserve
        : (!ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.transfer_handle
    // CHECK-NEXT: ttl.wait %[[RECEIVE]]
    ttl.wait %receive : !ttl.transfer_handle
    // CHECK-NEXT: ttl.cb_push %[[RECV_CB]]
    ttl.cb_push %recv_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // CHECK-NEXT: return
    func.return
  }

  func.func private @receive_2(
      %recv_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) {
    func.call @receive_once(%recv_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    func.call @receive_once(%recv_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    func.return
  }

  func.func private @receive_4(
      %recv_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) {
    func.call @receive_2(%recv_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    func.call @receive_2(%recv_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    func.return
  }

  func.func private @receive_8(
      %recv_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) {
    func.call @receive_4(%recv_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    func.call @receive_4(%recv_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    func.return
  }

  func.func private @receive_16(
      %recv_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) {
    func.call @receive_8(%recv_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    func.call @receive_8(%recv_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    func.return
  }

  // CHECK-LABEL: func.func private @receive_32(
  // CHECK-SAME: %[[RECV32_CB:.*]]: !ttl.cb
  // CHECK-SAME: %[[RECV32_PIPE:.*]]: !ttl.pipe
  func.func private @receive_32(
      %recv_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) {
    // CHECK-NEXT: call @receive_16(%[[RECV32_CB]], %[[RECV32_PIPE]])
    func.call @receive_16(%recv_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    // CHECK-NEXT: call @receive_16(%[[RECV32_CB]], %[[RECV32_PIPE]])
    func.call @receive_16(%recv_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    // CHECK-NEXT: return
    func.return
  }

  // The send leaf waits only for local write completion; it does not add a
  // receiver-wait event to the schedule graph.
  // CHECK-LABEL: func.func private @send_once(
  // CHECK-SAME: %[[SEND_CB:.*]]: !ttl.cb
  // CHECK-SAME: %[[SEND_PIPE:.*]]: !ttl.pipe
  func.func private @send_once(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) {
    // CHECK-NEXT: %[[SEND:.*]] = ttl.copy %[[SEND_CB]], %[[SEND_PIPE]]
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>)
        -> !ttl.transfer_handle<write>
    // CHECK-NEXT: ttl.wait %[[SEND]]
    ttl.wait %send : !ttl.transfer_handle<write>
    // CHECK-NEXT: return
    func.return
  }

  func.func private @send_2(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) {
    func.call @send_once(%send_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    func.call @send_once(%send_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    func.return
  }

  func.func private @send_4(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) {
    func.call @send_2(%send_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    func.call @send_2(%send_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    func.return
  }

  func.func private @send_8(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) {
    func.call @send_4(%send_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    func.call @send_4(%send_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    func.return
  }

  func.func private @send_16(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) {
    func.call @send_8(%send_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    func.call @send_8(%send_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    func.return
  }

  // CHECK-LABEL: func.func private @send_32(
  // CHECK-SAME: %[[SEND32_CB:.*]]: !ttl.cb
  // CHECK-SAME: %[[SEND32_PIPE:.*]]: !ttl.pipe
  func.func private @send_32(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) {
    // CHECK-NEXT: call @send_16(%[[SEND32_CB]], %[[SEND32_PIPE]])
    func.call @send_16(%send_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    // CHECK-NEXT: call @send_16(%[[SEND32_CB]], %[[SEND32_PIPE]])
    func.call @send_16(%send_cb, %pipe)
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
    // CHECK-NEXT: return
    func.return
  }

  // CHECK-LABEL: func.func @large_scatter_schedule
  func.func @large_scatter_schedule()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // CHECK-NEXT: %[[PIPE:.*]] = ttl.create_pipe
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(8, 7) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>
    // CHECK-NEXT: %[[ROOT_SEND_CB:.*]] = ttl.bind_cb
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // CHECK-NEXT: %[[ROOT_RECV_CB:.*]] = ttl.bind_cb
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // CHECK-NEXT: ttl.if_dst %[[PIPE]]
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0> {
      // CHECK-NEXT: call @receive_32(%[[ROOT_RECV_CB]], %[[PIPE]])
      func.call @receive_32(%recv_cb, %pipe)
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
      // CHECK-NEXT: }
    }
    // CHECK-NEXT: ttl.if_src %[[PIPE]]
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0> {
      // CHECK-NEXT: call @send_32(%[[ROOT_SEND_CB]], %[[PIPE]])
      func.call @send_32(%send_cb, %pipe)
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(8, 7) net 0>) -> ()
      // CHECK-NEXT: }
    }
    // CHECK-NEXT: return
    // CHECK-NOT: ttl.copy
    func.return
  }
}
