// RUN: ttlang-opt %s --split-input-file -ttl-verify-pipenet-schedule | FileCheck %s --check-prefixes=CHECK,PARTIAL,PAIR --enable-var-scope

// Summary: Verifies accepted PipeNet synchronization correspondence and
// call-site ordering.

// A send requires a receiver post but does not require a receiver wait.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @send_without_receiver_wait
  func.func @send_without_receiver_wait()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // CHECK: %[[PIPE:.*]] = ttl.create_pipe
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // CHECK: ttl.if_dst %[[PIPE]]
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // CHECK: %[[RESERVE:.*]] = ttl.cb_reserve
      %reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // CHECK-NEXT: %[[RECEIVE:.*]] = ttl.copy %[[PIPE]], %[[RESERVE]]
      %receive = ttl.copy %pipe, %reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      // CHECK-NOT: ttl.wait %[[RECEIVE]]
    }
    // CHECK: ttl.if_src %[[PIPE]]
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // CHECK: %[[SEND:.*]] = ttl.copy {{.*}}, %[[PIPE]]
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      // CHECK-NEXT: ttl.wait %[[SEND]]
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    // CHECK-NOT: ttl.copy
    func.return
  }
}

// -----

// Receiver waits may consume a prefix of the send completions without waiting
// for every send.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // PARTIAL-LABEL: func.func @partial_receiver_waits
  func.func @partial_receiver_waits()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // PARTIAL: %[[PIPE:.*]] = ttl.create_pipe
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // PARTIAL: %[[RESERVE0:.*]] = ttl.cb_reserve
      %reserve0 = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // PARTIAL-NEXT: %[[RECEIVE0:.*]] = ttl.copy %[[PIPE]], %[[RESERVE0]]
      %receive0 = ttl.copy %pipe, %reserve0
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      // PARTIAL-NEXT: ttl.wait %[[RECEIVE0]]
      ttl.wait %receive0 : !ttl.transfer_handle
      // PARTIAL-NEXT: %[[RESERVE1:.*]] = ttl.cb_reserve
      %reserve1 = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // PARTIAL-NEXT: %[[RECEIVE1:.*]] = ttl.copy %[[PIPE]], %[[RESERVE1]]
      %receive1 = ttl.copy %pipe, %reserve1
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      // PARTIAL-NOT: ttl.wait %[[RECEIVE1]]
    }
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // PARTIAL: %[[SEND0:.*]] = ttl.copy {{.*}}, %[[PIPE]]
      %send0 = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      // PARTIAL-NEXT: ttl.wait %[[SEND0]]
      ttl.wait %send0 : !ttl.transfer_handle<write>
      // PARTIAL-NEXT: %[[SEND1:.*]] = ttl.copy {{.*}}, %[[PIPE]]
      %send1 = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      // PARTIAL-NEXT: ttl.wait %[[SEND1]]
      ttl.wait %send1 : !ttl.transfer_handle<write>
    }
    // PARTIAL-NOT: ttl.copy
    func.return
  }
}

// -----

// A receiver post without a send is valid when no receiver wait can block.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @receiver_post_without_wait
  func.func @receiver_post_without_wait()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // CHECK: %[[PIPE:.*]] = ttl.create_pipe
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // CHECK: %[[RESERVE:.*]] = ttl.cb_reserve
      %reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // CHECK-NEXT: %[[RECEIVE:.*]] = ttl.copy %[[PIPE]], %[[RESERVE]]
      %receive = ttl.copy %pipe, %reserve
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      // CHECK-NOT: ttl.wait %[[RECEIVE]]
    }
    // CHECK-NOT: ttl.copy
    func.return
  }
}

// -----

// Two helper calls are two send occurrences and correspond to two receiver
// posts in caller program order.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // PAIR-LABEL: func.func private @send_helper
  func.func private @send_helper(
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    // PAIR: %[[HELPER_SEND:.*]] = ttl.copy %{{.*}}, %{{.*}}
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    // PAIR-NEXT: ttl.wait %[[HELPER_SEND]]
    ttl.wait %send : !ttl.transfer_handle<write>
    func.return
  }

  // PAIR-LABEL: func.func @two_helper_calls
  func.func @two_helper_calls()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // PAIR: %[[PIPE:.*]] = ttl.create_pipe
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.if_src %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // PAIR: func.call @send_helper(%[[SEND_CB:.*]], %[[PIPE]])
      func.call @send_helper(%send_cb, %pipe)
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
      // PAIR-NEXT: func.call @send_helper(%[[SEND_CB]], %[[PIPE]])
      func.call @send_helper(%send_cb, %pipe)
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    }
    ttl.if_dst %pipe : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      // PAIR: %[[RESERVE0:.*]] = ttl.cb_reserve %[[RECV_CB:.*]]
      %reserve0 = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // PAIR-NEXT: %[[RECEIVE0:.*]] = ttl.copy %[[PIPE]], %[[RESERVE0]]
      %receive0 = ttl.copy %pipe, %reserve0
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      // PAIR-NEXT: ttl.wait %[[RECEIVE0]]
      ttl.wait %receive0 : !ttl.transfer_handle
      // PAIR-NEXT: %[[RESERVE1:.*]] = ttl.cb_reserve %[[RECV_CB]]
      %reserve1 = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      // PAIR-NEXT: %[[RECEIVE1:.*]] = ttl.copy %[[PIPE]], %[[RESERVE1]]
      %receive1 = ttl.copy %pipe, %reserve1
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.transfer_handle
      // PAIR-NEXT: ttl.wait %[[RECEIVE1]]
      ttl.wait %receive1 : !ttl.transfer_handle
    }
    // PAIR-NOT: ttl.copy
    func.return
  }
}
