// RUN: ttlang-opt %s --split-input-file -ttl-verify-pipenet-schedule | FileCheck %s --check-prefixes=CHECK,PARTIAL,PAIR,UNRELATED --enable-var-scope

// Summary: Verifies accepted PipeNet synchronization correspondence,
// call-site ordering, and nested-region control flow.

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

// Multiple waits on one receive token observe the same completed transfer;
// they do not require additional sends.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @repeated_wait_on_one_receive
  func.func @repeated_wait_on_one_receive()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // CHECK: %[[PIPE:.*]] = ttl.create_pipe
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // CHECK: %[[RESERVE:.*]] = ttl.cb_reserve %[[RECV_CB:.*]]
    %reserve = ttl.cb_reserve %recv_cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // CHECK-NEXT: %[[RECEIVE:.*]] = ttl.copy %[[PIPE]], %[[RESERVE]]
    %receive = ttl.copy %pipe, %reserve
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.transfer_handle
    // CHECK-NEXT: %[[SEND:.*]] = ttl.copy %[[SEND_CB:.*]], %[[PIPE]]
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    // CHECK-NEXT: ttl.wait %[[SEND]]
    ttl.wait %send : !ttl.transfer_handle<write>
    // CHECK-NEXT: ttl.wait %[[RECEIVE]]
    ttl.wait %receive : !ttl.transfer_handle
    // CHECK-NEXT: ttl.wait %[[RECEIVE]]
    ttl.wait %receive : !ttl.transfer_handle
    // CHECK-NEXT: return
    func.return
  }
}

// -----

// A completed receive permits the same receiver DFB slot to be posted for the
// next transfer.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func @reuse_receiver_slot_after_completion
  func.func @reuse_receiver_slot_after_completion()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // CHECK: %[[PIPE:.*]] = ttl.create_pipe
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // CHECK: %[[RESERVE0:.*]] = ttl.cb_reserve %[[RECV_CB:[^ ]+]]
    %reserve0 = ttl.cb_reserve %recv_cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // CHECK-NEXT: %[[RECEIVE0:.*]] = ttl.copy %[[PIPE]], %[[RESERVE0]]
    %receive0 = ttl.copy %pipe, %reserve0
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.transfer_handle
    // CHECK-NEXT: %[[SEND0:.*]] = ttl.copy %[[SEND_CB:.*]], %[[PIPE]]
    %send0 = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    // CHECK-NEXT: ttl.wait %[[SEND0]]
    ttl.wait %send0 : !ttl.transfer_handle<write>
    // CHECK-NEXT: ttl.wait %[[RECEIVE0]]
    ttl.wait %receive0 : !ttl.transfer_handle
    // CHECK-NEXT: ttl.cb_push %[[RECV_CB]]
    ttl.cb_push %recv_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // CHECK-NEXT: %[[READY0:.*]] = ttl.cb_wait %[[RECV_CB]]
    %ready0 = ttl.cb_wait %recv_cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // CHECK-NEXT: ttl.cb_pop %[[RECV_CB]]
    ttl.cb_pop %recv_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // CHECK-NEXT: %[[RESERVE1:.*]] = ttl.cb_reserve %[[RECV_CB]]
    %reserve1 = ttl.cb_reserve %recv_cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // CHECK-NEXT: %[[RECEIVE1:.*]] = ttl.copy %[[PIPE]], %[[RESERVE1]]
    %receive1 = ttl.copy %pipe, %reserve1
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.transfer_handle
    // CHECK-NEXT: %[[SEND1:.*]] = ttl.copy %[[SEND_CB]], %[[PIPE]]
    %send1 = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    // CHECK-NEXT: ttl.wait %[[SEND1]]
    ttl.wait %send1 : !ttl.transfer_handle<write>
    // CHECK-NEXT: ttl.wait %[[RECEIVE1]]
    ttl.wait %receive1 : !ttl.transfer_handle
    // CHECK-NEXT: ttl.cb_push %[[RECV_CB]]
    ttl.cb_push %recv_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // CHECK-NEXT: %[[READY1:.*]] = ttl.cb_wait %[[RECV_CB]]
    %ready1 = ttl.cb_wait %recv_cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // CHECK-NEXT: ttl.cb_pop %[[RECV_CB]]
    ttl.cb_pop %recv_cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
    // CHECK-NEXT: return
    func.return
  }
}

// -----

// A multi-block nested region is valid when it does not contribute pipe
// events. The surrounding function's pipe events retain their linear order.

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  // UNRELATED-LABEL: func.func @unrelated_nested_cfg
  func.func @unrelated_nested_cfg()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // UNRELATED: %[[PIPE:.*]] = ttl.create_pipe
    %pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    // UNRELATED-NEXT: %[[SEND_CB:.*]] = ttl.bind_cb
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // UNRELATED-NEXT: %[[RECV_CB:.*]] = ttl.bind_cb
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // UNRELATED-NEXT: scf.execute_region {
    scf.execute_region {
      // UNRELATED-NEXT: cf.br ^[[EXIT:bb[0-9]+]]
      cf.br ^exit
    ^exit:
      // UNRELATED: ^[[EXIT]]:
      // UNRELATED-NEXT: scf.yield
      scf.yield
    }
    // UNRELATED: %[[RESERVE:.*]] = ttl.cb_reserve %[[RECV_CB]]
    %reserve = ttl.cb_reserve %recv_cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // UNRELATED-NEXT: %[[RECEIVE:.*]] = ttl.copy %[[PIPE]], %[[RESERVE]]
    %receive = ttl.copy %pipe, %reserve
        : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.transfer_handle
    // UNRELATED-NOT: ttl.wait %[[RECEIVE]]
    // UNRELATED: %[[SEND:.*]] = ttl.copy %[[SEND_CB]], %[[PIPE]]
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
        -> !ttl.transfer_handle<write>
    // UNRELATED-NEXT: ttl.wait %[[SEND]]
    ttl.wait %send : !ttl.transfer_handle<write>
    // UNRELATED-NEXT: return
    func.return
  }
}

// -----

// A runtime count passed from a kernel entry argument remains identical at the
// source and destination after resolving the helper call argument.

module attributes {ttl.launch_grid = [2 : i64, 1 : i64]} {
  // CHECK-LABEL: func.func private @runtime_count_helper(
  func.func private @runtime_count_helper(
      %count: index,
      %send_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %recv_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %pipe: !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: scf.for {{.*}} to %[[COUNT:[a-zA-Z0-9]+]] step
    scf.for %iteration = %c0 to %count step %c1 {
      // CHECK: ttl.if_src %[[PIPE:[a-zA-Z0-9]+]] :
      ttl.if_src %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        // CHECK: %[[SEND:[a-zA-Z0-9]+]] = ttl.copy %[[SEND_CB:[a-zA-Z0-9]+]], %[[PIPE]]
        %send = ttl.copy %send_cb, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
               !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
            -> !ttl.transfer_handle<write>
        // CHECK-NEXT: ttl.wait %[[SEND]]
        ttl.wait %send : !ttl.transfer_handle<write>
      }
      // CHECK: ttl.if_dst %[[PIPE]]
      ttl.if_dst %pipe
          : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
        // CHECK: %[[RESERVE:[a-zA-Z0-9]+]] = ttl.cb_reserve %[[RECV_CB:[a-zA-Z0-9]+]]
        %reserve = ttl.cb_reserve %recv_cb
            : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        // CHECK-NEXT: %[[RECEIVE:.*]] = ttl.copy %[[PIPE]], %[[RESERVE]]
        %receive = ttl.copy %pipe, %reserve
            : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.transfer_handle
        // CHECK-NEXT: ttl.wait %[[RECEIVE]]
        ttl.wait %receive : !ttl.transfer_handle
      }
    }
    func.return
  }

  // CHECK-LABEL: func.func @runtime_count_through_call
  // CHECK-SAME: (%[[COUNT:[a-zA-Z0-9]+]]: index)
  func.func @runtime_count_through_call(%count: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // CHECK: %[[PIPE:[a-zA-Z0-9]+]] = ttl.create_pipe
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    // CHECK-NEXT: %[[SEND_CB:[a-zA-Z0-9]+]] = ttl.bind_cb
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // CHECK-NEXT: %[[RECV_CB:[a-zA-Z0-9]+]] = ttl.bind_cb
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // CHECK-NEXT: call @runtime_count_helper(%[[COUNT]], %[[SEND_CB]], %[[RECV_CB]], %[[PIPE]])
    func.call @runtime_count_helper(%count, %send_cb, %recv_cb, %pipe)
        : (index, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> ()
    // CHECK-NOT: ttl.copy
    func.return
  }
}

// -----

// Receiver code may wait for only a subset of sends; other sends do not require
// receive waits.

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
