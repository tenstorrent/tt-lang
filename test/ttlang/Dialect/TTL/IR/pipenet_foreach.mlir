// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// Summary: Verifies PipeNet foreach ops and selected-pipe representations.

// A source foreach body sends from a DFB through the selected source pipe.

// CHECK-LABEL: func.func @foreach_src_send
func.func @foreach_src_send() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  // CHECK: ttl.pipenet_foreach_src
  // CHECK-SAME: records = #ttl.pipenet_records<net 0 name "row_net" pipes
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 name "row_net" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    // CHECK: ^bb0(%[[PIPE:.*]]: !ttl.selected_pipe_src)
    // CHECK: ttl.copy %{{.*}}, %[[PIPE]]
    %xf = ttl.copy %cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.selected_pipe_src)
        -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
    ttl.yield
  }
  func.return
}

// -----

// A destination foreach body receives into a reserved DFB slot through the
// selected destination pipe.

// CHECK-LABEL: func.func @foreach_dst_receive
func.func @foreach_dst_receive() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 name "row_net" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_dst):
    %recv = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // CHECK: ^bb0(%[[PIPE:.*]]: !ttl.selected_pipe_dst)
    // CHECK: ttl.copy %[[PIPE]], %{{.*}}
    %xf = ttl.copy %pipe, %recv
        : (!ttl.selected_pipe_dst,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.receive_request
    ttl.wait %xf : !ttl.receive_request
    ttl.yield
  }
  func.return
}

// -----

// Selection identifies why a callback executes, not its copy direction. A
// loopback source-selected record can therefore receive on the same node.

// CHECK-LABEL: func.func @source_selected_loopback_receive
func.func @source_selected_loopback_receive()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  // CHECK: %[[CB:.*]] = ttl.bind_cb
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  // CHECK: ttl.pipenet_foreach_src
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    // CHECK: ^bb0(%[[PIPE:.*]]: !ttl.selected_pipe_src)
    // CHECK-NEXT: %[[RESERVE:.*]] = ttl.cb_reserve %[[CB]]
    %reserve = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    // CHECK-NEXT: %[[COPY:.*]] = ttl.copy %[[PIPE]], %[[RESERVE]]
    %copy = ttl.copy %pipe, %reserve
        : (!ttl.selected_pipe_src,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.receive_request
    // CHECK-NEXT: ttl.wait %[[COPY]]
    ttl.wait %copy : !ttl.receive_request
    ttl.yield
  }
  func.return
}

// -----

// A loopback destination-selected record can send because its destination node
// is also the record source.

// CHECK-LABEL: func.func @destination_selected_loopback_send
func.func @destination_selected_loopback_send()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  // CHECK: %[[CB:.*]] = ttl.bind_cb
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  // CHECK: ttl.pipenet_foreach_dst
  ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_dst):
    // CHECK: ^bb0(%[[PIPE:.*]]: !ttl.selected_pipe_dst)
    // CHECK-NEXT: %[[COPY:.*]] = ttl.copy %[[CB]], %[[PIPE]]
    %copy = ttl.copy %cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
           !ttl.selected_pipe_dst)
        -> !ttl.transfer_handle<write>
    // CHECK-NEXT: ttl.wait %[[COPY]]
    ttl.wait %copy : !ttl.transfer_handle<write>
    ttl.yield
  }
  func.return
}

// -----

// Selected-pipe ops store the record index separately from coordinates.

// CHECK-LABEL: func.func @select_pipe_ops
func.func @select_pipe_ops() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %false = arith.constant false
  // CHECK: ttl.select_pipe_src record(%{{.*}})
  %src = ttl.select_pipe_src record(%c0) src (%c0, %c0) dst (%c1, %c0) to (%c1, %c0)
      num_dests (%c1) src_in_dst (%false) devices (%c0, %c0)
      {records = #ttl.pipenet_records<net 0 pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>
      ]>} : !ttl.selected_pipe_src
  // CHECK: ttl.select_pipe_dst record(%{{.*}})
  %dst = ttl.select_pipe_dst record(%c0) src (%c0, %c0) dst (%c1, %c0) to (%c1, %c0)
      num_dests (%c1) src_in_dst (%false) devices (%c0, %c0)
      {records = #ttl.pipenet_records<net 0 pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>
      ]>} : !ttl.selected_pipe_dst
  func.return
}
