// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// Summary: Verifies PipeNet foreach ops and selected-pipe region
// arguments.

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
        -> !ttl.transfer_handle
    ttl.wait %xf : !ttl.transfer_handle
    ttl.yield
  }
  func.return
}

// -----

// A destination foreach body can also contain the explicit receive post/wait
// ops produced by copy-wait canonicalization.

// CHECK-LABEL: func.func @foreach_dst_explicit_receive
func.func @foreach_dst_explicit_receive() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
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
    // CHECK: %[[POST:.*]] = ttl.pipe_recv_post %[[PIPE]], %{{.*}}
    // CHECK: ttl.pipe_recv_wait %[[POST]], %[[PIPE]], %{{.*}}
    %xf = ttl.pipe_recv_post %pipe, %recv
        : (!ttl.selected_pipe_dst,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
        -> !ttl.transfer_handle
    ttl.pipe_recv_wait %xf, %pipe, %recv
        : !ttl.transfer_handle,
          !ttl.selected_pipe_dst,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  }
  func.return
}
