// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verifies compact PipeNet foreach lowering through pipe-transfer IR.

// Source foreach lowering emits one loop and one send protocol body.

// CHECK-LABEL: func.func @foreach_src_send_compact
// CHECK: memref.alloca
// CHECK: scf.for
// CHECK-COUNT-1: ttkernel.noc_async_write %
// CHECK-NOT: ttl.pipenet_foreach_src
// CHECK-NOT: ttl.select_pipe_src
func.func @foreach_src_send_compact()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 name "row_net" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    %xf = ttl.copy %cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.selected_pipe_src)
        -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
    ttl.yield
  }
  func.return
}

// -----

// Destination foreach lowering emits one loop and one receive protocol body.

// CHECK-LABEL: func.func @foreach_dst_receive_compact
// CHECK: memref.alloca
// CHECK: scf.for
// CHECK-COUNT-1: ttkernel.noc_inline_dw_write(
// CHECK-COUNT-1: ttkernel.experimental::semaphore_wait_min(
// CHECK-NOT: ttl.pipenet_foreach_dst
// CHECK-NOT: ttl.select_pipe_dst
func.func @foreach_dst_receive_compact()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 name "row_net" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_dst):
    %recv = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf = ttl.copy %pipe, %recv
        : (!ttl.selected_pipe_dst,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf : !ttl.transfer_handle
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    ttl.yield
  }
  func.return
}
