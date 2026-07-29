// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verifies direct and table-driven PipeNet callback lowering through
// pipe-transfer IR.

// Small source foreach lowering emits direct static guards.

func.func private @foreach_src_send_direct_receiver()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 1, block_count = 2}
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

// CHECK-LABEL: func.func @foreach_src_send_direct
// CHECK-NOT: scf.for
// CHECK-COUNT-2: ttkernel.noc_async_write %
// CHECK-NOT: scf.for
// CHECK-NOT: ttl.pipenet_foreach_src
// CHECK-NOT: ttl.select_pipe_src
// CHECK: return
func.func @foreach_src_send_direct()
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

// Small destination foreach lowering emits direct static guards.

func.func private @foreach_dst_receive_direct_sender()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 1, block_count = 2}
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

// CHECK-LABEL: func.func @foreach_dst_receive_direct
// CHECK-NOT: scf.for
// CHECK: ttkernel.noc_inline_dw_write(
// CHECK: ttkernel.experimental.semaphore_wait_min(
// CHECK: ttkernel.noc_inline_dw_write(
// CHECK: ttkernel.experimental.semaphore_wait_min(
// CHECK-NOT: scf.for
// CHECK-NOT: ttl.pipenet_foreach_dst
// CHECK-NOT: ttl.select_pipe_dst
// CHECK: return
func.func @foreach_dst_receive_direct()
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

// -----

// Five-record source foreach lowering emits one loop and one send protocol
// body.

func.func private @foreach_src_send_table_driven_receiver()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 1, block_count = 5}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 5>
  ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 name "row_net" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>,
        #ttl.pipe_record<srcX = 0, srcY = 2, dstStartX = 1, dstStartY = 2, dstEndX = 1, dstEndY = 2>,
        #ttl.pipe_record<srcX = 0, srcY = 3, dstStartX = 1, dstStartY = 3, dstEndX = 1, dstEndY = 3>,
        #ttl.pipe_record<srcX = 0, srcY = 4, dstStartX = 1, dstStartY = 4, dstEndX = 1, dstEndY = 4>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_dst):
    %recv = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 5>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %xf = ttl.copy %pipe, %recv
        : (!ttl.selected_pipe_dst,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %xf : !ttl.transfer_handle
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 5>
    ttl.yield
  }
  func.return
}

// CHECK-LABEL: func.func @foreach_src_send_table_driven
// CHECK: scf.for
// CHECK: ttkernel.experimental.constant_table_lookup
// CHECK: ttkernel.experimental.semaphore_wait(
// CHECK-COUNT-1: ttkernel.noc_async_write %
// CHECK-NOT: ttl.pipenet_foreach_src
// CHECK-NOT: ttl.select_pipe_src
// CHECK: return
func.func @foreach_src_send_table_driven()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 name "row_net" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>,
        #ttl.pipe_record<srcX = 0, srcY = 2, dstStartX = 1, dstStartY = 2, dstEndX = 1, dstEndY = 2>,
        #ttl.pipe_record<srcX = 0, srcY = 3, dstStartX = 1, dstStartY = 3, dstEndX = 1, dstEndY = 3>,
        #ttl.pipe_record<srcX = 0, srcY = 4, dstStartX = 1, dstStartY = 4, dstEndX = 1, dstEndY = 4>
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

// Five-record destination foreach lowering emits one loop and one receive
// protocol body.

func.func private @foreach_dst_receive_table_driven_sender()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 name "row_net" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>,
        #ttl.pipe_record<srcX = 0, srcY = 2, dstStartX = 1, dstStartY = 2, dstEndX = 1, dstEndY = 2>,
        #ttl.pipe_record<srcX = 0, srcY = 3, dstStartX = 1, dstStartY = 3, dstEndX = 1, dstEndY = 3>,
        #ttl.pipe_record<srcX = 0, srcY = 4, dstStartX = 1, dstStartY = 4, dstEndX = 1, dstEndY = 4>
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

// CHECK-LABEL: func.func @foreach_dst_receive_table_driven
// CHECK: memref.alloca
// CHECK: scf.for
// CHECK-COUNT-1: ttkernel.noc_inline_dw_write(
// CHECK-COUNT-1: ttkernel.experimental.semaphore_wait_min(
// CHECK-NOT: ttl.pipenet_foreach_dst
// CHECK-NOT: ttl.select_pipe_dst
// CHECK: return
func.func @foreach_dst_receive_table_driven()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 name "row_net" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>,
        #ttl.pipe_record<srcX = 0, srcY = 2, dstStartX = 1, dstStartY = 2, dstEndX = 1, dstEndY = 2>,
        #ttl.pipe_record<srcX = 0, srcY = 3, dstStartX = 1, dstStartY = 3, dstEndX = 1, dstEndY = 3>,
        #ttl.pipe_record<srcX = 0, srcY = 4, dstStartX = 1, dstStartY = 4, dstEndX = 1, dstEndY = 4>
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
