// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-form-pipe-transports,convert-ttl-to-ttkernel)' | FileCheck %s
// The grouping pass must not expand any record-selected callback transfer.
// RUN: ttlang-opt %s --split-input-file -ttl-form-pipe-transports | FileCheck %s --check-prefix=FORM

// Summary: Verifies direct and table-driven PipeNet callback lowering through
// pipe-transfer IR, including when static transport grouping runs first.

// Four source records exercise the inclusive direct-lowering boundary. Each
// record emits one static guard and one payload write without a record loop.

func.func private @foreach_src_send_direct_receiver()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 name "row_net" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>,
        #ttl.pipe_record<srcX = 0, srcY = 2, dstStartX = 1, dstStartY = 2, dstEndX = 1, dstEndY = 2>,
        #ttl.pipe_record<srcX = 0, srcY = 3, dstStartX = 1, dstStartY = 3, dstEndX = 1, dstEndY = 3>
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
// CHECK-COUNT-4: ttkernel.noc_async_write %
// CHECK-NOT: scf.for
// CHECK-NOT: ttl.pipenet_foreach_src
// CHECK-NOT: ttl.select_pipe_src
// CHECK-NOT: ttkernel.noc_async_write %
// CHECK: return
func.func @foreach_src_send_direct()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 name "row_net" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>,
        #ttl.pipe_record<srcX = 0, srcY = 2, dstStartX = 1, dstStartY = 2, dstEndX = 1, dstEndY = 2>,
        #ttl.pipe_record<srcX = 0, srcY = 3, dstStartX = 1, dstStartY = 3, dstEndX = 1, dstEndY = 3>
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

// Four destination records exercise the inclusive direct-lowering boundary.
// Each record emits one address publication and one completion wait.

func.func private @foreach_dst_receive_direct_sender()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 name "row_net" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>,
        #ttl.pipe_record<srcX = 0, srcY = 2, dstStartX = 1, dstStartY = 2, dstEndX = 1, dstEndY = 2>,
        #ttl.pipe_record<srcX = 0, srcY = 3, dstStartX = 1, dstStartY = 3, dstEndX = 1, dstEndY = 3>
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
// CHECK: ttkernel.noc_inline_dw_write(
// CHECK: ttkernel.experimental.semaphore_wait_min(
// CHECK: ttkernel.noc_inline_dw_write(
// CHECK: ttkernel.experimental.semaphore_wait_min(
// CHECK-NOT: scf.for
// CHECK-NOT: ttl.pipenet_foreach_dst
// CHECK-NOT: ttl.select_pipe_dst
// CHECK-NOT: ttkernel.noc_inline_dw_write
// CHECK-NOT: ttkernel.experimental.semaphore_wait_min
// CHECK: return
func.func @foreach_dst_receive_direct()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 name "row_net" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>,
        #ttl.pipe_record<srcX = 0, srcY = 2, dstStartX = 1, dstStartY = 2, dstEndX = 1, dstEndY = 2>,
        #ttl.pipe_record<srcX = 0, srcY = 3, dstStartX = 1, dstStartY = 3, dstEndX = 1, dstEndY = 3>
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

// Direct record conditions use TTKernel logical coordinates. Pipe analysis
// must restrict each receive to its matching row before proving the shared
// outer reservation has one receive and wait.

module attributes {ttl.launch_grid = array<i64: 3, 2>} {

func.func @foreach_dst_outer_reserve_direct_sender()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %is_src = ttl.is_src {pipe_net_id = 0 : i64}
  scf.if %is_src {
    ttl.pipenet_foreach_src attributes {
        records = #ttl.pipenet_records<net 0 name "row_net" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 2, dstEndY = 0, isCollective = true>,
          #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 2, dstEndY = 1, isCollective = true>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %xf = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %xf : !ttl.transfer_handle<write>
      ttl.yield
    }
  }
  func.return
}

// CHECK-LABEL: func.func @foreach_dst_outer_reserve_direct
// CHECK-NOT: scf.for
// CHECK-COUNT-2: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.cb_push_back
// CHECK-NOT: scf.for
// CHECK: return
func.func @foreach_dst_outer_reserve_direct()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %is_dst = ttl.is_dst {pipe_net_id = 0 : i64}
  scf.if %is_dst {
    %recv = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.pipenet_foreach_dst attributes {
        records = #ttl.pipenet_records<net 0 name "row_net" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 2, dstEndY = 0, isCollective = true>,
          #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 2, dstEndY = 1, isCollective = true>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %xf = ttl.copy %pipe, %recv
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %xf : !ttl.transfer_handle
      ttl.yield
    }
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  }
  func.return
}

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

// Static transport grouping leaves record-selected callbacks for table-driven
// lowering because one operation represents several possible transfers.
// FORM-LABEL: func.func @foreach_src_send_table_driven
// FORM: ttl.pipenet_foreach_src
// FORM: ^bb0(%[[FORM_PIPE:.*]]: !ttl.selected_pipe_src):
// FORM: %[[FORM_SEND:.*]] = ttl.copy %{{.*}}, %[[FORM_PIPE]]
// FORM-NEXT: ttl.wait %[[FORM_SEND]]
// FORM: return

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

// -----

// A selected receive wait completes before a push after the record loop. The
// graph already represents the one matching iteration for each receiver.

module attributes {ttl.launch_grid = array<i64: 3, 5>} {

func.func @foreach_dst_outer_reserve_sender()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %is_src = ttl.is_src {pipe_net_id = 0 : i64}
  scf.if %is_src {
    ttl.pipenet_foreach_src attributes {
        records = #ttl.pipenet_records<net 0 name "row_net" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 2, dstEndY = 0, isCollective = true>,
          #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 2, dstEndY = 1, isCollective = true>,
          #ttl.pipe_record<srcX = 0, srcY = 2, dstStartX = 1, dstStartY = 2, dstEndX = 2, dstEndY = 2, isCollective = true>,
          #ttl.pipe_record<srcX = 0, srcY = 3, dstStartX = 1, dstStartY = 3, dstEndX = 2, dstEndY = 3, isCollective = true>,
          #ttl.pipe_record<srcX = 0, srcY = 4, dstStartX = 1, dstStartY = 4, dstEndX = 2, dstEndY = 4, isCollective = true>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %xf = ttl.copy %cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %xf : !ttl.transfer_handle<write>
      ttl.yield
    }
  }
  func.return
}

// CHECK-LABEL: func.func @foreach_dst_outer_reserve
// CHECK: scf.for
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.cb_push_back
// CHECK: return
func.func @foreach_dst_outer_reserve()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %is_dst = ttl.is_dst {pipe_net_id = 0 : i64}
  scf.if %is_dst {
    %recv = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.pipenet_foreach_dst attributes {
        records = #ttl.pipenet_records<net 0 name "row_net" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 2, dstEndY = 0, isCollective = true>,
          #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 1, dstStartY = 1, dstEndX = 2, dstEndY = 1, isCollective = true>,
          #ttl.pipe_record<srcX = 0, srcY = 2, dstStartX = 1, dstStartY = 2, dstEndX = 2, dstEndY = 2, isCollective = true>,
          #ttl.pipe_record<srcX = 0, srcY = 3, dstStartX = 1, dstStartY = 3, dstEndX = 2, dstEndY = 3, isCollective = true>,
          #ttl.pipe_record<srcX = 0, srcY = 4, dstStartX = 1, dstStartY = 4, dstEndX = 2, dstEndY = 4, isCollective = true>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %xf = ttl.copy %pipe, %recv
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %xf : !ttl.transfer_handle
      ttl.yield
    }
    ttl.cb_push %cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  }
  func.return
}

}

// -----

// Lowering an outer direct callback clones its nested table-driven callback.
// Both generated record-selection regions must remain visible to pipe analysis.

module attributes {ttl.launch_grid = array<i64: 2, 5>} {

// CHECK-LABEL: func.func @nested_foreach_sender
// CHECK: scf.if
// CHECK: scf.for
// CHECK: ttkernel.noc_async_write
// CHECK-NOT: ttl.pipenet_foreach
// CHECK: return
func.func @nested_foreach_sender()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %send_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 1 name "outer" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 4, isCollective = true>
      ]>} {
  ^bb0(%outer_pipe: !ttl.selected_pipe_src):
    ttl.pipenet_foreach_src attributes {
        records = #ttl.pipenet_records<net 0 name "inner" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 2, dstEndX = 1, dstEndY = 2>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 3, dstEndX = 1, dstEndY = 3>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 4, dstEndX = 1, dstEndY = 4>
        ]>} {
    ^bb0(%inner_pipe: !ttl.selected_pipe_src):
      %send = ttl.copy %send_cb, %inner_pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.yield
    }
    ttl.yield
  }
  func.return
}

// CHECK-LABEL: func.func @nested_foreach_receiver
// CHECK: scf.if
// CHECK: scf.for
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK-NOT: ttl.pipenet_foreach
// CHECK: return
func.func @nested_foreach_receiver()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 1 name "outer" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 4, isCollective = true>
      ]>} {
  ^bb0(%outer_pipe: !ttl.selected_pipe_dst):
    ttl.pipenet_foreach_dst attributes {
        records = #ttl.pipenet_records<net 0 name "inner" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 2, dstEndX = 1, dstEndY = 2>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 3, dstEndX = 1, dstEndY = 3>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 4, dstEndX = 1, dstEndY = 4>
        ]>} {
    ^bb0(%inner_pipe: !ttl.selected_pipe_dst):
      %reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %receive = ttl.copy %inner_pipe, %reserve
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %receive : !ttl.transfer_handle
      ttl.cb_push %recv_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      ttl.yield
    }
    ttl.yield
  }
  func.return
}

}

// -----

// A loopback collective publishes the source receiver address with a direct
// L1 store and publishes the remote receiver address with a NoC write.

module attributes {ttl.launch_grid = array<i64: 5, 1>} {

func.func @loopback_collective_sender()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %send_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 name "loopback" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0, isCollective = true>,
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0, isCollective = true>,
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 2, dstStartY = 0, dstEndX = 2, dstEndY = 0, isCollective = true>,
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 3, dstStartY = 0, dstEndX = 3, dstEndY = 0, isCollective = true>,
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 4, dstStartY = 0, dstEndX = 4, dstEndY = 0, isCollective = true>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.selected_pipe_src)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    ttl.yield
  }
  func.return
}

// CHECK-LABEL: func.func @loopback_collective_receiver
// CHECK-DAG: ttkernel.store_to_l1
// CHECK-DAG: ttkernel.noc_inline_dw_write
// CHECK: return
func.func @loopback_collective_receiver()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 name "loopback" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0, isCollective = true>,
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0, isCollective = true>,
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 2, dstStartY = 0, dstEndX = 2, dstEndY = 0, isCollective = true>,
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 3, dstStartY = 0, dstEndX = 3, dstEndY = 0, isCollective = true>,
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 4, dstStartY = 0, dstEndX = 4, dstEndY = 0, isCollective = true>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_dst):
    %reserve = ttl.cb_reserve %recv_cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %receive = ttl.copy %pipe, %reserve
        : (!ttl.selected_pipe_dst,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %receive : !ttl.transfer_handle
    ttl.cb_push %recv_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.yield
  }
  func.return
}

}

// -----

// Static and selected transfers can coexist in one module. Both contribute
// protocol operations and synchronization resources to the shared plan.

module attributes {ttl.launch_grid = array<i64: 4, 1>} {

// Static transfers become explicit IR for grouping while the selected transfer
// remains attached to its record table.
// FORM-LABEL: func.func @mixed_static_and_selected_sender
// FORM: %[[STATIC_TRANSFER:.*]] = ttl.pipe_transfer.create
// FORM: ttl.pipe_transfer.send %[[STATIC_TRANSFER]],
// FORM: ttl.pipenet_foreach_src
// FORM: ^bb0(%[[SELECTED_PIPE:.*]]: !ttl.selected_pipe_src):
// FORM-NOT: ttl.pipe_transfer
// FORM: %[[SELECTED_SEND:.*]] = ttl.copy %{{.*}}, %[[SELECTED_PIPE]]
// FORM-NEXT: ttl.wait %[[SELECTED_SEND]]
// FORM: return

// CHECK-LABEL: func.func @mixed_static_and_selected_sender
// CHECK-COUNT-2: ttkernel.noc_async_write
// CHECK-NOT: ttl.pipenet_foreach
// CHECK: return
func.func @mixed_static_and_selected_sender()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %send_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %static_pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  ttl.if_src %static_pipe
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %send = ttl.copy %send_cb, %static_pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
  }
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 1 name "selected" pipes [
        #ttl.pipe_record<srcX = 2, srcY = 0, dstStartX = 3, dstStartY = 0, dstEndX = 3, dstEndY = 0>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    %send = ttl.copy %send_cb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.selected_pipe_src)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    ttl.yield
  }
  func.return
}

func.func @mixed_static_and_selected_receiver()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  %static_pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  ttl.if_dst %static_pipe
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
    %reserve = ttl.cb_reserve %recv_cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %receive = ttl.copy %static_pipe, %reserve
        : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %receive : !ttl.transfer_handle
    ttl.cb_push %recv_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
  }
  ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 1 name "selected" pipes [
        #ttl.pipe_record<srcX = 2, srcY = 0, dstStartX = 3, dstStartY = 0, dstEndX = 3, dstEndY = 0>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_dst):
    %reserve = ttl.cb_reserve %recv_cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %receive = ttl.copy %pipe, %reserve
        : (!ttl.selected_pipe_dst,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %receive : !ttl.transfer_handle
    ttl.cb_push %recv_cb : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.yield
  }
  func.return
}

}
