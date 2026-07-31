// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Verifies that table-driven callbacks execute in record order and that one
// selected loopback record can provide both receive and send endpoints.

// Each callback releases the single receiver DFB slot before the next
// identical record executes.
// CHECK-LABEL: func.func @record_order_loopback
// CHECK: scf.for
// CHECK: scf.if
// CHECK: ttkernel.cb_reserve_back
// CHECK: ttkernel.noc_async_write
// CHECK: ttkernel.cb_push_back
// CHECK: ttkernel.cb_wait_front
// CHECK: ttkernel.cb_pop_front
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @record_order_loopback()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_dst attributes {
        records = #ttl.pipenet_records<net 0 name "loopback" pipes [
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
          #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>
        ]>} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %reserve = ttl.cb_reserve %recv_cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %receive = ttl.copy %pipe, %reserve
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.selected_pipe_dst)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.wait %receive : !ttl.transfer_handle
      ttl.cb_push %recv_cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      %ready = ttl.cb_wait %recv_cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %recv_cb
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      ttl.yield
    }
    func.return
  }
}
