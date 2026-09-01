// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s --implicit-check-not=ttl.pipenet_foreach

// Verifies that table-driven callbacks execute in record order and that one
// selected loopback record can provide both receive and send endpoints.

// Each callback releases the single receiver DFB slot before the next
// identical record executes.
// CHECK-LABEL: func.func @record_order_loopback
// CHECK-DAG: %[[ONE_I32:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[SEND_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-DAG: %[[RECEIVER_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: %[[RECORD_OFFSET:.*]] = ttkernel.experimental.constant_table_lookup %[[NODE_INDEX:.*]], [0] : index
// CHECK: %[[RECORD_COUNT:.*]] = ttkernel.experimental.constant_table_lookup %[[NODE_INDEX]], [5] : index
// CHECK: %[[RECORD_END:.*]] = arith.addi %[[RECORD_OFFSET]], %[[RECORD_COUNT]] : index
// CHECK: scf.for %[[SLICE_INDEX:.*]] = %[[RECORD_OFFSET]] to %[[RECORD_END]] step %[[ONE]] {
// CHECK: %[[RECORD_INDEX:.*]] = ttkernel.experimental.constant_table_lookup %[[SLICE_INDEX]], [0, 1, 2, 3, 4] : index
// CHECK: ttkernel.experimental.constant_table_lookup %[[RECORD_INDEX]], [0, 0, 0, 0, 0]
// CHECK: ttkernel.cb_reserve_back(%[[RECEIVER_DFB]], %[[ONE_I32]])
// CHECK: scf.if
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: %[[SEND_WRITE_PTR:.*]] = ttkernel.get_write_ptr(%[[SEND_DFB]])
// CHECK: ttkernel.noc_async_write %[[SEND_WRITE_PTR]], core[{{.*}}]
// CHECK: ttkernel.noc_async_write_barrier
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.cb_push_back(%[[RECEIVER_DFB]], %[[ONE_I32]])
// CHECK-NEXT: ttkernel.cb_wait_front(%[[RECEIVER_DFB]], %[[ONE_I32]])
// CHECK-NEXT: ttkernel.cb_pop_front(%[[RECEIVER_DFB]], %[[ONE_I32]])
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @record_order_loopback()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %send_cb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %recv_cb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
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
          -> !ttl.receive_request
      %send = ttl.copy %send_cb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.selected_pipe_dst)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.wait %receive : !ttl.receive_request
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
