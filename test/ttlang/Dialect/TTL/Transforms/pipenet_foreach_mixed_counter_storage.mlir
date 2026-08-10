// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Verifies that table-driven receiver lowering selects the correct completion
// address when its records use both local and global semaphore storage.

// CHECK: module attributes
// CHECK-SAME: ttl.pipe_global_semaphore_count = 2 : i64
// CHECK-SAME: ttl.pipe_sync_semaphore_count = 16 : i64

module attributes {ttl.launch_grid = array<i64: 18, 1>} {

// CHECK-LABEL: func.func @mixed_completion_counter_receiver
// CHECK: memref.alloca() : memref<17xi32>
// CHECK: scf.for
// CHECK: %[[LOCAL_ADDRESS:.*]] = ttkernel.get_semaphore
// CHECK: %[[TYPED_LOCAL_ADDRESS:.*]] = ttkernel.cast_to_l1_addr %[[LOCAL_ADDRESS]]
// CHECK: %[[GLOBAL_ADDRESS:.*]] = ttkernel.get_common_arg_val
// CHECK: %[[TYPED_GLOBAL_ADDRESS:.*]] = ttkernel.cast_to_l1_addr %[[GLOBAL_ADDRESS]]
// CHECK: %[[COMPLETION_ADDRESS:.*]] = arith.select %{{.*}}, %[[TYPED_GLOBAL_ADDRESS]], %[[TYPED_LOCAL_ADDRESS]]
// CHECK: %[[COMPLETION_POINTER:.*]] = ttkernel.reinterpret_cast{{.*}}(%[[COMPLETION_ADDRESS]])
// CHECK: ttkernel.experimental.semaphore_wait_min(%[[COMPLETION_POINTER]]
func.func @mixed_completion_counter_receiver()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %recv_dfb = ttl.bind_cb {cb_index = 1, block_count = 17}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 17>
  ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 name "mixed_completion" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 2, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 3, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 4, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 5, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 6, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 7, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 8, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 9, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 10, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 11, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 12, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 13, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 14, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 15, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 16, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_dst):
    %recv = ttl.cb_reserve %recv_dfb
        : <[1, 1], !ttcore.tile<32x32, f32>, 17>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %transfer = ttl.copy %pipe, %recv
        : (!ttl.selected_pipe_dst,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.receive_request
    ttl.wait %transfer : !ttl.receive_request
    ttl.cb_push %recv_dfb : <[1, 1], !ttcore.tile<32x32, f32>, 17>
    ttl.yield
  }
  func.return
}

func.func @mixed_completion_counter_senders()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 name "mixed_completion" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 2, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 3, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 4, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 5, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 6, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 7, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 8, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 9, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 10, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 11, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 12, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 13, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 14, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 15, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>,
        #ttl.pipe_record<srcX = 16, srcY = 0, dstStartX = 17, dstStartY = 0, dstEndX = 17, dstEndY = 0>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    %transfer = ttl.copy %send_dfb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
           !ttl.selected_pipe_src)
        -> !ttl.transfer_handle<write>
    ttl.wait %transfer : !ttl.transfer_handle<write>
    ttl.yield
  }
  func.return
}

}
