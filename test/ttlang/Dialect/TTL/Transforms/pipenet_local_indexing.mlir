// Summary: Verifies local PipeNet callbacks and role predicates use launch-node
// index tables instead of scanning every record.
// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

module attributes {ttl.launch_grid = array<i64: 8, 8>} {

func.func private @local_index_receiver()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %receive_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
      {dfb_id = 1 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 name "local_index" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 3, dstStartX = 0, dstStartY = 2, dstEndX = 0, dstEndY = 2>,
        #ttl.pipe_record<srcX = 0, srcY = 5, dstStartX = 0, dstStartY = 4, dstEndX = 0, dstEndY = 4>,
        #ttl.pipe_record<srcX = 0, srcY = 7, dstStartX = 0, dstStartY = 6, dstEndX = 0, dstEndY = 6>,
        #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_dst):
    %receive_block = ttl.cb_reserve %receive_dfb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %receive = ttl.copy %pipe, %receive_block
        : (!ttl.selected_pipe_dst,
           tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.transfer_handle
    ttl.wait %receive : !ttl.transfer_handle
    ttl.cb_push %receive_dfb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    ttl.yield
  }
  func.return
}

// CHECK-LABEL: func.func @local_index_sender
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-NOT: ttkernel.load_from_l1
// CHECK: %[[NODE_X:.*]] = ttkernel.my_logical_x
// CHECK: %[[NODE_Y:.*]] = ttkernel.my_logical_y
// CHECK: %[[ROW_OFFSET:.*]] = arith.muli %[[NODE_Y]]
// CHECK: %[[NODE_INDEX:.*]] = arith.addi %[[ROW_OFFSET]], %[[NODE_X]]
// CHECK: %[[NEXT_NODE:.*]] = arith.addi %[[NODE_INDEX]]
// CHECK: %[[LOWER:.*]] = ttkernel.experimental.constant_table_lookup %[[NODE_INDEX]]
// CHECK: %[[UPPER:.*]] = ttkernel.experimental.constant_table_lookup %[[NEXT_NODE]]
// CHECK: scf.for %[[ACTIVE_RECORD:.*]] = %[[LOWER]] to %[[UPPER]]
// CHECK: %[[RECORD_INDEX:.*]] = ttkernel.experimental.constant_table_lookup %[[ACTIVE_RECORD]]
// CHECK: ttkernel.noc_async_write {{.*}} posted true :
// CHECK: ttkernel.noc_inline_dw_write({{.*}}) posted true :
// CHECK-NEXT: ttkernel.noc_async_writes_flushed({{.*}}) posted true :
// CHECK-NOT: ttkernel.noc_async_write_barrier
// CHECK-NOT: ttkernel.noc_async_atomic_barrier
// CHECK: ttl.pipenet_local_record_loop
// CHECK: return
func.func @local_index_sender()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %send_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
      {dfb_id = 0 : index}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 name "local_index" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 1, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 0>,
        #ttl.pipe_record<srcX = 0, srcY = 3, dstStartX = 0, dstStartY = 2, dstEndX = 0, dstEndY = 2>,
        #ttl.pipe_record<srcX = 0, srcY = 5, dstStartX = 0, dstStartY = 4, dstEndX = 0, dstEndY = 4>,
        #ttl.pipe_record<srcX = 0, srcY = 7, dstStartX = 0, dstStartY = 6, dstEndX = 0, dstEndY = 6>,
        #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 1, dstEndX = 1, dstEndY = 1>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    %send = ttl.copy %send_dfb, %pipe
        : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
           !ttl.selected_pipe_src)
        -> !ttl.transfer_handle<write>
    ttl.wait %send : !ttl.transfer_handle<write>
    ttl.yield
  }
  func.return
}

// CHECK-LABEL: func.func @local_role_predicates
// CHECK: ttkernel.experimental.constant_table_lookup
// CHECK: arith.cmpi ne
// CHECK: ttkernel.experimental.constant_table_lookup
// CHECK: arith.cmpi ne
// CHECK: arith.ori
// CHECK: return
func.func @local_role_predicates() -> i1
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %is_source = ttl.is_src {pipe_net_id = 0 : i64}
  %is_destination = ttl.is_dst {pipe_net_id = 0 : i64}
  %is_active = arith.ori %is_source, %is_destination : i1
  func.return %is_active : i1
}

}
