// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Verifies that local PipeNet callbacks enumerate only the records for the
// current launch node while preserving attribute order within each node slice.

module attributes {ttl.launch_grid = array<i64: 3, 2>} {

// CHECK-LABEL: func.func @participant_sender
// CHECK: %[[NODE_X:.*]] = ttkernel.my_logical_x
// CHECK: %[[NODE_Y:.*]] = ttkernel.my_logical_y
// CHECK: %[[ROW_OFFSET:.*]] = arith.muli %[[NODE_Y]], %{{.*}} : index
// CHECK: %[[NODE_INDEX:.*]] = arith.addi %[[ROW_OFFSET]], %[[NODE_X]] : index
// CHECK: %[[RECORD_OFFSET:.*]] = ttkernel.experimental.constant_table_lookup %[[NODE_INDEX]], [0, 3, 4, 4, 4, 4] : index
// CHECK: %[[RECORD_COUNT:.*]] = ttkernel.experimental.constant_table_lookup %[[NODE_INDEX]], [3, 1, 0, 0, 0, 1] : index
// CHECK: %[[RECORD_END:.*]] = arith.addi %[[RECORD_OFFSET]], %[[RECORD_COUNT]] : index
// CHECK: scf.for %[[PARTICIPANT_INDEX:.*]] = %[[RECORD_OFFSET]] to %[[RECORD_END]]
// CHECK: %[[RECORD_INDEX:.*]] = ttkernel.experimental.constant_table_lookup %[[PARTICIPANT_INDEX]], [0, 2, 4, 3, 1] : index
// CHECK-NOT: scf.if
// CHECK: return
func.func @participant_sender()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.pipenet_foreach_src attributes {
      records = #ttl.pipenet_records<net 0 name "participant_net" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 2, dstEndY = 0, isCollective = true>,
        #ttl.pipe_record<srcX = 2, srcY = 1, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 1, isCollective = true>,
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 1, dstEndX = 2, dstEndY = 1, isCollective = true>,
        #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 2, dstStartY = 0, dstEndX = 2, dstEndY = 1, isCollective = true>,
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0, isCollective = true>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_src):
    %dst_start_x, %dst_start_y, %dst_end_x, %dst_end_y =
        ttl.selected_pipe_destination_coordinates %pipe
        : !ttl.selected_pipe_src
    "ttl.dprint"(%dst_start_x) {
        fmt = "source participant destination x={}", mode = "scalar"}
        : (index) -> ()
    ttl.yield
  }
  func.return
}

// CHECK-LABEL: func.func @participant_receiver
// CHECK: %[[NODE_X:.*]] = ttkernel.my_logical_x
// CHECK: %[[NODE_Y:.*]] = ttkernel.my_logical_y
// CHECK: %[[ROW_OFFSET:.*]] = arith.muli %[[NODE_Y]], %{{.*}} : index
// CHECK: %[[NODE_INDEX:.*]] = arith.addi %[[ROW_OFFSET]], %[[NODE_X]] : index
// CHECK: %[[RECORD_OFFSET:.*]] = ttkernel.experimental.constant_table_lookup %[[NODE_INDEX]], [0, 1, 3, 5, 7, 8] : index
// CHECK: %[[RECORD_COUNT:.*]] = ttkernel.experimental.constant_table_lookup %[[NODE_INDEX]], [1, 2, 2, 2, 1, 2] : index
// CHECK: %[[RECORD_END:.*]] = arith.addi %[[RECORD_OFFSET]], %[[RECORD_COUNT]] : index
// CHECK: scf.for %[[PARTICIPANT_INDEX:.*]] = %[[RECORD_OFFSET]] to %[[RECORD_END]]
// CHECK: %[[RECORD_INDEX:.*]] = ttkernel.experimental.constant_table_lookup %[[PARTICIPANT_INDEX]], [1, 0, 4, 0, 3, 1, 2, 2, 2, 3] : index
// CHECK-NOT: scf.if
// CHECK: return
func.func @participant_receiver()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.pipenet_foreach_dst attributes {
      records = #ttl.pipenet_records<net 0 name "participant_net" pipes [
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 2, dstEndY = 0, isCollective = true>,
        #ttl.pipe_record<srcX = 2, srcY = 1, dstStartX = 0, dstStartY = 0, dstEndX = 0, dstEndY = 1, isCollective = true>,
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 1, dstEndX = 2, dstEndY = 1, isCollective = true>,
        #ttl.pipe_record<srcX = 1, srcY = 0, dstStartX = 2, dstStartY = 0, dstEndX = 2, dstEndY = 1, isCollective = true>,
        #ttl.pipe_record<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0, dstEndX = 1, dstEndY = 0, isCollective = true>
      ]>} {
  ^bb0(%pipe: !ttl.selected_pipe_dst):
    %src_x, %src_y = ttl.selected_pipe_source_coordinates %pipe
        : !ttl.selected_pipe_dst
    "ttl.dprint"(%src_x) {
        fmt = "destination participant source x={}", mode = "scalar"}
        : (index) -> ()
    ttl.yield
  }
  func.return
}

}
