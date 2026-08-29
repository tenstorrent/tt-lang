// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verify dense edge-major, grid-major device records select only the
// current device's edge blocks and current node's record.

// Each device is the source of one edge. The lowering indexes a compact
// device-to-edge range, then combines the selected edge block with the
// row-major logical node index. It does not scan all eight records or compare
// their endpoint device/node coordinates at runtime.
// CHECK-LABEL: func.func @sender()
// CHECK: %[[NODE_X:.*]] = ttkernel.my_logical_x_
// CHECK-NEXT: %[[NODE_Y:.*]] = ttkernel.my_logical_y_
// CHECK-NEXT: %[[NODE_ROW:.*]] = arith.muli %[[NODE_Y]], %{{.*}} : index
// CHECK-NEXT: %[[NODE_INDEX:.*]] = arith.addi %[[NODE_ROW]], %[[NODE_X]] : index
// CHECK: %[[DEVICE_I32:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[DEVICE:.*]] = arith.index_cast %[[DEVICE_I32]] : i32 to index
// CHECK-NEXT: %[[NEXT_DEVICE:.*]] = arith.addi %[[DEVICE]], %{{.*}} : index
// CHECK-NEXT: %[[LOWER:.*]] = ttkernel.experimental.constant_table_lookup %[[DEVICE]], [0, 1, 2] : index
// CHECK-NEXT: %[[UPPER:.*]] = ttkernel.experimental.constant_table_lookup %[[NEXT_DEVICE]], [0, 1, 2] : index
// CHECK: scf.for %[[EDGE_POSITION:.*]] = %[[LOWER]] to %[[UPPER]] step
// CHECK-NEXT: %[[EDGE_BLOCK:.*]] = ttkernel.experimental.constant_table_lookup %[[EDGE_POSITION]], [0, 1] : index
// CHECK-NEXT: %[[EDGE_OFFSET:.*]] = arith.muli %[[EDGE_BLOCK]], %{{.*}} : index
// CHECK-NEXT: %[[RECORD:.*]] = arith.addi %[[EDGE_OFFSET]], %[[NODE_INDEX]] : index
// CHECK-NOT: arith.cmpi
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc

// Each device is also the destination of one reverse edge. Destination
// lowering uses the same compact range and row-major record calculation.
// CHECK-LABEL: func.func @receiver()
// CHECK: %[[DST_NODE_X:.*]] = ttkernel.my_logical_x_
// CHECK-NEXT: %[[DST_NODE_Y:.*]] = ttkernel.my_logical_y_
// CHECK-NEXT: %[[DST_NODE_ROW:.*]] = arith.muli %[[DST_NODE_Y]], %{{.*}} : index
// CHECK-NEXT: %[[DST_NODE_INDEX:.*]] = arith.addi %[[DST_NODE_ROW]], %[[DST_NODE_X]] : index
// CHECK: %[[DST_DEVICE_I32:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[DST_DEVICE:.*]] = arith.index_cast %[[DST_DEVICE_I32]] : i32 to index
// CHECK-NEXT: %[[DST_NEXT_DEVICE:.*]] = arith.addi %[[DST_DEVICE]], %{{.*}} : index
// CHECK-NEXT: %[[DST_LOWER:.*]] = ttkernel.experimental.constant_table_lookup %[[DST_DEVICE]], [0, 1, 2] : index
// CHECK-NEXT: %[[DST_UPPER:.*]] = ttkernel.experimental.constant_table_lookup %[[DST_NEXT_DEVICE]], [0, 1, 2] : index
// CHECK: scf.for %[[DST_EDGE_POSITION:.*]] = %[[DST_LOWER]] to %[[DST_UPPER]] step
// CHECK-NEXT: %[[DST_EDGE_BLOCK:.*]] = ttkernel.experimental.constant_table_lookup %[[DST_EDGE_POSITION]], [1, 0] : index
// CHECK-NEXT: %[[DST_EDGE_OFFSET:.*]] = arith.muli %[[DST_EDGE_BLOCK]], %{{.*}} : index
// CHECK-NEXT: %[[DST_RECORD:.*]] = arith.addi %[[DST_EDGE_OFFSET]], %[[DST_NODE_INDEX]] : index
// CHECK-NOT: arith.cmpi
// CHECK: ttkernel.routing_plane.atomic_inc

#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#records = #ttl.pipenet_records<net 0 name "grid_major" pipes [
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [1]>>>>,
  #ttl.pipe_record<
      srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0,
      dstEndX = 1, dstEndY = 0,
      deviceTransfer = <
        domain = #domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [1]>>>>,
  #ttl.pipe_record<
      srcX = 0, srcY = 1, dstStartX = 0, dstStartY = 1,
      dstEndX = 0, dstEndY = 1,
      deviceTransfer = <
        domain = #domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [1]>>>>,
  #ttl.pipe_record<
      srcX = 1, srcY = 1, dstStartX = 1, dstStartY = 1,
      dstEndX = 1, dstEndY = 1,
      deviceTransfer = <
        domain = #domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [1]>>>>,
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #domain,
        edge = <source = <coordinates = [1]>,
                destination = <coordinates = [0]>>>>,
  #ttl.pipe_record<
      srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0,
      dstEndX = 1, dstEndY = 0,
      deviceTransfer = <
        domain = #domain,
        edge = <source = <coordinates = [1]>,
                destination = <coordinates = [0]>>>>,
  #ttl.pipe_record<
      srcX = 0, srcY = 1, dstStartX = 0, dstStartY = 1,
      dstEndX = 0, dstEndY = 1,
      deviceTransfer = <
        domain = #domain,
        edge = <source = <coordinates = [1]>,
                destination = <coordinates = [0]>>>>,
  #ttl.pipe_record<
      srcX = 1, srcY = 1, dstStartX = 1, dstStartY = 1,
      dstEndX = 1, dstEndY = 1,
      deviceTransfer = <
        domain = #domain,
        edge = <source = <coordinates = [1]>,
                destination = <coordinates = [0]>>>>
]>

module attributes {ttl.launch_grid = array<i64: 2, 2>} {
  func.func @sender() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_src attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %send = ttl.copy %source, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.yield
    }
    func.return
  }

  func.func @receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_dst attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %reserved = ttl.cb_reserve %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %receive = ttl.copy %pipe, %reserved
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
      ttl.cb_push %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      ttl.yield
    }
    func.return
  }
}
