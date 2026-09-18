// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verify selected fabric records preserve record-aligned route and
// resource tables when one source device communicates with two destinations.

// Fabric records compute receiver DFB addresses and do not allocate a
// receiver-published address table.
// CHECK-LABEL: module attributes
// CHECK-NOT: ttl.pipe_sram_scratch_bytes

// The sender uses route slots 0 and 1 for records 0 and 1. Its readiness
// resources use distinct compiler-managed common arguments and counter slots.
// CHECK-LABEL: func.func @sender()
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK: %[[FABRIC_BASE_I32:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[FABRIC_BASE:.*]] = arith.index_cast %[[FABRIC_BASE_I32]] : i32 to index
// CHECK: scf.for %[[RECORD:.*]] =
// CHECK: %[[ROUTE:.*]] = ttkernel.experimental.constant_table_lookup %[[RECORD]], [0, 1] : index
// CHECK-NEXT: %[[READY_ARG_INDEX:.*]] = ttkernel.experimental.constant_table_lookup %[[RECORD]], [2, 3] : index
// CHECK-NEXT: %[[READY_ADDRESS:.*]] = ttkernel.get_common_arg_val(%[[READY_ARG_INDEX]]) : (index) -> i32
// CHECK-NEXT: %[[READY_COUNTER:.*]] = ttkernel.experimental.constant_table_lookup %[[RECORD]], [0, 1] : index
// CHECK: %[[DEST_DEVICE_RELATIVE_INDEX:.*]] = arith.addi %[[ROUTE]], {{.*}} : index
// CHECK: %[[DEST_MESH_RELATIVE_INDEX:.*]] = arith.addi %[[ROUTE]], {{.*}} : index
// CHECK: %[[DEST_HOPS_RELATIVE_INDEX:.*]] = arith.addi %[[ROUTE]], {{.*}} : index
// CHECK-NEXT: %[[DEST_DEVICE_ARG_INDEX:.*]] = arith.addi %[[FABRIC_BASE]], %[[DEST_DEVICE_RELATIVE_INDEX]] : index
// CHECK-NEXT: %[[DEST_MESH_ARG_INDEX:.*]] = arith.addi %[[FABRIC_BASE]], %[[DEST_MESH_RELATIVE_INDEX]] : index
// CHECK-NEXT: %[[DEST_HOPS_ARG_INDEX:.*]] = arith.addi %[[FABRIC_BASE]], %[[DEST_HOPS_RELATIVE_INDEX]] : index
// CHECK: %[[DEST_DEVICE:.*]] = ttkernel.get_arg_val(%[[DEST_DEVICE_ARG_INDEX]]) : (index) -> i32
// CHECK: %[[DEST_MESH:.*]] = ttkernel.get_arg_val(%[[DEST_MESH_ARG_INDEX]]) : (index) -> i32
// CHECK: %[[DEST_HOPS:.*]] = ttkernel.get_arg_val(%[[DEST_HOPS_ARG_INDEX]]) : (index) -> i32
// CHECK: %[[CONNECTION_RELATIVE_INDEX:.*]] = arith.addi %[[ROUTE]], {{.*}} : index
// CHECK-NEXT: %[[CONNECTION_ARG_INDEX:.*]] = arith.addi %[[FABRIC_BASE]], %[[CONNECTION_RELATIVE_INDEX]] : index
// CHECK-NEXT: %[[CONNECTION:.*]] = ttkernel.get_arg_val(%[[CONNECTION_ARG_INDEX]]) : (index) -> i32
// CHECK: scf.if
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc({{.*}}, %[[CONNECTION]], %[[DEST_DEVICE]], %[[DEST_MESH]], %[[DEST_HOPS]],

// Each receiver record resolves its own logical device and reverse-route
// destination while both records use reverse route slot zero.
// CHECK-LABEL: func.func @receiver()
// CHECK: %[[REVERSE_FABRIC_BASE_I32:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[REVERSE_FABRIC_BASE:.*]] = arith.index_cast %[[REVERSE_FABRIC_BASE_I32]] : i32 to index
// CHECK: scf.for %[[RECORD:.*]] =
// CHECK: %[[DEST_DEVICE:.*]] = ttkernel.experimental.constant_table_lookup %[[RECORD]], [1, 2] : index
// CHECK: arith.cmpi eq, {{.*}}, %[[DEST_DEVICE]] : index
// CHECK: %[[COMPLETION_ARG_INDEX:.*]] = ttkernel.experimental.constant_table_lookup %[[RECORD]], [1, 2] : index
// CHECK-NEXT: %[[COMPLETION_ADDRESS:.*]] = ttkernel.get_common_arg_val(%[[COMPLETION_ARG_INDEX]]) : (index) -> i32
// CHECK-NEXT: %[[REVERSE_ROUTE:.*]] = ttkernel.experimental.constant_table_lookup %[[RECORD]], [0, 0] : index
// CHECK: %[[REVERSE_DEVICE_RELATIVE_INDEX:.*]] = arith.addi %[[REVERSE_ROUTE]], {{.*}} : index
// CHECK: %[[REVERSE_MESH_RELATIVE_INDEX:.*]] = arith.addi %[[REVERSE_ROUTE]], {{.*}} : index
// CHECK: %[[REVERSE_HOPS_RELATIVE_INDEX:.*]] = arith.addi %[[REVERSE_ROUTE]], {{.*}} : index
// CHECK-NEXT: %[[REVERSE_DEVICE_ARG_INDEX:.*]] = arith.addi %[[REVERSE_FABRIC_BASE]], %[[REVERSE_DEVICE_RELATIVE_INDEX]] : index
// CHECK-NEXT: %[[REVERSE_MESH_ARG_INDEX:.*]] = arith.addi %[[REVERSE_FABRIC_BASE]], %[[REVERSE_MESH_RELATIVE_INDEX]] : index
// CHECK-NEXT: %[[REVERSE_HOPS_ARG_INDEX:.*]] = arith.addi %[[REVERSE_FABRIC_BASE]], %[[REVERSE_HOPS_RELATIVE_INDEX]] : index
// CHECK: %[[REVERSE_DEVICE:.*]] = ttkernel.get_arg_val(%[[REVERSE_DEVICE_ARG_INDEX]]) : (index) -> i32
// CHECK: %[[REVERSE_MESH:.*]] = ttkernel.get_arg_val(%[[REVERSE_MESH_ARG_INDEX]]) : (index) -> i32
// CHECK: %[[REVERSE_HOPS:.*]] = ttkernel.get_arg_val(%[[REVERSE_HOPS_ARG_INDEX]]) : (index) -> i32
// CHECK: %[[REVERSE_CONNECTION_RELATIVE_INDEX:.*]] = arith.addi %[[REVERSE_ROUTE]], {{.*}} : index
// CHECK-NEXT: %[[REVERSE_CONNECTION_ARG_INDEX:.*]] = arith.addi %[[REVERSE_FABRIC_BASE]], %[[REVERSE_CONNECTION_RELATIVE_INDEX]] : index
// CHECK-NEXT: %[[REVERSE_CONNECTION:.*]] = ttkernel.get_arg_val(%[[REVERSE_CONNECTION_ARG_INDEX]]) : (index) -> i32
// CHECK: %[[COMPLETION_COUNTER:.*]] = ttkernel.experimental.constant_table_lookup %[[RECORD]], [0, 0] : index
// CHECK-NEXT: %[[COMPLETION_STATE_ARG_INDEX:.*]] = ttkernel.experimental.constant_table_lookup %[[RECORD]], [0, 0] : index
// CHECK-NEXT: ttkernel.get_common_arg_val(%[[COMPLETION_STATE_ARG_INDEX]]) : (index) -> i32
// CHECK: scf.if
// CHECK: ttkernel.routing_plane.atomic_inc({{.*}}, %[[REVERSE_CONNECTION]], %[[REVERSE_DEVICE]], %[[REVERSE_MESH]], %[[REVERSE_HOPS]],
// CHECK: memref.load {{.*}}[%[[COMPLETION_COUNTER]]]

#domain = #ttl.device_domain<components = <name = "device", extent = [3]>>
#records = #ttl.pipenet_records<net 0 name "selected_tables" pipes [
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [1]>>>>,
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [2]>>>>
]>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @sender() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_src attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %send = ttl.copy %src, %pipe
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
    %dst = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_dst attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %reserved = ttl.cb_reserve %dst
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post = ttl.copy %pipe, %reserved
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %post : !ttl.receive_request
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      ttl.yield
    }
    func.return
  }
}
