// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verify selected fabric records preserve record-aligned route and
// resource tables when one source device communicates with two destinations.

// Fabric records compute receiver DFB addresses before dispatch. One-shot
// records therefore require only the completion semaphore.
// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.pipe_global_semaphore_count = 1 : i64
// CHECK-SAME: ttl.pipe_sync_semaphore_count = 0 : i64
// CHECK-NOT: ttl.pipe_sram_scratch_bytes

// Each record selects a disjoint pair of preconfigured route headers and the
// corresponding connection range. It emits the transfer without a
// sender-readiness wait because destination storage is stable for the
// operation's single execution.
// CHECK-LABEL: func.func @sender()
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK: %[[FABRIC_BASE_I32:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[FABRIC_BASE:.*]] = arith.index_cast %[[FABRIC_BASE_I32]] : i32 to index
// CHECK: scf.for %[[RECORD:.*]] =
// CHECK: %[[ROUTE:.*]] = ttkernel.experimental.constant_table_lookup %[[RECORD]], [0, 1] : index
// CHECK-NOT: ttkernel.experimental.semaphore_wait
// CHECK: %[[ROUTE_HEADER_BASE_INDEX:.*]] = arith.muli %[[ROUTE]], {{.*}} : index
// CHECK-NEXT: %[[ROUTE_HEADER_BASE:.*]] = arith.index_cast %[[ROUTE_HEADER_BASE_INDEX]] : index to i32
// CHECK: %[[CONNECTION_RELATIVE_INDEX:.*]] = arith.addi %[[ROUTE]], {{.*}} : index
// CHECK-NEXT: %[[CONNECTION_ARG_INDEX:.*]] = arith.addi %[[FABRIC_BASE]], %[[CONNECTION_RELATIVE_INDEX]] : index
// CHECK-NEXT: %[[CONNECTION:.*]] = ttkernel.get_arg_val(%[[CONNECTION_ARG_INDEX]]) : (index) -> i32
// CHECK-NEXT: %[[CONNECTION_COUNT_RELATIVE_INDEX:.*]] = arith.addi %[[ROUTE]], {{.*}} : index
// CHECK-NEXT: %[[CONNECTION_COUNT_ARG_INDEX:.*]] = arith.addi %[[FABRIC_BASE]], %[[CONNECTION_COUNT_RELATIVE_INDEX]] : index
// CHECK-NEXT: %[[CONNECTION_COUNT:.*]] = ttkernel.get_arg_val(%[[CONNECTION_COUNT_ARG_INDEX]]) : (index) -> i32
// CHECK: scf.if
// CHECK-NOT: ttkernel.routing_plane.atomic_inc
// CHECK: ttkernel.routing_plane.striped_fused_write_atomic_inc({{.*}}, %[[ROUTE_HEADER_BASE]], %[[CONNECTION]], %[[CONNECTION_COUNT]],{{.*}}) {posted = true}

// Each receiver record resolves its logical device, then waits on its shared
// completion counter. No reverse-route readiness atomic is emitted.
// CHECK-LABEL: func.func @receiver()
// CHECK: scf.for %[[RECORD:.*]] =
// CHECK: %[[DEST_DEVICE:.*]] = ttkernel.experimental.constant_table_lookup %[[RECORD]], [1, 2] : index
// CHECK: arith.cmpi eq, {{.*}}, %[[DEST_DEVICE]] : index
// CHECK: %[[COMPLETION_COUNTER:.*]] = ttkernel.experimental.constant_table_lookup %[[RECORD]], [0, 0] : index
// CHECK-NEXT: %[[COMPLETION_ADDRESS:.*]] = ttkernel.get_common_arg_val(%[[COMPLETION_COUNTER]]) : (index) -> i32
// CHECK-NEXT: %[[COMPLETION_POINTER:.*]] = ttkernel.reinterpret_cast(%[[COMPLETION_ADDRESS]])
// CHECK: scf.if
// CHECK-NOT: ttkernel.routing_plane.atomic_inc
// CHECK: ttkernel.experimental.semaphore_wait_min(%[[COMPLETION_POINTER]], {{.*}})

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
          -> !ttl.transfer_handle
      ttl.wait %post : !ttl.transfer_handle
      ttl.cb_push %dst : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      ttl.yield
    }
    func.return
  }
}
