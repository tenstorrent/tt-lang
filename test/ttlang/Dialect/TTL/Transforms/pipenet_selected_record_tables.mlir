// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s
// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s --check-prefix=MIXED

// Summary: Verify selected fabric records preserve record-aligned route and
// resource tables, including per-record receiver-readiness decisions.

// Fabric records compute receiver DFB addresses before dispatch. One-shot
// records therefore require only completion synchronization.
// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.pipe_global_semaphore_count = 1 : i64
// CHECK-SAME: ttl.pipe_sync_semaphore_count = 0 : i64
// CHECK-NOT: ttl.pipe_sram_scratch_bytes

// The sender uses route slots 0 and 1 without waiting for receiver readiness.
// CHECK-LABEL: func.func @sender()
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK: %[[FABRIC_BASE_I32:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[FABRIC_BASE:.*]] = arith.index_cast %[[FABRIC_BASE_I32]] : i32 to index
// CHECK: scf.for %[[RECORD:.*]] =
// CHECK: scf.if
// CHECK-NOT: ttkernel.experimental.semaphore_wait
// CHECK: %[[ROUTE:.*]] = ttkernel.experimental.constant_table_lookup %[[RECORD]], [0, 1] : index
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
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc({{.*}}, %[[CONNECTION]], %[[DEST_DEVICE]], %[[DEST_MESH]], %[[DEST_HOPS]],

// Each receiver record resolves its logical device and waits for completion.
// No reverse-route readiness operation is emitted.
// CHECK-LABEL: func.func @receiver()
// CHECK: scf.for %[[RECORD:.*]] =
// CHECK: %[[DEST_DEVICE:.*]] = ttkernel.experimental.constant_table_lookup %[[RECORD]], [1, 2] : index
// CHECK: arith.cmpi eq, {{.*}}, %[[DEST_DEVICE]] : index
// CHECK: scf.if
// CHECK-NOT: ttkernel.routing_plane.atomic_inc
// CHECK: %[[COMPLETION_COUNTER:.*]] = ttkernel.experimental.constant_table_lookup %[[RECORD]], [0, 0] : index
// CHECK-NEXT: %[[COMPLETION_VALUE:.*]] = memref.load {{.*}}[%[[COMPLETION_COUNTER]]]
// CHECK-NEXT: %[[NEXT_COMPLETION:.*]] = arith.addi %[[COMPLETION_VALUE]], {{.*}} : i32
// CHECK-NEXT: memref.store %[[NEXT_COMPLETION]], {{.*}}[%[[COMPLETION_COUNTER]]]
// CHECK-NEXT: %[[COMPLETION_STATE_ARG_INDEX:.*]] = ttkernel.experimental.constant_table_lookup %[[RECORD]], [0, 0] : index
// CHECK-NEXT: ttkernel.get_common_arg_val(%[[COMPLETION_STATE_ARG_INDEX]]) : (index) -> i32

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

// -----

// One selected receiver operation can require readiness for only the record
// whose DFB block is reused by a later transfer.

// The sender waits only when the selected record targets device 1.
// MIXED-LABEL: func.func @mixed_sender()
// MIXED: scf.for %[[SENDER_RECORD:.*]] =
// MIXED: %[[SENDER_USES_READINESS:.*]] = ttkernel.experimental.constant_table_lookup %[[SENDER_RECORD]], [1, 0] : index
// MIXED-NEXT: %[[SENDER_REQUIRES_WAIT:.*]] = arith.cmpi ne, %[[SENDER_USES_READINESS]], {{.*}} : index
// MIXED-NEXT: scf.if %[[SENDER_REQUIRES_WAIT]] {
// MIXED: ttkernel.experimental.semaphore_wait_min
// MIXED: ttkernel.routing_plane.fused_write_atomic_inc

// The receiver owns a reverse connection only for the record that sends a
// readiness message. The other route remains present for record-table lookup
// but has no source node for runtime binding.
// MIXED-LABEL: func.func @mixed_receiver()
// MIXED-SAME: ttl.fabric_routes = [{local = #ttl.device_ref<coordinates = [1]>
// MIXED-SAME: source_nodes = [array<i64: 0, 0>]}
// MIXED-SAME: {local = #ttl.device_ref<coordinates = [2]>
// MIXED-SAME: source_nodes = []}]
// MIXED: scf.for %[[RECEIVER_RECORD:.*]] =
// MIXED: %[[RECEIVER_USES_READINESS:.*]] = ttkernel.experimental.constant_table_lookup %[[RECEIVER_RECORD]], [1, 0] : index
// MIXED-NEXT: %[[RECEIVER_REQUIRES_SIGNAL:.*]] = arith.cmpi ne, %[[RECEIVER_USES_READINESS]], {{.*}} : index
// MIXED-NEXT: scf.if %[[RECEIVER_REQUIRES_SIGNAL]] {
// MIXED: ttkernel.routing_plane.atomic_inc

#mixed_domain = #ttl.device_domain<components = <name = "device", extent = [3]>>
#mixed_records = #ttl.pipenet_records<net 0 name "mixed_readiness" pipes [
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #mixed_domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [1]>>>>,
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #mixed_domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [2]>>>>
]>
#extra_transfer = #ttl.device_transfer<
    domain = #mixed_domain,
    edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @mixed_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_src attributes {records = #mixed_records} {
    ^bb0(%selected_pipe: !ttl.selected_pipe_src):
      %selected_send = ttl.copy %source, %selected_pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %selected_send : !ttl.transfer_handle<write>
      ttl.yield
    }
    %extra_pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1 {
        deviceTransfer = #extra_transfer}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
    %is_source = ttl.is_device <coordinates = [0]> in #mixed_domain : i1
    scf.if %is_source {
      %extra_send = ttl.copy %source, %extra_pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>)
          -> !ttl.transfer_handle<write>
      ttl.wait %extra_send : !ttl.transfer_handle<write>
    }
    func.return
  }

  func.func @mixed_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.pipenet_foreach_dst attributes {records = #mixed_records} {
    ^bb0(%selected_pipe: !ttl.selected_pipe_dst):
      %selected_reservation = ttl.cb_reserve %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %selected_post = ttl.copy %selected_pipe, %selected_reservation
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %selected_post : !ttl.receive_request
      ttl.cb_push %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      %selected_payload = ttl.cb_wait %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      ttl.yield
    }
    %extra_pipe = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 1 {
        deviceTransfer = #extra_transfer}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>
    %is_destination = ttl.is_device <coordinates = [1]>
        in #mixed_domain : i1
    scf.if %is_destination {
      %extra_reservation = ttl.cb_reserve %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %extra_post = ttl.copy %extra_pipe, %extra_reservation
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 1>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %extra_post : !ttl.receive_request
      ttl.cb_push %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
      %extra_payload = ttl.cb_wait %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.cb_pop %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    func.return
  }
}
