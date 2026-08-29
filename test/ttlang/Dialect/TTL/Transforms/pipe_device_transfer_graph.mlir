// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verify that PipeGraph preserves distinct logical-device transfers
// that share one node-level PipeKey.

// Receiver declarations intentionally reverse the send order. Each edge must
// retain its route, predicate, payload send, completion wait, and receiver DFB.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.pipe_global_semaphore_count = 2 : i64
// CHECK-SAME: ttl.pipe_sync_semaphore_count = 0 : i64
// CHECK-LABEL: func.func @senders
// CHECK-SAME: ttl.fabric_routes = [
// CHECK-SAME: local = #ttl.device_ref<coordinates = [0]>
// CHECK-SAME: remote = #ttl.device_ref<coordinates = [1]>
// CHECK-SAME: route_index = 0 : i64
// CHECK-SAME: source_nodes = [array<i64: 0, 0>]
// CHECK-SAME: local = #ttl.device_ref<coordinates = [2]>
// CHECK-SAME: remote = #ttl.device_ref<coordinates = [3]>
// CHECK-SAME: route_index = 0 : i64
// CHECK-SAME: source_nodes = [array<i64: 0, 0>]
// CHECK-SAME: }]
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1, 2>
// CHECK: %[[SOURCE_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[DEVICE_0:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[IS_DEVICE_0:.*]] = arith.cmpi eq, %[[DEVICE_0]], %{{.*}} : i32
// CHECK: %[[FABRIC_BASE_0:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[FABRIC_BASE_INDEX_0:.*]] = arith.index_cast %[[FABRIC_BASE_0]] : i32 to index
// CHECK-NEXT: %[[CONNECTIONS_0:.*]] = ttkernel.get_arg_val(%[[FABRIC_BASE_INDEX_0]])
// CHECK-NEXT: %[[RUNTIME_ARG_BASE_0:.*]] = arith.addi %[[FABRIC_BASE_INDEX_0]],
// CHECK: scf.if %[[IS_DEVICE_0]] {
// CHECK-NEXT: %[[CONNECTION_MANAGER_0:.*]] = ttkernel.routing_plane.create_connection_manager
// CHECK-NEXT: %[[CONNECTION_COUNT_0:.*]] = ttkernel.routing_plane.open_connections %[[CONNECTION_MANAGER_0]], %[[CONNECTIONS_0]] runtime_arg_base = %[[RUNTIME_ARG_BASE_0]]
// CHECK: %[[PAYLOAD_0:.*]] = ttkernel.get_write_ptr
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc(%[[CONNECTION_MANAGER_0]], %[[CONNECTION_COUNT_0]], {{.*}}, {{.*}}, {{.*}}, %[[PAYLOAD_0]],
// CHECK-NEXT: ttkernel.routing_plane.close_connections(%[[CONNECTION_MANAGER_0]], %[[CONNECTIONS_0]])
// CHECK-NEXT: }
// CHECK: %[[DEVICE_2:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[IS_DEVICE_2:.*]] = arith.cmpi eq, %[[DEVICE_2]], %{{.*}} : i32
// CHECK: %[[FABRIC_BASE_2:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[FABRIC_BASE_INDEX_2:.*]] = arith.index_cast %[[FABRIC_BASE_2]] : i32 to index
// CHECK-NEXT: %[[CONNECTIONS_2:.*]] = ttkernel.get_arg_val(%[[FABRIC_BASE_INDEX_2]])
// CHECK-NEXT: %[[RUNTIME_ARG_BASE_2:.*]] = arith.addi %[[FABRIC_BASE_INDEX_2]],
// CHECK: scf.if %[[IS_DEVICE_2]] {
// CHECK-NEXT: %[[CONNECTION_MANAGER_2:.*]] = ttkernel.routing_plane.create_connection_manager
// CHECK-NEXT: %[[CONNECTION_COUNT_2:.*]] = ttkernel.routing_plane.open_connections %[[CONNECTION_MANAGER_2]], %[[CONNECTIONS_2]] runtime_arg_base = %[[RUNTIME_ARG_BASE_2]]
// CHECK: %[[PAYLOAD_1:.*]] = ttkernel.get_write_ptr
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc(%[[CONNECTION_MANAGER_2]], %[[CONNECTION_COUNT_2]], {{.*}}, {{.*}}, {{.*}}, %[[PAYLOAD_1]],
// CHECK-NEXT: ttkernel.routing_plane.close_connections(%[[CONNECTION_MANAGER_2]], %[[CONNECTIONS_2]])
// CHECK-NEXT: }
// CHECK-NOT: ttkernel.routing_plane.fused_write_atomic_inc
// CHECK-LABEL: func.func @receivers
// CHECK: %[[RECEIVER_DFB_0:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-NEXT: %[[RECEIVER_DFB_1:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: %[[DEVICE_3:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[IS_DEVICE_3:.*]] = arith.cmpi eq, %[[DEVICE_3]], %{{.*}} : i32
// CHECK: %[[RECEIVER_FABRIC_BASE_3:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[RECEIVER_FABRIC_BASE_INDEX_3:.*]] = arith.index_cast %[[RECEIVER_FABRIC_BASE_3]] : i32 to index
// CHECK-NEXT: %[[RECEIVER_CONNECTIONS_3:.*]] = ttkernel.get_arg_val(%[[RECEIVER_FABRIC_BASE_INDEX_3]])
// CHECK-NEXT: %[[RECEIVER_RUNTIME_ARG_BASE_3:.*]] = arith.addi %[[RECEIVER_FABRIC_BASE_INDEX_3]],
// CHECK: scf.if %[[IS_DEVICE_3]] {
// CHECK-NEXT: ttkernel.cb_reserve_back(%[[RECEIVER_DFB_1]],
// CHECK-NEXT: %[[RECEIVER_CONNECTION_MANAGER_3:.*]] = ttkernel.routing_plane.create_connection_manager
// CHECK-NEXT: %[[RECEIVER_CONNECTION_COUNT_3:.*]] = ttkernel.routing_plane.open_connections %[[RECEIVER_CONNECTION_MANAGER_3]], %[[RECEIVER_CONNECTIONS_3]] runtime_arg_base = %[[RECEIVER_RUNTIME_ARG_BASE_3]]
// CHECK-NEXT: ttkernel.routing_plane.atomic_inc(%[[RECEIVER_CONNECTION_MANAGER_3]], %[[RECEIVER_CONNECTION_COUNT_3]],
// CHECK: ttkernel.routing_plane.close_connections(%[[RECEIVER_CONNECTION_MANAGER_3]], %[[RECEIVER_CONNECTIONS_3]])
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK-NEXT: ttkernel.cb_push_back(%[[RECEIVER_DFB_1]],
// CHECK-NEXT: }
// CHECK: %[[DEVICE_1:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[IS_DEVICE_1:.*]] = arith.cmpi eq, %[[DEVICE_1]], %{{.*}} : i32
// CHECK: %[[RECEIVER_FABRIC_BASE_1:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[RECEIVER_FABRIC_BASE_INDEX_1:.*]] = arith.index_cast %[[RECEIVER_FABRIC_BASE_1]] : i32 to index
// CHECK-NEXT: %[[RECEIVER_CONNECTIONS_1:.*]] = ttkernel.get_arg_val(%[[RECEIVER_FABRIC_BASE_INDEX_1]])
// CHECK-NEXT: %[[RECEIVER_RUNTIME_ARG_BASE_1:.*]] = arith.addi %[[RECEIVER_FABRIC_BASE_INDEX_1]],
// CHECK: scf.if %[[IS_DEVICE_1]] {
// CHECK-NEXT: ttkernel.cb_reserve_back(%[[RECEIVER_DFB_0]],
// CHECK-NEXT: %[[RECEIVER_CONNECTION_MANAGER_1:.*]] = ttkernel.routing_plane.create_connection_manager
// CHECK-NEXT: %[[RECEIVER_CONNECTION_COUNT_1:.*]] = ttkernel.routing_plane.open_connections %[[RECEIVER_CONNECTION_MANAGER_1]], %[[RECEIVER_CONNECTIONS_1]] runtime_arg_base = %[[RECEIVER_RUNTIME_ARG_BASE_1]]
// CHECK-NEXT: ttkernel.routing_plane.atomic_inc(%[[RECEIVER_CONNECTION_MANAGER_1]], %[[RECEIVER_CONNECTION_COUNT_1]],
// CHECK: ttkernel.routing_plane.close_connections(%[[RECEIVER_CONNECTION_MANAGER_1]], %[[RECEIVER_CONNECTIONS_1]])
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK-NEXT: ttkernel.cb_push_back(%[[RECEIVER_DFB_0]],
// CHECK-NEXT: }
// CHECK-NOT: ttkernel.experimental.semaphore_wait_min
// CHECK-NEXT: return

#domain = #ttl.device_domain<components = <name = "device", extent = [4]>>
#transfer_0 = #ttl.device_transfer<
    domain = #domain,
    edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>
#transfer_1 = #ttl.device_transfer<
    domain = #domain,
    edge = <source = <coordinates = [2]>, destination = <coordinates = [3]>>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @senders() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %src = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe_0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 9 {
        deviceTransfer = #transfer_0}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 9>
    %pipe_1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 9 {
        deviceTransfer = #transfer_1}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 9>
    %is_source_0 = ttl.is_device <coordinates = [0]> in #domain : i1
    scf.if %is_source_0 {
      %send_0 = ttl.copy %src, %pipe_0
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 9>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send_0 : !ttl.transfer_handle<write>
    }
    %is_source_1 = ttl.is_device <coordinates = [2]> in #domain : i1
    scf.if %is_source_1 {
      %send_1 = ttl.copy %src, %pipe_1
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 9>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send_1 : !ttl.transfer_handle<write>
    }
    func.return
  }

  func.func @receivers()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dst_0 = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst_1 = ttl.bind_cb {cb_index = 2, block_count = 1} {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe_0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 9 {
        deviceTransfer = #transfer_0}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 9>
    %pipe_1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 9 {
        deviceTransfer = #transfer_1}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 9>
    %is_destination_1 = ttl.is_device <coordinates = [3]> in #domain : i1
    scf.if %is_destination_1 {
      %reserved_1 = ttl.cb_reserve %dst_1
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post_1 = ttl.copy %pipe_1, %reserved_1
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 9>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %post_1 : !ttl.receive_request
      ttl.cb_push %dst_1 : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    %is_destination_0 = ttl.is_device <coordinates = [1]> in #domain : i1
    scf.if %is_destination_0 {
      %reserved_0 = ttl.cb_reserve %dst_0
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post_0 = ttl.copy %pipe_0, %reserved_0
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 9>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %post_0 : !ttl.receive_request
      ttl.cb_push %dst_0 : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    func.return
  }
}
