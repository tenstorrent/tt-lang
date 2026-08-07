// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verify that PipeGraph preserves distinct logical-device transfers
// that share one node-level PipeKey.

// Receiver declarations intentionally reverse the send order. Each edge must
// retain its route, predicate, payload send, completion wait, and receiver DFB.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.pipe_global_semaphore_count = 1 : i64
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
// CHECK: %[[CONNECTION_MANAGER:.*]] = ttkernel.routing_plane.create_connection_manager
// CHECK-NEXT: %[[CONNECTION_COUNT:.*]] = ttkernel.routing_plane.open_connections %[[CONNECTION_MANAGER]]
// CHECK: %[[DEVICE_0:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[IS_DEVICE_0:.*]] = arith.cmpi eq, %[[DEVICE_0]], %{{.*}} : i32
// CHECK: scf.if %[[IS_DEVICE_0]] {
// CHECK: %[[PAYLOAD_0:.*]] = ttkernel.get_write_ptr
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc(%[[CONNECTION_MANAGER]], %[[CONNECTION_COUNT]], {{.*}}, {{.*}}, {{.*}}, %[[PAYLOAD_0]],
// CHECK-NEXT: }
// CHECK: %[[DEVICE_2:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[IS_DEVICE_2:.*]] = arith.cmpi eq, %[[DEVICE_2]], %{{.*}} : i32
// CHECK: scf.if %[[IS_DEVICE_2]] {
// CHECK: %[[PAYLOAD_1:.*]] = ttkernel.get_write_ptr
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc(%[[CONNECTION_MANAGER]], %[[CONNECTION_COUNT]], {{.*}}, {{.*}}, {{.*}}, %[[PAYLOAD_1]],
// CHECK-NEXT: }
// CHECK-NOT: ttkernel.routing_plane.fused_write_atomic_inc
// CHECK: ttkernel.routing_plane.close_connections(%[[CONNECTION_MANAGER]],
// CHECK-LABEL: func.func @receivers
// CHECK: %[[RECEIVER_DFB_0:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-NEXT: %[[RECEIVER_DFB_1:.*]] = ttkernel.get_compile_time_arg_val(2)
// CHECK: %[[DEVICE_3:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[IS_DEVICE_3:.*]] = arith.cmpi eq, %[[DEVICE_3]], %{{.*}} : i32
// CHECK: scf.if %[[IS_DEVICE_3]] {
// CHECK-NEXT: ttkernel.cb_reserve_back(%[[RECEIVER_DFB_1]],
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK-NEXT: ttkernel.cb_push_back(%[[RECEIVER_DFB_1]],
// CHECK-NEXT: }
// CHECK: %[[DEVICE_1:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[IS_DEVICE_1:.*]] = arith.cmpi eq, %[[DEVICE_1]], %{{.*}} : i32
// CHECK: scf.if %[[IS_DEVICE_1]] {
// CHECK-NEXT: ttkernel.cb_reserve_back(%[[RECEIVER_DFB_0]],
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
    %src = ttl.bind_cb {cb_index = 0, block_count = 1}
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
    %dst_0 = ttl.bind_cb {cb_index = 1, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %dst_1 = ttl.bind_cb {cb_index = 2, block_count = 1}
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
          -> !ttl.transfer_handle
      ttl.wait %post_1 : !ttl.transfer_handle
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
          -> !ttl.transfer_handle
      ttl.wait %post_0 : !ttl.transfer_handle
      ttl.cb_push %dst_0 : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    func.return
  }
}
