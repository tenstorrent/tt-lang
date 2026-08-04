// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s --implicit-check-not=ttkernel.noc_inline_dw_write

// Summary: Verify computed receiver addresses for multiple logical-device
// transfers that reserve successive blocks in one destination DFB.
// Computed-address fabric lowering must not publish receiver DFB addresses
// with NoC inline writes.

// Two source devices send to one destination device. The destination executes
// both reservations in program order, so the second transfer uses DFB block 1.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.pipe_global_semaphore_count = 2 : i64
// CHECK-SAME: ttl.pipe_sync_semaphore_count = 0 : i64
// CHECK-LABEL: func.func @senders
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[BLOCK_BYTES:.*]] = arith.constant 4096 : i32
// CHECK-DAG: %[[BASE_ARG:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[COMPLETION_0_ARG:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[COMPLETION_1_ARG:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[DEVICE_ARG:.*]] = arith.constant 3 : index
// CHECK: %[[CONNECTION_MANAGER:.*]] = ttkernel.routing_plane.create_connection_manager
// CHECK-NEXT: %[[CONNECTION_COUNT:.*]] = ttkernel.routing_plane.open_connections %[[CONNECTION_MANAGER]],
// CHECK: %[[SOURCE_DFB:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK: %[[DEVICE_0:.*]] = ttkernel.get_common_arg_val(%[[DEVICE_ARG]])
// CHECK-NEXT: %[[IS_DEVICE_0:.*]] = arith.cmpi eq, %[[DEVICE_0]], %[[ZERO]]
// CHECK-NEXT: scf.if %[[IS_DEVICE_0]] {
// CHECK-NEXT: %[[PAYLOAD_0:.*]] = ttkernel.get_write_ptr(%[[SOURCE_DFB]])
// CHECK-NEXT: %[[BASE_0:.*]] = ttkernel.get_common_arg_val(%[[BASE_ARG]])
// CHECK-NEXT: %[[COMPLETION_0:.*]] = ttkernel.get_common_arg_val(%[[COMPLETION_0_ARG]])
// CHECK: %[[DATA_NOC_0:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[BASE_0]],
// CHECK: %[[COMPLETION_NOC_0:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[COMPLETION_0]],
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc(%[[CONNECTION_MANAGER]], %[[CONNECTION_COUNT]], {{.*}}, {{.*}}, {{.*}}, %[[PAYLOAD_0]], %[[BLOCK_BYTES]], %[[DATA_NOC_0]], %[[COMPLETION_NOC_0]], %[[ONE]])
// CHECK-NEXT: }
// CHECK-NEXT: %[[DEVICE_1:.*]] = ttkernel.get_common_arg_val(%[[DEVICE_ARG]])
// CHECK-NEXT: %[[IS_DEVICE_1:.*]] = arith.cmpi eq, %[[DEVICE_1]], %[[ONE]]
// CHECK-NEXT: scf.if %[[IS_DEVICE_1]] {
// CHECK-NEXT: %[[PAYLOAD_1:.*]] = ttkernel.get_write_ptr(%[[SOURCE_DFB]])
// CHECK-NEXT: %[[BASE_1:.*]] = ttkernel.get_common_arg_val(%[[BASE_ARG]])
// CHECK-NEXT: %[[BLOCK_1:.*]] = arith.addi %[[BASE_1]], %[[BLOCK_BYTES]] : i32
// CHECK-NEXT: %[[COMPLETION_1:.*]] = ttkernel.get_common_arg_val(%[[COMPLETION_1_ARG]])
// CHECK: %[[DATA_NOC_1:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[BLOCK_1]],
// CHECK: %[[COMPLETION_NOC_1:.*]] = ttkernel.get_noc_addr({{.*}}, {{.*}}, %[[COMPLETION_1]],
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc(%[[CONNECTION_MANAGER]], %[[CONNECTION_COUNT]], {{.*}}, {{.*}}, {{.*}}, %[[PAYLOAD_1]], %[[BLOCK_BYTES]], %[[DATA_NOC_1]], %[[COMPLETION_NOC_1]], %[[ONE]])
// CHECK-NOT: ttkernel.routing_plane.fused_write_atomic_inc
// CHECK-LABEL: func.func @receiver
// CHECK-DAG: %[[RECEIVER_ONE:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[RECEIVER_TWO:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[RECEIVER_COMPLETION_0_ARG:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[RECEIVER_COMPLETION_1_ARG:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[RECEIVER_DEVICE_ARG:.*]] = arith.constant 2 : index
// CHECK: %[[RECEIVER_DFB:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK: %[[RECEIVER_DEVICE_0:.*]] = ttkernel.get_common_arg_val(%[[RECEIVER_DEVICE_ARG]])
// CHECK-NEXT: %[[IS_RECEIVER_DEVICE_0:.*]] = arith.cmpi eq, %[[RECEIVER_DEVICE_0]], %[[RECEIVER_TWO]]
// CHECK-NEXT: scf.if %[[IS_RECEIVER_DEVICE_0]] {
// CHECK-NEXT: ttkernel.cb_reserve_back(%[[RECEIVER_DFB]], %[[RECEIVER_ONE]])
// CHECK: %[[COMPLETION_COUNT_0:.*]] = arith.addi {{.*}}, %[[RECEIVER_ONE]] : i32
// CHECK: %[[COMPLETION_ADDRESS_0:.*]] = ttkernel.get_common_arg_val(%[[RECEIVER_COMPLETION_0_ARG]])
// CHECK-NEXT: %[[COMPLETION_POINTER_0:.*]] = ttkernel.reinterpret_cast(%[[COMPLETION_ADDRESS_0]])
// CHECK-NEXT: ttkernel.experimental.semaphore_wait_min(%[[COMPLETION_POINTER_0]], %[[COMPLETION_COUNT_0]])
// CHECK-NEXT: ttkernel.cb_push_back(%[[RECEIVER_DFB]], %[[RECEIVER_ONE]])
// CHECK-NEXT: }
// CHECK-NEXT: %[[RECEIVER_DEVICE_1:.*]] = ttkernel.get_common_arg_val(%[[RECEIVER_DEVICE_ARG]])
// CHECK-NEXT: %[[IS_RECEIVER_DEVICE_1:.*]] = arith.cmpi eq, %[[RECEIVER_DEVICE_1]], %[[RECEIVER_TWO]]
// CHECK-NEXT: scf.if %[[IS_RECEIVER_DEVICE_1]] {
// CHECK-NEXT: ttkernel.cb_reserve_back(%[[RECEIVER_DFB]], %[[RECEIVER_ONE]])
// CHECK: %[[COMPLETION_COUNT_1:.*]] = arith.addi {{.*}}, %[[RECEIVER_ONE]] : i32
// CHECK: %[[COMPLETION_ADDRESS_1:.*]] = ttkernel.get_common_arg_val(%[[RECEIVER_COMPLETION_1_ARG]])
// CHECK-NEXT: %[[COMPLETION_POINTER_1:.*]] = ttkernel.reinterpret_cast(%[[COMPLETION_ADDRESS_1]])
// CHECK-NEXT: ttkernel.experimental.semaphore_wait_min(%[[COMPLETION_POINTER_1]], %[[COMPLETION_COUNT_1]])
// CHECK-NEXT: ttkernel.cb_push_back(%[[RECEIVER_DFB]], %[[RECEIVER_ONE]])
// CHECK-NOT: ttkernel.cb_reserve_back
// CHECK-NOT: ttkernel.experimental.semaphore_wait_min
// CHECK-NEXT: }
// CHECK-NEXT: return

#domain = #ttl.device_domain<components = <name = "device", extent = [3]>>
#transfer_0 = #ttl.device_transfer<
    domain = #domain,
    edge = <source = <coordinates = [0]>, destination = <coordinates = [2]>>>
#transfer_1 = #ttl.device_transfer<
    domain = #domain,
    edge = <source = <coordinates = [1]>, destination = <coordinates = [2]>>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @senders() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe_0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #transfer_0}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %pipe_1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #transfer_1}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %is_source_0 = ttl.is_device <coordinates = [0]> in #domain : i1
    scf.if %is_source_0 {
      %send_0 = ttl.copy %source, %pipe_0
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send_0 : !ttl.transfer_handle<write>
    }
    %is_source_1 = ttl.is_device <coordinates = [1]> in #domain : i1
    scf.if %is_source_1 {
      %send_1 = ttl.copy %source, %pipe_1
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send_1 : !ttl.transfer_handle<write>
    }
    func.return
  }

  func.func @receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %pipe_0 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #transfer_0}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %pipe_1 = ttl.create_pipe src(0, 0) dst(0, 0) to(0, 0) net 0 {
        deviceTransfer = #transfer_1}
        : !ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>
    %is_destination_0 = ttl.is_device <coordinates = [2]> in #domain : i1
    scf.if %is_destination_0 {
      %reserve_0 = ttl.cb_reserve %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post_0 = ttl.copy %pipe_0, %reserve_0
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %post_0 : !ttl.transfer_handle
      ttl.cb_push %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    %is_destination_1 = ttl.is_device <coordinates = [2]> in #domain : i1
    scf.if %is_destination_1 {
      %reserve_1 = ttl.cb_reserve %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %post_1 = ttl.copy %pipe_1, %reserve_1
          : (!ttl.pipe<src(0, 0) dst(0, 0) to(0, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.transfer_handle
      ttl.wait %post_1 : !ttl.transfer_handle
      ttl.cb_push %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 2>
    }
    func.return
  }
}
