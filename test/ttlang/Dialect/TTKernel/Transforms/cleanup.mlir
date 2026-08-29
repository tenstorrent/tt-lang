// RUN: ttlang-opt %s -ttkernel-cleanup | FileCheck %s

// Summary: Verifies that the reusable cleanup pass applies TTKernel command
// optimizations exposed by specialization and loop unrolling.

// The translated destination coordinates and one-packet command setup move
// before the blocking sender-ready wait. The local semaphore reset preserves
// resident asynchronous-write command state.
// CHECK-LABEL: func.func @prearm_posted_write
// CHECK: %[[DST_X:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[NOC_ADDR:.*]] = ttkernel.get_noc_addr(%[[DST_X]], %[[DST_Y]],
// CHECK-NEXT: ttkernel.noc_async_write_one_packet_set_state(%[[NOC_ADDR]], {{.*}}) posted true
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.noc_semaphore_set
// CHECK: ttkernel.noc_async_write_one_packet_with_state({{.*}}) posted true
// CHECK-NOT: ttkernel.noc_async_write{{[ (]}}
func.func @prearm_posted_write(%src: i32, %dst: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i8 = arith.constant 0 : i8
  %c1_i32 = arith.constant 1 : i32
  %size = arith.constant 896 : i32
  %ready_sem = ttkernel.get_semaphore(%c1)
      : (index) -> !ttkernel.local_semaphore
  %ready_addr = ttkernel.reinterpret_cast(%ready_sem)
      : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
  ttkernel.experimental.semaphore_wait(%ready_addr, %c1_i32)
      : (!ttkernel.l1_addr_ptr, i32) -> ()
  ttkernel.noc_semaphore_set(%ready_addr, %c0)
      : (!ttkernel.l1_addr_ptr, index) -> ()
  %dst_x = ttkernel.experimental.convert_logical_x_to_translated(%c1)
      : (index) -> index
  %dst_y = ttkernel.experimental.convert_logical_y_to_translated(%c0)
      : (index) -> index
  ttkernel.noc_async_write
      %src, core[%dst_x, %dst_y], %dst, %size, noc %c0_i8 posted true
      : (i32, index, index, i32, i32, i8) -> ()
  func.return
}

// Configure the one-packet fused command before both blocking waits while
// retaining the general packetized operation when the runtime packet limit is
// smaller than the constant transfer.
// CHECK-LABEL: func.func @preconfigure_fabric_write
// CHECK: %[[CONFIGURED:.*]] = ttkernel.routing_plane.set_fused_write_atomic_inc_state(
// CHECK-SAME: ) : (i32, i32, i32, !ttkernel.noc_addr, !ttkernel.noc_addr, i32) -> i1
// CHECK: ttkernel.cb_wait_front
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: scf.if %[[CONFIGURED]] {
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc_with_state({{.*}}) posted true
// CHECK: } else {
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc({{.*}}) {posted = true}
// CHECK: }
func.func @preconfigure_fabric_write(
    %manager: !ttkernel.routing_plane_connection_manager,
    %source: i32, %destination: !ttkernel.noc_addr,
    %remote_semaphore: !ttkernel.noc_addr) {
  %c0 = arith.constant 0 : index
  %route_id = arith.constant 0 : i32
  %route_index = arith.constant 0 : i32
  %connection_index = arith.constant 0 : i32
  %size = arith.constant 64 : i32
  %increment = arith.constant 1 : i32
  %ready = arith.constant 1 : i32
  %dataflow_buffer = ttkernel.get_compile_time_arg_val(0)
      : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %ready_semaphore = ttkernel.get_semaphore(%c0)
      : (index) -> !ttkernel.local_semaphore
  %ready_address = ttkernel.reinterpret_cast(%ready_semaphore)
      : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
  ttkernel.cb_wait_front(%dataflow_buffer, %ready)
      : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, i32) -> ()
  ttkernel.experimental.semaphore_wait(%ready_address, %ready)
      : (!ttkernel.l1_addr_ptr, i32) -> ()
  ttkernel.routing_plane.fused_write_atomic_inc(
      %manager, %route_id, %route_index, %connection_index,
      %source, %size, %destination, %remote_semaphore, %increment)
      {posted = true}
      : (!ttkernel.routing_plane_connection_manager, i32, i32, i32, i32,
         i32, !ttkernel.noc_addr, !ttkernel.noc_addr, i32) -> ()
  func.return
}
