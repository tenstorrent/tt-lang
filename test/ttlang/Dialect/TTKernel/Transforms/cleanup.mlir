// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verifies TTKernel cleanup optimizations for posted NoC writes.

// Destination translation and one-packet command setup can precede the
// sender-ready wait because the intervening semaphore reset preserves NoC
// write-command state.
// CHECK-LABEL: func.func @schedule_posted_write_state_before_wait
// CHECK: %[[DST_X:.*]] = ttkernel.experimental.convert_logical_x_to_translated
// CHECK: %[[DST_Y:.*]] = ttkernel.experimental.convert_logical_y_to_translated
// CHECK: %[[NOC_ADDR:.*]] = ttkernel.get_noc_addr(%[[DST_X]], %[[DST_Y]],
// CHECK-NEXT: ttkernel.noc_async_write_one_packet_set_state(%[[NOC_ADDR]], {{.*}}) posted true
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.noc_semaphore_set
// CHECK: ttkernel.noc_async_write_one_packet_with_state({{.*}}) posted true
// CHECK-NOT: ttkernel.noc_async_write %
func.func @schedule_posted_write_state_before_wait(%src: i32, %dst: i32) {
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

// -----

// A non-posted write retains completion semantics and cannot move command
// setup before the wait.
// CHECK-LABEL: func.func @preserve_nonposted_write
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.noc_async_write %{{.*}}, core
// CHECK-NOT: posted true
// CHECK: return
func.func @preserve_nonposted_write(%src: i32, %dst: i32) {
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
  %dst_x = ttkernel.experimental.convert_logical_x_to_translated(%c1)
      : (index) -> index
  %dst_y = ttkernel.experimental.convert_logical_y_to_translated(%c0)
      : (index) -> index
  ttkernel.noc_async_write
      %src, core[%dst_x, %dst_y], %dst, %size, noc %c0_i8
      : (i32, index, index, i32, i32, i8) -> ()
  func.return
}

// -----

// Intervening command-state configuration prevents the posted write from
// moving state setup before the wait.
// CHECK-LABEL: func.func @preserve_posted_write_after_interference
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: ttkernel.noc_async_write_one_packet_set_state
// CHECK: ttkernel.noc_async_write %{{.*}}, core{{.*}}posted true
// CHECK-NOT: ttkernel.noc_async_write_one_packet_with_state
// CHECK: return
func.func @preserve_posted_write_after_interference(%src: i32, %dst: i32) {
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
  %state_addr = ttkernel.get_noc_addr(%c0, %c0, %dst, %c0_i8)
      : (index, index, i32, i8) -> !ttkernel.noc_addr
  ttkernel.noc_async_write_one_packet_set_state(
      %state_addr, %size, noc %c0_i8)
      : (!ttkernel.noc_addr, i32, i8) -> ()
  %dst_x = ttkernel.experimental.convert_logical_x_to_translated(%c1)
      : (index) -> index
  %dst_y = ttkernel.experimental.convert_logical_y_to_translated(%c0)
      : (index) -> index
  ttkernel.noc_async_write
      %src, core[%dst_x, %dst_y], %dst, %size, noc %c0_i8 posted true
      : (i32, index, index, i32, i32, i8) -> ()
  func.return
}

// -----

// A destination coordinate loaded after the wait cannot move before it. The
// generic posted write remains valid without speculative memory motion.
// CHECK-LABEL: func.func @preserve_posted_write_with_loaded_coordinate
// CHECK-NOT: ttkernel.noc_async_write_one_packet_set_state
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: %[[LOADED_X:.*]] = memref.load
// CHECK: ttkernel.noc_async_write %{{.*}}, core[%[[LOADED_X]], {{.*}}]{{.*}}posted true
// CHECK-NOT: ttkernel.noc_async_write_one_packet_with_state
// CHECK: return
func.func @preserve_posted_write_with_loaded_coordinate(
    %src: i32, %dst: i32, %coordinates: memref<2xindex>) {
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
  %dst_x = memref.load %coordinates[%c0] : memref<2xindex>
  %dst_y = memref.load %coordinates[%c1] : memref<2xindex>
  ttkernel.noc_async_write
      %src, core[%dst_x, %dst_y], %dst, %size, noc %c0_i8 posted true
      : (i32, index, index, i32, i32, i8) -> ()
  func.return
}

// -----

// A nested call that can reconfigure write-command state prevents setup from
// moving across the earlier wait.
// CHECK-LABEL: func.func @preserve_posted_write_after_nested_call
// CHECK-NOT: ttkernel.noc_async_write_one_packet_set_state
// CHECK: ttkernel.experimental.semaphore_wait
// CHECK: scf.if
// CHECK: func.call @configure_write_command
// CHECK: ttkernel.noc_async_write %{{.*}}, core{{.*}}posted true
// CHECK-NOT: ttkernel.noc_async_write_one_packet_with_state
// CHECK: return
func.func private @configure_write_command(%dst: i32) {
  %c0 = arith.constant 0 : index
  %c0_i8 = arith.constant 0 : i8
  %size = arith.constant 896 : i32
  %state_addr = ttkernel.get_noc_addr(%c0, %c0, %dst, %c0_i8)
      : (index, index, i32, i8) -> !ttkernel.noc_addr
  ttkernel.noc_async_write_one_packet_set_state(
      %state_addr, %size, noc %c0_i8)
      : (!ttkernel.noc_addr, i32, i8) -> ()
  func.return
}

func.func @preserve_posted_write_after_nested_call(
    %src: i32, %dst: i32, %condition: i1) {
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
  scf.if %condition {
    func.call @configure_write_command(%dst) : (i32) -> ()
  }
  %dst_x = ttkernel.experimental.convert_logical_x_to_translated(%c1)
      : (index) -> index
  %dst_y = ttkernel.experimental.convert_logical_y_to_translated(%c0)
      : (index) -> index
  ttkernel.noc_async_write
      %src, core[%dst_x, %dst_y], %dst, %size, noc %c0_i8 posted true
      : (i32, index, index, i32, i32, i8) -> ()
  func.return
}
