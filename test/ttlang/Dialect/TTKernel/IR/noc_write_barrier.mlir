// RUN: ttlang-opt %s --canonicalize | FileCheck %s

// Summary: Verifies a stateful write prevents removal of a subsequent write
// barrier, while redundant barriers without an intervening write are removed.

// CHECK-LABEL: func.func @preserve_after_stateful_write
// CHECK: ttkernel.noc_async_write_barrier
// CHECK: ttkernel.noc_async_write_one_packet_with_state
// CHECK: ttkernel.noc_async_write_barrier
func.func @preserve_after_stateful_write() {
  %coordinate = arith.constant 0 : index
  %noc = arith.constant 0 : i8
  %destination_address = arith.constant 4096 : i32
  %source_address = arith.constant 8192 : i32
  %size = arith.constant 896 : i32
  %destination_noc_address = ttkernel.get_noc_addr(
      %coordinate, %coordinate, %destination_address, %noc)
      : (index, index, i32, i8) -> !ttkernel.noc_addr
  ttkernel.noc_async_write_one_packet_set_state(
      %destination_noc_address, %size, noc %noc)
      : (!ttkernel.noc_addr, i32, i8) -> ()
  ttkernel.noc_async_write_one_packet_with_state(
      %source_address, %destination_address, noc %noc)
      : (i32, i32, i8) -> ()
  ttkernel.noc_async_write_barrier(%noc) : (i8) -> ()
  ttkernel.noc_async_write_one_packet_with_state(
      %source_address, %destination_address, noc %noc)
      : (i32, i32, i8) -> ()
  ttkernel.noc_async_write_barrier(%noc) : (i8) -> ()
  func.return
}

// CHECK-LABEL: func.func @remove_redundant_barrier
// CHECK-COUNT-1: ttkernel.noc_async_write_barrier
func.func @remove_redundant_barrier() {
  %noc = arith.constant 0 : i8
  ttkernel.noc_async_write_barrier(%noc) : (i8) -> ()
  ttkernel.noc_async_write_barrier(%noc) : (i8) -> ()
  func.return
}

// CHECK-LABEL: func.func @departure_wait_does_not_issue_write
// CHECK-COUNT-1: ttkernel.noc_async_write_barrier
// CHECK: ttkernel.noc_async_writes_flushed
func.func @departure_wait_does_not_issue_write() {
  %noc = arith.constant 0 : i8
  ttkernel.noc_async_write_barrier(%noc) : (i8) -> ()
  ttkernel.noc_async_writes_flushed(%noc) : (i8) -> ()
  ttkernel.noc_async_write_barrier(%noc) : (i8) -> ()
  func.return
}
