// RUN: ttlang-opt %s --split-input-file --verify-each -o /dev/null

// Summary: Preserve valid state reuse across independent commands, calls,
// shared conditional entry, and per-iteration setup restoration.

// A pure helper and an independent atomic command preserve the selected write state.
func.func private @pure_helper(%value: i32) -> i32 {
  return %value : i32
}
func.func @preserving_calls_and_atomics(
    %state_address: !ttkernel.noc_addr, %source_address: i32,
    %destination_address: i32, %size: i32, %noc: i8,
    %coordinate: index, %value: i32, %byte_enable: i8) {
  ttkernel.noc_async_write_one_packet_set_state(
      %state_address, %size, noc %noc) : (!ttkernel.noc_addr, i32, i8) -> ()
  %unchanged = func.call @pure_helper(%value) : (i32) -> i32
  ttkernel.noc_semaphore_inc(%state_address, %coordinate, %noc)
      : (!ttkernel.noc_addr, index, i8) -> ()
  ttkernel.noc_async_write_one_packet_with_state(
      %source_address, %destination_address, noc %noc) : (i32, i32, i8) -> ()
  return
}

// -----

// Setup and issue inside the same switch case have identical entry conditions.
func.func @same_switch_case(
    %state_address: !ttkernel.noc_addr, %source_address: i32,
    %destination_address: i32, %size: i32, %noc: i8,
    %coordinate: index, %value: i32, %byte_enable: i8) {
  scf.index_switch %coordinate
  case 0 {
    ttkernel.noc_async_write_one_packet_set_state(
        %state_address, %size, noc %noc) : (!ttkernel.noc_addr, i32, i8) -> ()
    ttkernel.noc_async_write_one_packet_with_state(
        %source_address, %destination_address, noc %noc) : (i32, i32, i8) -> ()
    scf.yield
  }
  default {
    scf.yield
  }
  return
}

// -----

// Restoring setup every iteration permits an inline write after each issue.
func.func @restored_after_inline(
    %state_address: !ttkernel.noc_addr, %source_address: i32,
    %destination_address: i32, %size: i32, %noc: i8,
    %coordinate: index, %value: i32, %byte_enable: i8) {
  %lower = arith.constant 0 : index
  %upper = arith.constant 2 : index
  %step = arith.constant 1 : index
  scf.for %iteration = %lower to %upper step %step {
    ttkernel.noc_async_write_one_packet_set_state(
        %state_address, %size, noc %noc) : (!ttkernel.noc_addr, i32, i8) -> ()
    ttkernel.noc_async_write_one_packet_with_state(
        %source_address, %destination_address, noc %noc) : (i32, i32, i8) -> ()
    ttkernel.noc_inline_dw_write(
        core[%coordinate, %coordinate], %destination_address, %value,
        %byte_enable, noc %noc) : (index, index, i32, i32, i8, i8) -> ()
  }
  return
}
