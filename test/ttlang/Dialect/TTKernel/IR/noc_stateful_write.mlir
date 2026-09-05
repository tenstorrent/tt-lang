// Summary: Verifies valid stateful one-packet write setup and issue contracts.
// RUN: ttlang-opt %s -verify-each -o /dev/null

// Proven-distinct constant NoC selectors do not interfere with resident state.
func.func @distinct_constant_noc_does_not_interfere(
    %initial_state_address: !ttkernel.noc_addr,
    %intervening_state_address: !ttkernel.noc_addr, %source_address: i32,
    %destination_address: i32, %size: i32) {
  %noc0 = arith.constant 0 : i8
  %noc1 = arith.constant 1 : i8
  ttkernel.noc_async_write_one_packet_set_state(
      %initial_state_address, %size, noc %noc0)
      : (!ttkernel.noc_addr, i32, i8) -> ()
  ttkernel.noc_async_write_one_packet_set_state(
      %intervening_state_address, %size, noc %noc1) posted true
      : (!ttkernel.noc_addr, i32, i8) -> ()
  ttkernel.noc_async_write_one_packet_with_state(
      %source_address, %destination_address, noc %noc0)
      : (i32, i32, i8) -> ()
  func.return
}

// A setup at the start of each iteration restores state changed after the
// preceding issue.
func.func @loop_setup_restores_state_before_each_issue(
    %initial_state_address: !ttkernel.noc_addr,
    %later_state_address: !ttkernel.noc_addr, %source_address: i32,
    %destination_address: i32, %size: i32, %noc: i8) {
  %lower = arith.constant 0 : index
  %upper = arith.constant 2 : index
  %step = arith.constant 1 : index
  scf.for %iteration = %lower to %upper step %step {
    ttkernel.noc_async_write_one_packet_set_state(
        %initial_state_address, %size, noc %noc)
        : (!ttkernel.noc_addr, i32, i8) -> ()
    ttkernel.noc_async_write_one_packet_with_state(
        %source_address, %destination_address, noc %noc)
        : (i32, i32, i8) -> ()
    ttkernel.noc_async_write_one_packet_set_state(
        %later_state_address, %size, noc %noc) posted true
        : (!ttkernel.noc_addr, i32, i8) -> ()
  }
  func.return
}
