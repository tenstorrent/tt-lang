// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// Summary: Verifies stateful one-packet write setup and issue contracts.

// The issue operation must use the response mode programmed in the resident
// write command.
func.func @posted_mode_mismatch(
    %state_address: !ttkernel.noc_addr, %source_address: i32,
    %destination_address: i32, %size: i32, %noc: i8) {
  ttkernel.noc_async_write_one_packet_set_state(
      %state_address, %size, noc %noc) posted true
      : (!ttkernel.noc_addr, i32, i8) -> ()
  // expected-error @below {{'ttkernel.noc_async_write_one_packet_with_state' op posted mode must match the preceding one-packet write state setup}}
  ttkernel.noc_async_write_one_packet_with_state(
      %source_address, %destination_address, noc %noc)
      : (i32, i32, i8) -> ()
  func.return
}

// -----

// A conditional setup cannot initialize state for an unconditional issue.
func.func @conditional_setup(
    %condition: i1, %state_address: !ttkernel.noc_addr,
    %source_address: i32, %destination_address: i32, %size: i32, %noc: i8) {
  scf.if %condition {
    ttkernel.noc_async_write_one_packet_set_state(
        %state_address, %size, noc %noc)
        : (!ttkernel.noc_addr, i32, i8) -> ()
  }
  // expected-error @below {{'ttkernel.noc_async_write_one_packet_with_state' op requires a preceding one-packet write state setup on the same NoC whose execution conditions cover this operation}}
  ttkernel.noc_async_write_one_packet_with_state(
      %source_address, %destination_address, noc %noc)
      : (i32, i32, i8) -> ()
  func.return
}

// -----

// State programmed on another NoC does not configure this issue operation.
func.func @different_noc(
    %state_address: !ttkernel.noc_addr, %source_address: i32,
    %destination_address: i32, %size: i32, %setup_noc: i8, %use_noc: i8) {
  ttkernel.noc_async_write_one_packet_set_state(
      %state_address, %size, noc %setup_noc)
      : (!ttkernel.noc_addr, i32, i8) -> ()
  // expected-error @below {{'ttkernel.noc_async_write_one_packet_with_state' op requires a preceding one-packet write state setup on the same NoC whose execution conditions cover this operation}}
  ttkernel.noc_async_write_one_packet_with_state(
      %source_address, %destination_address, noc %use_noc)
      : (i32, i32, i8) -> ()
  func.return
}

// -----

// A setup that executes conditionally between the common setup and issue can
// change the resident destination, size, and response mode for some issues.
func.func @conditional_intervening_setup(
    %condition: i1, %initial_state_address: !ttkernel.noc_addr,
    %conditional_state_address: !ttkernel.noc_addr, %source_address: i32,
    %destination_address: i32, %size: i32, %noc: i8) {
  ttkernel.noc_async_write_one_packet_set_state(
      %initial_state_address, %size, noc %noc)
      : (!ttkernel.noc_addr, i32, i8) -> ()
  scf.if %condition {
    // expected-note @below {{this setup may replace the selected state before a later issue}}
    ttkernel.noc_async_write_one_packet_set_state(
        %conditional_state_address, %size, noc %noc) posted true
        : (!ttkernel.noc_addr, i32, i8) -> ()
  }
  // expected-error @below {{'ttkernel.noc_async_write_one_packet_with_state' op cannot identify one preceding write state setup for every execution}}
  ttkernel.noc_async_write_one_packet_with_state(
      %source_address, %destination_address, noc %noc)
      : (i32, i32, i8) -> ()
  func.return
}
