// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// Summary: Reject overwritten write state and conditional setups that do not
// execute on every issue.

// An inline word write replaces the resident write command.
func.func @inline_clobber(
    %state_address: !ttkernel.noc_addr, %source_address: i32,
    %destination_address: i32, %size: i32, %noc: i8,
    %coordinate: index, %value: i32, %byte_enable: i8) {
  ttkernel.noc_async_write_one_packet_set_state(
      %state_address, %size, noc %noc) : (!ttkernel.noc_addr, i32, i8) -> ()
  // expected-note @below {{this operation may replace the selected state before a later issue}}
  ttkernel.noc_inline_dw_write(
      core[%coordinate, %coordinate], %destination_address, %value,
      %byte_enable, noc %noc) : (index, index, i32, i32, i8, i8) -> ()
  // expected-error @below {{cannot identify one preceding write state setup for every execution}}
  ttkernel.noc_async_write_one_packet_with_state(
      %source_address, %destination_address, noc %noc) : (i32, i32, i8) -> ()
  return
}

// -----

// An ordinary payload write also replaces command state.
func.func @ordinary_clobber(
    %state_address: !ttkernel.noc_addr, %source_address: i32,
    %destination_address: i32, %size: i32, %noc: i8,
    %coordinate: index, %value: i32, %byte_enable: i8) {
  ttkernel.noc_async_write_one_packet_set_state(
      %state_address, %size, noc %noc) : (!ttkernel.noc_addr, i32, i8) -> ()
  // expected-note @below {{this operation may replace the selected state before a later issue}}
  ttkernel.noc_async_write %source_address,
      core[%coordinate, %coordinate], %destination_address, %size, noc %noc
      : (i32, index, index, i32, i32, i8) -> ()
  // expected-error @below {{cannot identify one preceding write state setup for every execution}}
  ttkernel.noc_async_write_one_packet_with_state(
      %source_address, %destination_address, noc %noc) : (i32, i32, i8) -> ()
  return
}

// -----

// A resolved helper's write effect must be attributed to its call.
func.func @write_helper(
    %state_address: !ttkernel.noc_addr, %source_address: i32,
    %destination_address: i32, %size: i32, %noc: i8,
    %coordinate: index, %value: i32, %byte_enable: i8) {
  ttkernel.noc_inline_dw_write(
      core[%coordinate, %coordinate], %destination_address, %value,
      %byte_enable, noc %noc) : (index, index, i32, i32, i8, i8) -> ()
  return
}
func.func @called_clobber(
    %state_address: !ttkernel.noc_addr, %source_address: i32,
    %destination_address: i32, %size: i32, %noc: i8,
    %coordinate: index, %value: i32, %byte_enable: i8) {
  ttkernel.noc_async_write_one_packet_set_state(
      %state_address, %size, noc %noc) : (!ttkernel.noc_addr, i32, i8) -> ()
  // expected-note @below {{this operation may replace the selected state before a later issue}}
  func.call @write_helper(%state_address, %source_address, %destination_address, %size, %noc, %coordinate, %value, %byte_enable)
      : (!ttkernel.noc_addr, i32, i32, i32, i8, index, i32, i8) -> ()
  // expected-error @below {{cannot identify one preceding write state setup for every execution}}
  ttkernel.noc_async_write_one_packet_with_state(
      %source_address, %destination_address, noc %noc) : (i32, i32, i8) -> ()
  return
}

// -----

// An external helper has unknown effects on resident command state.
func.func private @external_helper()
func.func @unknown_call(
    %state_address: !ttkernel.noc_addr, %source_address: i32,
    %destination_address: i32, %size: i32, %noc: i8,
    %coordinate: index, %value: i32, %byte_enable: i8) {
  ttkernel.noc_async_write_one_packet_set_state(
      %state_address, %size, noc %noc) : (!ttkernel.noc_addr, i32, i8) -> ()
  // expected-note @below {{this operation may replace the selected state before a later issue}}
  func.call @external_helper() : () -> ()
  // expected-error @below {{cannot identify one preceding write state setup for every execution}}
  ttkernel.noc_async_write_one_packet_with_state(
      %source_address, %destination_address, noc %noc) : (i32, i32, i8) -> ()
  return
}

// -----

// Recursive summaries remain conservative rather than treating a cycle as pure.
func.func private @recursive_helper() {
  func.call @recursive_helper() : () -> ()
  return
}
func.func @recursive_call(
    %state_address: !ttkernel.noc_addr, %source_address: i32,
    %destination_address: i32, %size: i32, %noc: i8,
    %coordinate: index, %value: i32, %byte_enable: i8) {
  ttkernel.noc_async_write_one_packet_set_state(
      %state_address, %size, noc %noc) : (!ttkernel.noc_addr, i32, i8) -> ()
  // expected-note @below {{this operation may replace the selected state before a later issue}}
  func.call @recursive_helper() : () -> ()
  // expected-error @below {{cannot identify one preceding write state setup for every execution}}
  ttkernel.noc_async_write_one_packet_with_state(
      %source_address, %destination_address, noc %noc) : (i32, i32, i8) -> ()
  return
}

// -----

// A setup in only one switch case does not cover an issue after the switch.
func.func @switch_setup(
    %state_address: !ttkernel.noc_addr, %source_address: i32,
    %destination_address: i32, %size: i32, %noc: i8,
    %coordinate: index, %value: i32, %byte_enable: i8) {
  scf.index_switch %coordinate
  case 0 {
    ttkernel.noc_async_write_one_packet_set_state(
        %state_address, %size, noc %noc) : (!ttkernel.noc_addr, i32, i8) -> ()
    scf.yield
  }
  default {
    scf.yield
  }
  // expected-error @below {{requires a preceding one-packet write state setup on the same NoC whose execution conditions cover this operation}}
  ttkernel.noc_async_write_one_packet_with_state(
      %source_address, %destination_address, noc %noc) : (i32, i32, i8) -> ()
  return
}

// -----

// An inline write after an issue invalidates state for the next iteration.
func.func @loop_carried_inline_clobber(
    %state_address: !ttkernel.noc_addr, %source_address: i32,
    %destination_address: i32, %size: i32, %noc: i8,
    %coordinate: index, %value: i32, %byte_enable: i8) {
  %lower = arith.constant 0 : index
  %upper = arith.constant 2 : index
  %step = arith.constant 1 : index
  ttkernel.noc_async_write_one_packet_set_state(
      %state_address, %size, noc %noc) : (!ttkernel.noc_addr, i32, i8) -> ()
  scf.for %iteration = %lower to %upper step %step {
    // expected-error @below {{cannot identify one preceding write state setup for every execution}}
    ttkernel.noc_async_write_one_packet_with_state(
        %source_address, %destination_address, noc %noc) : (i32, i32, i8) -> ()
    // expected-note @below {{this operation may replace the selected state before a later issue}}
    ttkernel.noc_inline_dw_write(
        core[%coordinate, %coordinate], %destination_address, %value,
        %byte_enable, noc %noc) : (index, index, i32, i32, i8, i8) -> ()
  }
  return
}
