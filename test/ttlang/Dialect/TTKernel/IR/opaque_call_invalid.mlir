// Verify that opaque_call rejects invalid inputs.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// -----
// Test: empty callee name
func.func @empty_callee() {
  // expected-error @below {{'ttkernel.opaque_call' op callee name must not be empty}}
  ttkernel.opaque_call "" () {header = "h.hpp"} : () -> ()
  return
}

// -----
// Test: empty header path
func.func @empty_header() {
  // expected-error @below {{'ttkernel.opaque_call' op header path must not be empty}}
  ttkernel.opaque_call "foo" () {header = ""} : () -> ()
  return
}

// -----
// Test: block argument used as template arg
func.func @block_arg_template_arg(%arg0: i32) {
  // expected-error @below {{'ttkernel.opaque_call' op template arg must be a compile-time evaluable value (arith.constant or ttkernel.get_dfb_id), got a block argument}}
  ttkernel.opaque_call "foo" template_args(%arg0) () {header = "h.hpp"} : () -> ()
  return
}

// -----
// Test: runtime value used as template arg
func.func @runtime_template_arg(%arg0: i32) {
  %sum = arith.addi %arg0, %arg0 : i32
  // expected-error @below {{'ttkernel.opaque_call' op template arg must be a compile-time evaluable value (arith.constant or ttkernel.get_dfb_id), got 'arith.addi'}}
  ttkernel.opaque_call "foo" template_args(%sum) () {header = "h.hpp"} : () -> ()
  return
}
