// Verify that opaque_call rejects non-constant template args.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// -----
// Test: block argument used as template arg
func.func @block_arg_template_arg(%arg0: i32) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  // expected-error @below {{'ttl.opaque_call' op template arg must be a compile-time evaluable value (arith.constant or ttl.get_dfb_id), got a block argument}}
  ttl.opaque_call "foo" template_args(%arg0) () {header = "h.hpp"} : () -> ()
  return
}

// -----
// Test: runtime value used as template arg
func.func @runtime_template_arg(%arg0: i32) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %sum = arith.addi %arg0, %arg0 : i32
  // expected-error @below {{'ttl.opaque_call' op template arg must be a compile-time evaluable value (arith.constant or ttl.get_dfb_id), got 'arith.addi'}}
  ttl.opaque_call "foo" template_args(%sum) () {header = "h.hpp"} : () -> ()
  return
}
