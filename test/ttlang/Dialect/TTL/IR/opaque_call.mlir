// Round-trip verification of ttl.opaque_call with template arg SSA values.
// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// -----
// Test: void call with no template args
// CHECK-LABEL: func.func @void_no_template_args
// CHECK: ttl.opaque_call "my_func" () {header = "my_header.hpp"} : () -> ()
func.func @void_no_template_args() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "my_func" () {header = "my_header.hpp"} : () -> ()
  return
}

// -----
// Test: call with constant template args
// CHECK-LABEL: func.func @const_template_args
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
// CHECK: ttl.opaque_call "add_vals" template_args(%[[C1]], %[[C2]]) () {header = "math.hpp"} : () -> ()
func.func @const_template_args() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  ttl.opaque_call "add_vals" template_args(%c1, %c2) () {header = "math.hpp"} : () -> ()
  return
}

// -----
// Test: call with func args and template args
// CHECK-LABEL: func.func @func_and_template_args
// CHECK: %[[TA:.*]] = arith.constant 42 : i32
// CHECK: ttl.opaque_call "process" template_args(%[[TA]]) (%arg0) {header = "proc.hpp"} : (i32) -> ()
func.func @func_and_template_args(%arg0: i32) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %ta = arith.constant 42 : i32
  ttl.opaque_call "process" template_args(%ta) (%arg0) {header = "proc.hpp"} : (i32) -> ()
  return
}

// -----
// Test: call with get_dfb_id template arg
// CHECK-LABEL: func.func @dfb_template_arg
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2}
// CHECK-NEXT: %[[ID:.*]] = ttl.get_dfb_id %[[CB]]
// CHECK-NEXT: ttl.opaque_call "drain" template_args(%[[ID]]) () {header = "drain.hpp"} : () -> ()
func.func @dfb_template_arg() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %id = ttl.get_dfb_id %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.opaque_call "drain" template_args(%id) () {header = "drain.hpp"} : () -> ()
  return
}
