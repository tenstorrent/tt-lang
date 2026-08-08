// Verify ttl.opaque_call preserves ordered static arguments and DFB identity.
// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// A void call omits both optional template segments.
// CHECK-LABEL: func.func @void_no_template_args
// CHECK: ttl.opaque_call "my_func" () {header = "my_header.hpp"} : () -> ()
func.func @void_no_template_args() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "my_func" () {header = "my_header.hpp"} : () -> ()
  return
}

// -----

// Scalar template arguments are attributes rather than artificial SSA values.
// CHECK-LABEL: func.func @scalar_template_args
// CHECK: ttl.opaque_call "add_vals" template_args [#ttl.external_template_arg<signed_integer, 1>, #ttl.external_template_arg<boolean, 1>, #ttl.external_template_arg<unsigned_integer, 3212836864>] () {header = "math.hpp"} : () -> ()
func.func @scalar_template_args() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "add_vals" template_args [#ttl.external_template_arg<signed_integer, 1>, #ttl.external_template_arg<boolean, 1>, #ttl.external_template_arg<unsigned_integer, 3212836864>] () {header = "math.hpp"} : () -> ()
  return
}

// -----

// Function operands remain independent from the static argument list.
// CHECK-LABEL: func.func @func_and_template_args
// CHECK: ttl.opaque_call "process" template_args [#ttl.external_template_arg<signed_integer, 42>] (%arg0) {header = "proc.hpp"} : (i32) -> ()
func.func @func_and_template_args(%arg0: i32) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "process" template_args [#ttl.external_template_arg<signed_integer, 42>] (%arg0) {header = "proc.hpp"} : (i32) -> ()
  return
}

// -----

// Unsigned address semantics remain explicit at the external call boundary.
// CHECK-LABEL: func.func @unsigned_func_arg
// CHECK: ttl.opaque_call "use_address" (%arg0) {header = "address.hpp", unsigned_arg_indices = array<i32: 0>} : (i32) -> ()
func.func @unsigned_func_arg(%arg0: i32) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "use_address" (%arg0) {header = "address.hpp", unsigned_arg_indices = array<i32: 0>} : (i32) -> ()
  return
}

// -----

// A DFB index entry references a typed DFB operand without declaring access.
// CHECK-LABEL: func.func @dfb_index_template_arg
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2}
// CHECK-NEXT: ttl.opaque_call "inspect" template_args [#ttl.external_template_arg<dfb_index, 0>] template_dfbs(%[[CB]] : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) (%[[CB]]) {header = "inspect.hpp"}
func.func @dfb_index_template_arg() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.opaque_call "inspect" template_args [#ttl.external_template_arg<dfb_index, 0>] template_dfbs(%cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) (%cb) {header = "inspect.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
  return
}

// -----

// A descriptor entry makes its referenced DFB a direct storage dependency.
// CHECK-LABEL: func.func @dfb_descriptor_template_arg
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = 3, block_count = 2}
// CHECK-NEXT: ttl.opaque_call "describe" template_args [#ttl.external_template_arg<dfb_descriptor, 0>] template_dfbs(%[[CB]] : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>) () {header = "describe.hpp"}
func.func @dfb_descriptor_template_arg() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  ttl.opaque_call "describe" template_args [#ttl.external_template_arg<dfb_descriptor, 0>] template_dfbs(%cb : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>) () {header = "describe.hpp"} : () -> ()
  return
}
