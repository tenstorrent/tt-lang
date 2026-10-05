// Verify ttkernel.opaque_call round trips ordered static arguments and DFB
// resource requirements.
// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// A void call omits the optional template argument list.
// CHECK-LABEL: func.func @void_no_template_args
// CHECK: ttkernel.opaque_call "my_func"() {header = "my_header.hpp"} : () -> ()
func.func @void_no_template_args() {
  ttkernel.opaque_call "my_func" () {header = "my_header.hpp"} : () -> ()
  return
}

// -----

// Static argument kinds and source order remain explicit in TTKernel IR.
// CHECK-LABEL: func.func @typed_template_args
// CHECK: ttkernel.opaque_call "describe" template_args [-5 : si32, true, 4294967295 : ui32, #ttkernel.dfb_descriptor<3, 2, 4, 4096>]() {dfb_resource_indices = array<i32: 3>, header = "describe.hpp"} : () -> ()
func.func @typed_template_args() {
  ttkernel.opaque_call "describe" template_args [-5 : si32, true, 4294967295 : ui32, #ttkernel.dfb_descriptor<3, 2, 4, 4096>] () {dfb_resource_indices = array<i32: 3>, header = "describe.hpp"} : () -> ()
  return
}

// -----

// Signless storage may require an explicit unsigned C++ call expression.
// CHECK-LABEL: func.func @unsigned_func_arg
// CHECK: ttkernel.opaque_call "use_address"(%arg0) {header = "address.hpp", unsigned_arg_indices = array<i32: 0>} : (i32) -> ()
func.func @unsigned_func_arg(%arg0: i32) {
  ttkernel.opaque_call "use_address" (%arg0) {header = "address.hpp", unsigned_arg_indices = array<i32: 0>} : (i32) -> ()
  return
}
