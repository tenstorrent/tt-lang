// Round-trip verification of ttkernel.opaque_call with template arg SSA values.
// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// -----
// Test: void call with no template args
// CHECK-LABEL: func.func @void_no_template_args
// CHECK: ttkernel.opaque_call "my_func" () {header = "my_header.hpp"} : () -> ()
func.func @void_no_template_args() {
  ttkernel.opaque_call "my_func" () {header = "my_header.hpp"} : () -> ()
  return
}

// -----
// Test: call with constant template args
// CHECK-LABEL: func.func @const_template_args
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[C10:.*]] = arith.constant 10 : i32
// CHECK: ttkernel.opaque_call "calc" template_args(%[[C5]], %[[C10]]) () {header = "calc.hpp"} : () -> ()
func.func @const_template_args() {
  %c5 = arith.constant 5 : i32
  %c10 = arith.constant 10 : i32
  ttkernel.opaque_call "calc" template_args(%c5, %c10) () {header = "calc.hpp"} : () -> ()
  return
}

// -----
// Test: call with get_dfb_id template arg
// CHECK-LABEL: func.func @dfb_template_arg
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(3)
// CHECK-NEXT: %[[ID:.*]] = ttkernel.get_dfb_id %[[CB]]
// CHECK-NEXT: ttkernel.opaque_call "flush" template_args(%[[ID]]) () {header = "flush.hpp"} : () -> ()
func.func @dfb_template_arg() {
  %cb = ttkernel.get_compile_time_arg_val(3) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %id = ttkernel.get_dfb_id %cb : <1, !ttcore.tile<32x32, bf16>>
  ttkernel.opaque_call "flush" template_args(%id) () {header = "flush.hpp"} : () -> ()
  return
}
