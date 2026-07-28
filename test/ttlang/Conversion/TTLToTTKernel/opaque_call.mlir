// Verify opaque_call lowering from TTL to TTKernel.
// Checks template arg forwarding, DFB-to-CB conversion for func_args, and
// get_dfb_id lowering from ttl to ttkernel dialect.
// RUN: ttlang-opt --convert-ttl-to-ttkernel --split-input-file %s | FileCheck %s

// Void call with no args lowers directly.
// CHECK-LABEL: func.func @void_call_no_args
// CHECK: ttkernel.opaque_call "noop" () {header = "noop.hpp"} : () -> ()
func.func @void_call_no_args() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "noop" () {header = "noop.hpp"} : () -> ()
  return
}

// -----

// Constant template args are forwarded as-is.
// CHECK-LABEL: func.func @call_with_template_args
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : i32
// CHECK-DAG: %[[C7:.*]] = arith.constant 7 : i32
// CHECK: ttkernel.opaque_call "compute" template_args(%[[C3]], %[[C7]]) () {header = "compute.hpp"} : () -> ()
func.func @call_with_template_args() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c3 = arith.constant 3 : i32
  %c7 = arith.constant 7 : i32
  ttl.opaque_call "compute" template_args(%c3, %c7) () {header = "compute.hpp"} : () -> ()
  return
}

// -----

// DFB template arg: ttl.get_dfb_id lowers to ttkernel.get_dfb_id.
// CHECK-LABEL: func.func @call_with_dfb_template_arg
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val
// CHECK-NEXT: %[[ID:.*]] = ttkernel.get_dfb_id %[[CB]]
// CHECK-NEXT: ttkernel.opaque_call "drain" template_args(%[[ID]]) () {header = "drain.hpp"} : () -> ()
func.func @call_with_dfb_template_arg() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %id = ttl.get_dfb_id %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
  ttl.opaque_call "drain" template_args(%id) () {header = "drain.hpp"} : () -> ()
  return
}

// -----

// DFB func_arg is lowered to get_compile_time_arg_val (i32 CB index).
// CHECK-LABEL: func.func @call_with_dfb_func_arg
// CHECK: %[[CB_IDX:.*]] = ttkernel.get_compile_time_arg_val(1) : () -> i32
// CHECK-NEXT: ttkernel.opaque_call "use_cb" (%[[CB_IDX]]) {header = "use_cb.hpp"} : (i32) -> ()
func.func @call_with_dfb_func_arg() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.opaque_call "use_cb" (%cb) {header = "use_cb.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
  return
}
