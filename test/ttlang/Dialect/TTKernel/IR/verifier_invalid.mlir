// Negative tests for TTKernel operation verification.
// Verifies that operations requiring a kernel function reject module scope.

// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// -----

// Test: dataflow buffer queue ops must appear inside a kernel function.
%cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
%count = arith.constant 1 : i32
// expected-error @below {{'ttkernel.cb_push_back' op CBPushBackOp must be inside a kernel function}}
ttkernel.cb_push_back(%cb, %count) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()
