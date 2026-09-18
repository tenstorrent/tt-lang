// Negative tests for TTKernel operation verification.
// Verifies that operations requiring a kernel function reject module scope.

// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// -----

// Test: dataflow buffer queue ops must appear inside a kernel function.
%cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
%count = arith.constant 1 : i32
// expected-error @below {{'ttkernel.cb_push_back' op CBPushBackOp must be inside a kernel function}}
ttkernel.cb_push_back(%cb, %count) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, i32) -> ()

// -----

// Test: constant lookup tables must contain at least one value.
func.func @empty_constant_table(%index: index) -> index {
  // expected-error @below {{'ttkernel.experimental.constant_table_lookup' op requires at least one table value}}
  %value = ttkernel.experimental.constant_table_lookup %index, [] : index
  return %value : index
}

// -----

// Test: constant lookup tables contain only non-negative values.
func.func @negative_constant_table_value(%index: index) -> index {
  // expected-error @below {{'ttkernel.experimental.constant_table_lookup' op requires non-negative table values}}
  %value = ttkernel.experimental.constant_table_lookup %index, [0, -1] : index
  return %value : index
}

// -----

// Test: a constant lookup index must identify an element in the table.
func.func @constant_table_index_out_of_bounds() -> index {
  %index = arith.constant 2 : index
  // expected-error @below {{'ttkernel.experimental.constant_table_lookup' op constant index 2 is outside the table bounds [0, 2)}}
  %value = ttkernel.experimental.constant_table_lookup %index, [10, 20] : index
  return %value : index
}
