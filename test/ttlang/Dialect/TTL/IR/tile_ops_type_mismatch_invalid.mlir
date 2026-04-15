// Verify that tile binary/unary ops reject mismatched operand/result types.
// The AllTypesMatch trait enforces that tile operands and result are the same
// type; the index-typed dst_index is excluded from the check.

// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// Binary: lhs/rhs type mismatch.
func.func @tile_add_type_mismatch(%a: !ttcore.tile<32x32, bf16>,
                                   %b: !ttcore.tile<32x32, f32>) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{failed to verify that all of {lhs, rhs, result} have same type}}
  %0 = "ttl.tile_add"(%a, %b, %c0) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, f32>, index) -> !ttcore.tile<32x32, bf16>
  return
}

// -----

// Binary: operand/result type mismatch.
func.func @tile_mul_result_mismatch(%a: !ttcore.tile<32x32, bf16>,
                                     %b: !ttcore.tile<32x32, bf16>) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{failed to verify that all of {lhs, rhs, result} have same type}}
  %0 = "ttl.tile_mul"(%a, %b, %c0) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, index) -> !ttcore.tile<32x32, f32>
  return
}

// -----

// Unary: input/result type mismatch.
func.func @tile_exp_type_mismatch(%a: !ttcore.tile<32x32, bf16>) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{failed to verify that all of {input, result} have same type}}
  %0 = "ttl.tile_exp"(%a, %c0) : (!ttcore.tile<32x32, bf16>, index) -> !ttcore.tile<32x32, f32>
  return
}

// -----

// CopyDst: src_tile/result type mismatch.
func.func @copy_dst_type_mismatch(%a: !ttcore.tile<32x32, bf16>) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{failed to verify that all of {src_tile, result} have same type}}
  %0 = "ttl.copy_dst"(%a, %c0) : (!ttcore.tile<32x32, bf16>, index) -> !ttcore.tile<32x32, f32>
  return
}
