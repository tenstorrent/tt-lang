// Verifier tests for tensor-level TTL elementwise ops.
// RUN: ttlang-opt --verify-diagnostics --split-input-file %s

// -----

// Binary elementwise operands must have the same type.
func.func @add_operand_type_mismatch(
    %lhs: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{requires the same type for all operands and results}}
  %0 = ttl.add %lhs, %rhs : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  return %0 : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// Unary elementwise results must preserve the complete input tensor type.
func.func @unary_result_type_mismatch(
    %input: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<16x32, bf16>> {
  // expected-error @below {{requires the same type for all operands and results}}
  %0 = ttl.exp %input : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<16x32, bf16>>
  return %0 : tensor<1x1x!ttcore.tile<16x32, bf16>>
}

// -----

// Binary elementwise results must preserve the complete operand tensor type.
func.func @binary_result_type_mismatch(
    %lhs: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<16x32, bf16>> {
  // expected-error @below {{requires the same type for all operands and results}}
  %0 = ttl.add %lhs, %rhs : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<16x32, bf16>>
  return %0 : tensor<1x1x!ttcore.tile<16x32, bf16>>
}
