// Verifier tests for tensor-level TTL elementwise ops.
// RUN: ttlang-opt --verify-diagnostics --split-input-file %s

// -----

// Binary elementwise operands must have the same type.
func.func @add_operand_type_mismatch(
    %lhs: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{requires all operands to have the same type}}
  %0 = ttl.add %lhs, %rhs : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  return %0 : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
