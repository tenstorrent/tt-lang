// RUN: ttlang-opt %s -split-input-file -verify-diagnostics
// Negative tests for ttl.transpose verifier.

// -----

// Rank-3 input
func.func @transpose_rank3(
    %a: tensor<1x2x3x!ttcore.tile<32x32, bf16>>) -> tensor<3x2x1x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{input must be rank 2, got rank 3}}
  %r = ttl.transpose %a : tensor<1x2x3x!ttcore.tile<32x32, bf16>> -> tensor<3x2x1x!ttcore.tile<32x32, bf16>>
  return %r : tensor<3x2x1x!ttcore.tile<32x32, bf16>>
}

// -----

// Result shape not transposed: (3, 4) input should give (4, 3) not (3, 4)
func.func @transpose_wrong_shape(
    %a: tensor<3x4x!ttcore.tile<32x32, bf16>>) -> tensor<3x4x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{result shape [3, 4] must be the transpose of input shape [3, 4]}}
  %r = ttl.transpose %a : tensor<3x4x!ttcore.tile<32x32, bf16>> -> tensor<3x4x!ttcore.tile<32x32, bf16>>
  return %r : tensor<3x4x!ttcore.tile<32x32, bf16>>
}

// -----

// Mismatched element types
func.func @transpose_element_type_mismatch(
    %a: tensor<2x3x!ttcore.tile<32x32, f32>>) -> tensor<3x2x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{result element type '!ttcore.tile<32x32, bf16>' must match input element type '!ttcore.tile<32x32, f32>'}}
  %r = ttl.transpose %a : tensor<2x3x!ttcore.tile<32x32, f32>> -> tensor<3x2x!ttcore.tile<32x32, bf16>>
  return %r : tensor<3x2x!ttcore.tile<32x32, bf16>>
}
