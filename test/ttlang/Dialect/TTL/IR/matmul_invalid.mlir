// RUN: ttlang-opt %s -split-input-file -verify-diagnostics
// Negative tests for ttl.matmul verifier.

// Test: K dimension mismatch
func.func @matmul_k_mismatch(
    %a: tensor<2x3x!ttcore.tile<32x32, bf16>>,
    %b: tensor<4x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{K dimension mismatch: lhs has 3 columns but rhs has 4 rows}}
  %r = ttl.matmul %a, %b : tensor<2x3x!ttcore.tile<32x32, bf16>>, tensor<4x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  return %r : tensor<2x2x!ttcore.tile<32x32, bf16>>
}

// -----

// Test: Wrong result shape
func.func @matmul_bad_result_shape(
    %a: tensor<2x3x!ttcore.tile<32x32, bf16>>,
    %b: tensor<3x4x!ttcore.tile<32x32, bf16>>) -> tensor<2x3x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{result shape [2, 3] does not match expected [2, 4]}}
  %r = ttl.matmul %a, %b : tensor<2x3x!ttcore.tile<32x32, bf16>>, tensor<3x4x!ttcore.tile<32x32, bf16>> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  return %r : tensor<2x3x!ttcore.tile<32x32, bf16>>
}

// -----

// Test: lhs not rank 2
func.func @matmul_lhs_rank3(
    %a: tensor<1x2x3x!ttcore.tile<32x32, bf16>>,
    %b: tensor<3x4x!ttcore.tile<32x32, bf16>>) -> tensor<2x4x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{lhs must be rank 2, got rank 3}}
  %r = ttl.matmul %a, %b : tensor<1x2x3x!ttcore.tile<32x32, bf16>>, tensor<3x4x!ttcore.tile<32x32, bf16>> -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  return %r : tensor<2x4x!ttcore.tile<32x32, bf16>>
}

// -----

// Test: Element type mismatch between inputs
func.func @matmul_element_mismatch(
    %a: tensor<2x3x!ttcore.tile<32x32, bf16>>,
    %b: tensor<3x4x!ttcore.tile<32x32, f32>>) -> tensor<2x4x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{element type mismatch}}
  %r = ttl.matmul %a, %b : tensor<2x3x!ttcore.tile<32x32, bf16>>, tensor<3x4x!ttcore.tile<32x32, f32>> -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  return %r : tensor<2x4x!ttcore.tile<32x32, bf16>>
}

// -----

// Test: rhs not rank 2
func.func @matmul_rhs_rank1(
    %a: tensor<2x3x!ttcore.tile<32x32, bf16>>,
    %b: tensor<3x!ttcore.tile<32x32, bf16>>) -> tensor<2x3x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{rhs must be rank 2, got rank 1}}
  %r = ttl.matmul %a, %b : tensor<2x3x!ttcore.tile<32x32, bf16>>, tensor<3x!ttcore.tile<32x32, bf16>> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  return %r : tensor<2x3x!ttcore.tile<32x32, bf16>>
}

// -----

// Test: Dynamic shape on lhs
func.func @matmul_dynamic_lhs(
    %a: tensor<?x3x!ttcore.tile<32x32, bf16>>,
    %b: tensor<3x4x!ttcore.tile<32x32, bf16>>) -> tensor<2x4x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{lhs must have static shape}}
  %r = ttl.matmul %a, %b : tensor<?x3x!ttcore.tile<32x32, bf16>>, tensor<3x4x!ttcore.tile<32x32, bf16>> -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  return %r : tensor<2x4x!ttcore.tile<32x32, bf16>>
}

// -----

// Test: Result element type mismatch (inputs bf16 but result f32)
func.func @matmul_result_element_mismatch(
    %a: tensor<2x3x!ttcore.tile<32x32, bf16>>,
    %b: tensor<3x4x!ttcore.tile<32x32, bf16>>) -> tensor<2x4x!ttcore.tile<32x32, f32>> {
  // expected-error @below {{result element type '!ttcore.tile<32x32, f32>' must match input element type '!ttcore.tile<32x32, bf16>'}}
  %r = ttl.matmul %a, %b : tensor<2x3x!ttcore.tile<32x32, bf16>>, tensor<3x4x!ttcore.tile<32x32, bf16>> -> tensor<2x4x!ttcore.tile<32x32, f32>>
  return %r : tensor<2x4x!ttcore.tile<32x32, f32>>
}
