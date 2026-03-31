// RUN: ttlang-opt %s -split-input-file -verify-diagnostics
// Negative tests for ttl.reduce verifier.

// -----

// Rank-3 input
func.func @reduce_rank3(
    %a: tensor<1x2x3x!ttcore.tile<32x32, bf16>>,
    %s: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x2x1x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{input must be rank 2, got rank 3}}
  %r = ttl.reduce %a, %s 0 : i32 [2] : (tensor<1x2x3x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x2x1x!ttcore.tile<32x32, bf16>>
  return %r : tensor<1x2x1x!ttcore.tile<32x32, bf16>>
}

// -----

// Empty dims
func.func @reduce_empty_dims(
    %a: tensor<2x2x!ttcore.tile<32x32, bf16>>,
    %s: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{dims must be non-empty}}
  %r = ttl.reduce %a, %s 0 : i32 [] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  return %r : tensor<2x2x!ttcore.tile<32x32, bf16>>
}

// -----

// Out-of-range dim
func.func @reduce_dim_out_of_range(
    %a: tensor<2x2x!ttcore.tile<32x32, bf16>>,
    %s: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<2x1x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{dim 5 is out of range for rank 2}}
  %r = ttl.reduce %a, %s 0 : i32 [5] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  return %r : tensor<2x1x!ttcore.tile<32x32, bf16>>
}

// -----

// Wrong result shape: reducing dim 0 should give (1, N) not (N, N)
func.func @reduce_wrong_result_shape(
    %a: tensor<4x3x!ttcore.tile<32x32, bf16>>,
    %s: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<4x3x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{result dim 0 is 4 but expected 1}}
  %r = ttl.reduce %a, %s 0 : i32 [0] : (tensor<4x3x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<4x3x!ttcore.tile<32x32, bf16>>
  return %r : tensor<4x3x!ttcore.tile<32x32, bf16>>
}

// -----

// Mismatched element types
func.func @reduce_element_type_mismatch(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %s: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x2x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{result element type '!ttcore.tile<32x32, bf16>' must match input element type '!ttcore.tile<32x32, f32>'}}
  %r = ttl.reduce %a, %s 0 : i32 [0] : (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  return %r : tensor<1x2x!ttcore.tile<32x32, bf16>>
}

// -----

// Duplicate dims
func.func @reduce_duplicate_dims(
    %a: tensor<2x2x!ttcore.tile<32x32, bf16>>,
    %s: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x2x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{duplicate dim 0}}
  %r = ttl.reduce %a, %s 0 : i32 [0, 0] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  return %r : tensor<1x2x!ttcore.tile<32x32, bf16>>
}

// -----

// Scaler must be (1, 1)
func.func @reduce_scaler_wrong_shape(
    %a: tensor<2x2x!ttcore.tile<32x32, bf16>>,
    %s: tensor<3x3x!ttcore.tile<32x32, bf16>>) -> tensor<1x2x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{scaler dim 0 is 3 but must be 1}}
  %r = ttl.reduce %a, %s 0 : i32 [0] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<3x3x!ttcore.tile<32x32, bf16>>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  return %r : tensor<1x2x!ttcore.tile<32x32, bf16>>
}

// -----

// Scaler must be rank 2
func.func @reduce_scaler_rank1(
    %a: tensor<2x2x!ttcore.tile<32x32, bf16>>,
    %s: tensor<1x!ttcore.tile<32x32, bf16>>) -> tensor<1x2x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{scaler must be rank 2, got rank 1}}
  %r = ttl.reduce %a, %s 0 : i32 [0] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x!ttcore.tile<32x32, bf16>>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  return %r : tensor<1x2x!ttcore.tile<32x32, bf16>>
}
