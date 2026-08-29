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

// Test: unsupported mixed input data types.
func.func @matmul_element_mismatch(
    %a: tensor<2x3x!ttcore.tile<32x32, bf16>>,
    %b: tensor<3x4x!ttcore.tile<32x32, f32>>) -> tensor<2x4x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{unsupported matmul element data type combination: lhs has !ttcore.tile<32x32, bf16>, rhs has !ttcore.tile<32x32, f32>, and result has !ttcore.tile<32x32, bf16>}}
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

// Test: result data type must match the supported input combination.
func.func @matmul_result_element_mismatch(
    %a: tensor<2x3x!ttcore.tile<32x32, bf16>>,
    %b: tensor<3x4x!ttcore.tile<32x32, bf16>>) -> tensor<2x4x!ttcore.tile<32x32, f32>> {
  // expected-error @below {{unsupported matmul element data type combination: lhs has !ttcore.tile<32x32, bf16>, rhs has !ttcore.tile<32x32, bf16>, and result has !ttcore.tile<32x32, f32>}}
  %r = ttl.matmul %a, %b : tensor<2x3x!ttcore.tile<32x32, bf16>>, tensor<3x4x!ttcore.tile<32x32, bf16>> -> tensor<2x4x!ttcore.tile<32x32, f32>>
  return %r : tensor<2x4x!ttcore.tile<32x32, f32>>
}

// -----

// Mixed-format matmul rejects a transposed BFP4_B rhs.
func.func @matmul_transposed_bfp4_rhs(
    %a: tensor<2x3x!ttcore.tile<32x32, bf16>>,
    %b: tensor<4x3x!ttcore.tile<32x32, bfp_bf4>>)
    -> tensor<2x4x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{unsupported matmul element data type combination: lhs has !ttcore.tile<32x32, bf16>, rhs has !ttcore.tile<32x32, bfp_bf4>, and result has !ttcore.tile<32x32, bf16>}}
  %r = ttl.matmul %a, %b {transpose_rhs}
      : tensor<2x3x!ttcore.tile<32x32, bf16>>,
        tensor<4x3x!ttcore.tile<32x32, bfp_bf4>>
        -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  return %r : tensor<2x4x!ttcore.tile<32x32, bf16>>
}

// -----

// Mixed-format matmul rejects a transposed BFP8_B rhs.
func.func @matmul_transposed_bfp8_rhs(
    %a: tensor<2x3x!ttcore.tile<32x32, bf16>>,
    %b: tensor<4x3x!ttcore.tile<32x32, bfp_bf8>>)
    -> tensor<2x4x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{unsupported matmul element data type combination: lhs has !ttcore.tile<32x32, bf16>, rhs has !ttcore.tile<32x32, bfp_bf8>, and result has !ttcore.tile<32x32, bf16>}}
  %r = ttl.matmul %a, %b {transpose_rhs}
      : tensor<2x3x!ttcore.tile<32x32, bf16>>,
        tensor<4x3x!ttcore.tile<32x32, bfp_bf8>>
        -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  return %r : tensor<2x4x!ttcore.tile<32x32, bf16>>
}

// -----

// Mixed BFP4_B matmul requires a BF16 result.
func.func @matmul_bfp4_result_type(
    %a: tensor<2x3x!ttcore.tile<32x32, bf16>>,
    %b: tensor<3x4x!ttcore.tile<32x32, bfp_bf4>>)
    -> tensor<2x4x!ttcore.tile<32x32, bfp_bf4>> {
  // expected-error @below {{unsupported matmul element data type combination: lhs has !ttcore.tile<32x32, bf16>, rhs has !ttcore.tile<32x32, bfp_bf4>, and result has !ttcore.tile<32x32, bfp_bf4>}}
  %r = ttl.matmul %a, %b
      : tensor<2x3x!ttcore.tile<32x32, bf16>>,
        tensor<3x4x!ttcore.tile<32x32, bfp_bf4>>
        -> tensor<2x4x!ttcore.tile<32x32, bfp_bf4>>
  return %r : tensor<2x4x!ttcore.tile<32x32, bfp_bf4>>
}

// -----

// Mixed BFP8_B matmul requires a BF16 result.
func.func @matmul_bfp8_result_type(
    %a: tensor<2x3x!ttcore.tile<32x32, bf16>>,
    %b: tensor<3x4x!ttcore.tile<32x32, bfp_bf8>>)
    -> tensor<2x4x!ttcore.tile<32x32, bfp_bf8>> {
  // expected-error @below {{unsupported matmul element data type combination: lhs has !ttcore.tile<32x32, bf16>, rhs has !ttcore.tile<32x32, bfp_bf8>, and result has !ttcore.tile<32x32, bfp_bf8>}}
  %r = ttl.matmul %a, %b
      : tensor<2x3x!ttcore.tile<32x32, bf16>>,
        tensor<3x4x!ttcore.tile<32x32, bfp_bf8>>
        -> tensor<2x4x!ttcore.tile<32x32, bfp_bf8>>
  return %r : tensor<2x4x!ttcore.tile<32x32, bfp_bf8>>
}

// -----

// Test: transpose_rhs K mismatch. RHS is [N, K]=[4, 5] so its K (5) does not
// match lhs K (3).
func.func @matmul_transpose_k_mismatch(
    %a: tensor<2x3x!ttcore.tile<32x32, bf16>>,
    %b: tensor<4x5x!ttcore.tile<32x32, bf16>>) -> tensor<2x4x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{K dimension mismatch: lhs has 3 columns but rhs has 5 columns}}
  %r = ttl.matmul %a, %b {transpose_rhs} : tensor<2x3x!ttcore.tile<32x32, bf16>>, tensor<4x5x!ttcore.tile<32x32, bf16>> -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  return %r : tensor<2x4x!ttcore.tile<32x32, bf16>>
}

// -----

// Test: transpose_rhs wrong result shape. RHS is [N, K]=[4, 3] so the result
// must be [M, N]=[2, 4], not [2, 3].
func.func @matmul_transpose_bad_result(
    %a: tensor<2x3x!ttcore.tile<32x32, bf16>>,
    %b: tensor<4x3x!ttcore.tile<32x32, bf16>>) -> tensor<2x3x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{result shape [2, 3] does not match expected [2, 4]}}
  %r = ttl.matmul %a, %b {transpose_rhs} : tensor<2x3x!ttcore.tile<32x32, bf16>>, tensor<4x3x!ttcore.tile<32x32, bf16>> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  return %r : tensor<2x3x!ttcore.tile<32x32, bf16>>
}

// -----

// Test: physical tile K dimensions must match independently of tensor K.
func.func @matmul_tile_k_mismatch(
    %a: tensor<1x2x!ttcore.tile<4x32, bf16>>,
    %b: tensor<2x1x!ttcore.tile<16x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<4x32, bf16>> {
  // expected-error @below {{tile K dimension mismatch: lhs tile width 32 does not match rhs tile height 16}}
  %r = ttl.matmul %a, %b
      : tensor<1x2x!ttcore.tile<4x32, bf16>>,
        tensor<2x1x!ttcore.tile<16x32, bf16>>
        -> tensor<1x1x!ttcore.tile<4x32, bf16>>
  return %r : tensor<1x1x!ttcore.tile<4x32, bf16>>
}

// -----

// Test: result physical dimensions derive from the two operands.
func.func @matmul_tile_result_mismatch(
    %a: tensor<1x2x!ttcore.tile<4x32, bf16>>,
    %b: tensor<2x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<8x32, bf16>> {
  // expected-error @below {{result tile dimensions 8x32 do not match expected 4x32}}
  %r = ttl.matmul %a, %b
      : tensor<1x2x!ttcore.tile<4x32, bf16>>,
        tensor<2x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<8x32, bf16>>
  return %r : tensor<1x1x!ttcore.tile<8x32, bf16>>
}

// -----

// Test: transpose_rhs contracts the physical widths of both operands.
func.func @matmul_transpose_tile_k_mismatch(
    %a: tensor<1x2x!ttcore.tile<32x16, bf16>>,
    %b: tensor<1x2x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{tile K dimension mismatch: lhs tile width 16 does not match rhs tile width 32}}
  %r = ttl.matmul %a, %b {transpose_rhs}
      : tensor<1x2x!ttcore.tile<32x16, bf16>>,
        tensor<1x2x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  return %r : tensor<1x1x!ttcore.tile<32x32, bf16>>
}

// -----

// Test: transpose_rhs result width is the rhs physical tile height.
func.func @matmul_transpose_tile_result_mismatch(
    %a: tensor<1x2x!ttcore.tile<4x32, bf16>>,
    %b: tensor<1x2x!ttcore.tile<16x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<4x32, bf16>> {
  // expected-error @below {{result tile dimensions 4x32 do not match expected 4x16}}
  %r = ttl.matmul %a, %b {transpose_rhs}
      : tensor<1x2x!ttcore.tile<4x32, bf16>>,
        tensor<1x2x!ttcore.tile<16x32, bf16>>
        -> tensor<1x1x!ttcore.tile<4x32, bf16>>
  return %r : tensor<1x1x!ttcore.tile<4x32, bf16>>
}
