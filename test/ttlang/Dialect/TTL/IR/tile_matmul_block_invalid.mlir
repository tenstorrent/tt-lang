// RUN: ttlang-opt %s --split-input-file --verify-diagnostics
// Verifies target-independent tile_matmul_block type relations.

// Physical tile K dimensions must match.
func.func @tile_matmul_k_mismatch(
    %lhs: !ttcore.tile<4x32, bf16>,
    %rhs: !ttcore.tile<16x32, bf16>) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{tile K dimension mismatch: lhs tile width 32 does not match rhs tile height 16}}
  %result = ttl.tile_matmul_block %lhs, %rhs into dst[%c0]
      : !ttcore.tile<4x32, bf16>, !ttcore.tile<16x32, bf16>
        -> !ttcore.tile<4x32, bf16>
  return
}

// -----

// The result dimensions derive from the lhs height and rhs width.
func.func @tile_matmul_result_mismatch(
    %lhs: !ttcore.tile<4x32, bf16>,
    %rhs: !ttcore.tile<32x32, bf16>) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{result tile dimensions 8x32 do not match expected 4x32}}
  %result = ttl.tile_matmul_block %lhs, %rhs into dst[%c0]
      : !ttcore.tile<4x32, bf16>, !ttcore.tile<32x32, bf16>
        -> !ttcore.tile<8x32, bf16>
  return
}

// -----

// The optional accumulator must have the result tile type.
func.func @tile_matmul_accumulator_mismatch(
    %lhs: !ttcore.tile<4x32, bf16>,
    %rhs: !ttcore.tile<32x32, bf16>,
    %accumulator: !ttcore.tile<8x32, bf16>) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{accumulator tile type '!ttcore.tile<8x32, bf16>' must match result tile type '!ttcore.tile<4x32, bf16>'}}
  %result = ttl.tile_matmul_block %lhs, %rhs, %accumulator into dst[%c0]
      : !ttcore.tile<4x32, bf16>, !ttcore.tile<32x32, bf16>,
        !ttcore.tile<8x32, bf16> -> !ttcore.tile<4x32, bf16>
  return
}

// -----

// Ranked tensors must contain tile elements.
func.func @tile_matmul_scalar_tensor(
    %lhs: tensor<1x1xbf16>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{lhs must be a tile or tensor of tiles, got 'tensor<1x1xbf16>'}}
  %result = ttl.tile_matmul_block %lhs, %rhs into dst[%c0]
      : tensor<1x1xbf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> !ttcore.tile<32x32, bf16>
  return
}

// -----

// BF16 lhs is required for mixed block-float matmul.
func.func @tile_matmul_unsupported_mixed_types(
    %lhs: !ttcore.tile<32x32, f32>,
    %rhs: !ttcore.tile<32x32, bfp_bf4>) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{unsupported matmul element data type combination: lhs has !ttcore.tile<32x32, f32>, rhs has !ttcore.tile<32x32, bfp_bf4>, and result has !ttcore.tile<32x32, f32>}}
  %result = ttl.tile_matmul_block %lhs, %rhs into dst[%c0]
      : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, bfp_bf4>
        -> !ttcore.tile<32x32, f32>
  return
}

// -----

// BF16 lhs is required for mixed BFP8_B matmul.
func.func @tile_matmul_unsupported_bfp8_types(
    %lhs: !ttcore.tile<32x32, f32>,
    %rhs: !ttcore.tile<32x32, bfp_bf8>) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{unsupported matmul element data type combination: lhs has !ttcore.tile<32x32, f32>, rhs has !ttcore.tile<32x32, bfp_bf8>, and result has !ttcore.tile<32x32, f32>}}
  %result = ttl.tile_matmul_block %lhs, %rhs into dst[%c0]
      : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, bfp_bf8>
        -> !ttcore.tile<32x32, f32>
  return
}
