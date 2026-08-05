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
