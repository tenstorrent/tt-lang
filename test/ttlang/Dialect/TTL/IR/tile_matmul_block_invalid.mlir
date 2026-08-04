// RUN: ttlang-opt %s --split-input-file --verify-diagnostics
// Verifies tile_matmul_block's physical tile relations and LLK restrictions.

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

// A 16x16 lhs is not implemented by the current matmul LLKs.
func.func @tile_matmul_lhs_16x16(
    %lhs: !ttcore.tile<16x16, bf16>,
    %rhs: !ttcore.tile<16x32, bf16>) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{matmul lhs tile dimensions 16x16 are not implemented by the current compute LLKs}}
  %result = ttl.tile_matmul_block %lhs, %rhs into dst[%c0]
      : !ttcore.tile<16x16, bf16>, !ttcore.tile<16x32, bf16>
        -> !ttcore.tile<16x32, bf16>
  return
}

// -----

// A 16x16 rhs is a valid physical tile but is not implemented by matmul.
func.func @tile_matmul_rhs_16x16(
    %lhs: !ttcore.tile<32x16, bf16>,
    %rhs: !ttcore.tile<16x16, bf16>) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{matmul rhs tile dimensions 16x16 are not implemented by the current compute LLKs; supported rhs dimensions are 16x32, 32x16, and 32x32}}
  %result = ttl.tile_matmul_block %lhs, %rhs into dst[%c0]
      : !ttcore.tile<32x16, bf16>, !ttcore.tile<16x16, bf16>
        -> !ttcore.tile<32x16, bf16>
  return
}

// -----

// Transpose is not implemented for a 32x16 rhs tile.
func.func @tile_matmul_transpose_rhs_32x16(
    %lhs: !ttcore.tile<32x16, bf16>,
    %rhs: !ttcore.tile<32x16, bf16>) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{matmul rhs tile dimensions 32x16 do not support transpose_rhs in the current compute LLKs}}
  %result = ttl.tile_matmul_block %lhs, %rhs into dst[%c0] {transpose_rhs}
      : !ttcore.tile<32x16, bf16>, !ttcore.tile<32x16, bf16>
        -> !ttcore.tile<32x32, bf16>
  return
}

// -----

// Transposing a 16x32 rhs does not compute the second lhs face row for a
// 32x32 lhs.
func.func @tile_matmul_transpose_lhs_32x32_rhs_16x32(
    %lhs: !ttcore.tile<32x32, bf16>,
    %rhs: !ttcore.tile<16x32, bf16>) {
  %c0 = arith.constant 0 : index
  // expected-error @below {{matmul tile dimensions lhs 32x32 and rhs 16x32 do not support transpose_rhs in the current compute LLKs}}
  %result = ttl.tile_matmul_block %lhs, %rhs into dst[%c0] {transpose_rhs}
      : !ttcore.tile<32x32, bf16>, !ttcore.tile<16x32, bf16>
        -> !ttcore.tile<32x16, bf16>
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
