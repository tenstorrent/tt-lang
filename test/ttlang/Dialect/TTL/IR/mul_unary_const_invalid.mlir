// Verifier tests for ttl.mul_unary_const and ttl.tile_mul_unary_const.
// Both ops require input and result types to match exactly (AllTypesMatch).
//
// RUN: ttlang-opt --verify-diagnostics --split-input-file %s

// Tensor-level: input and result tensor types must match.
func.func @mul_unary_const_shape_mismatch(
    %arg0: tensor<2x2x!ttcore.tile<32x32, bf16>>)
    -> tensor<3x3x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{all of {input, result} have same type}}
  %0 = ttl.mul_unary_const %arg0, 1.000000e+00
      : tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
  return %0 : tensor<3x3x!ttcore.tile<32x32, bf16>>
}

// -----

// Tensor-level: input and result tile element dtypes must match.
func.func @mul_unary_const_dtype_mismatch(
    %arg0: tensor<2x2x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // expected-error @below {{all of {input, result} have same type}}
  %0 = ttl.mul_unary_const %arg0, 1.000000e+00
      : tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Tile-level: input and result tile types must match.
func.func @tile_mul_unary_const_dtype_mismatch(%a: !ttcore.tile<32x32, bf16>)
    -> !ttcore.tile<32x32, f32> {
  %c0 = arith.constant 0 : index
  // expected-error @below {{all of {input, result} have same type}}
  %0 = ttl.tile_mul_unary_const %a, 1.000000e+00 into dst[%c0]
       : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
  return %0 : !ttcore.tile<32x32, f32>
}
