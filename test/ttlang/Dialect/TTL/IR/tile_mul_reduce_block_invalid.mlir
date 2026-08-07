// Verifies structural, type, and scale invariants of the internal
// multiply-reduction block operation.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// The target schedule supports bf16 tiles only.
module {
  func.func @unsupported_dtype(
      %lhs: !ttcore.tile<32x32, f32>,
      %output: !ttcore.tile<32x32, f32>) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{'ttl.tile_mul_reduce_block' op supports bf16 tiles only}}
    %result = ttl.tile_mul_reduce_block
        %lhs, %lhs, %output scale = 1.000000e+00 into dst[%c0]
        : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>,
          !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    return
  }
}

// -----

// Tensor inputs must have identical static rank-2 types.
module {
  func.func @mismatched_inputs(
      %lhs: tensor<1x2x!ttcore.tile<32x32, bf16>>,
      %rhs: tensor<1x3x!ttcore.tile<32x32, bf16>>,
      %output: tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{'ttl.tile_mul_reduce_block' op lhs and rhs tensor types must match}}
    %result = ttl.tile_mul_reduce_block
        %lhs, %rhs, %output scale = 1.000000e+00 into dst[%c0]
        : tensor<1x2x!ttcore.tile<32x32, bf16>>,
          tensor<1x3x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

// Full scalar reduction publishes one logical output tile.
module {
  func.func @nonscalar_output(
      %input: tensor<1x2x!ttcore.tile<32x32, bf16>>,
      %output: tensor<1x2x!ttcore.tile<32x32, bf16>>) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{'ttl.tile_mul_reduce_block' op output tensor must have shape 1x1}}
    %result = ttl.tile_mul_reduce_block
        %input, %input, %output scale = 1.000000e+00 into dst[%c0]
        : tensor<1x2x!ttcore.tile<32x32, bf16>>,
          tensor<1x2x!ttcore.tile<32x32, bf16>>,
          tensor<1x2x!ttcore.tile<32x32, bf16>>
          -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

// The semantic reduction scale must be finite and positive.
module {
  func.func @nonpositive_scale(
      %input: !ttcore.tile<32x32, bf16>,
      %output: !ttcore.tile<32x32, bf16>) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{'ttl.tile_mul_reduce_block' op scale must be finite and positive}}
    %result = ttl.tile_mul_reduce_block
        %input, %input, %output scale = 0.000000e+00 into dst[%c0]
        : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>,
          !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    return
  }
}
