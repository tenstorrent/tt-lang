// Verifies structural and type invariants of the internal row-normalization
// block operation.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// The specialized hardware schedule supports bf16 tiles only.
module {
  func.func @unsupported_dtype(
      %input: !ttcore.tile<32x32, f32>,
      %output: !ttcore.tile<32x32, f32>) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{'ttl.tile_row_normalization_block' op supports bf16 tiles only}}
    %result = ttl.tile_row_normalization_block
        %input, %input, %output scale = 1.000000e+00 epsilon = 1.000000e-05
        has_gamma = false num_tiles = 1 into dst[%c0]
        : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>,
          !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    return
  }
}

// -----

// The semantic scale must be finite and positive.
module {
  func.func @nonfinite_scale(
      %input: !ttcore.tile<32x32, bf16>,
      %output: !ttcore.tile<32x32, bf16>) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{'ttl.tile_row_normalization_block' op scale must be finite and positive}}
    %result = ttl.tile_row_normalization_block
        %input, %input, %output scale = 0x7FC00000 epsilon = 1.000000e-05
        has_gamma = false num_tiles = 1 into dst[%c0]
        : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>,
          !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

// Epsilon must be finite and positive.
module {
  func.func @nonpositive_epsilon(
      %input: !ttcore.tile<32x32, bf16>,
      %output: !ttcore.tile<32x32, bf16>) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{'ttl.tile_row_normalization_block' op epsilon must be finite and positive}}
    %result = ttl.tile_row_normalization_block
        %input, %input, %output scale = 1.000000e+00 epsilon = 0.000000e+00
        has_gamma = false num_tiles = 1 into dst[%c0]
        : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>,
          !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

// Full gamma must have the same row extent as the output.
module {
  func.func @mismatched_full_gamma(
      %input: tensor<1x3x!ttcore.tile<32x32, bf16>>,
      %gamma: tensor<1x1x!ttcore.tile<32x32, bf16>>,
      %output: tensor<1x3x!ttcore.tile<32x32, bf16>>) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{'ttl.tile_row_normalization_block' op gamma tensor shape must match the output shape}}
    %result = ttl.tile_row_normalization_block
        %input, %gamma, %output scale = 1.000000e+00 epsilon = 1.000000e-05
        has_gamma = true num_tiles = 3 into dst[%c0]
        : tensor<1x3x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x3x!ttcore.tile<32x32, bf16>>
          -> tensor<1x3x!ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// Tensor operands must be static rank-2 values.
module {
  func.func @invalid_tensor_rank(
      %input: tensor<1x1x1x!ttcore.tile<32x32, bf16>>,
      %output: tensor<1x1x1x!ttcore.tile<32x32, bf16>>) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{'ttl.tile_row_normalization_block' op tensor operands must be static rank-2 tensors}}
    %result = ttl.tile_row_normalization_block
        %input, %input, %output scale = 1.000000e+00 epsilon = 1.000000e-05
        has_gamma = false num_tiles = 1 into dst[%c0]
        : tensor<1x1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x1x!ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// Input and output must describe the same one-row tile tensor.
module {
  func.func @mismatched_input_output_shape(
      %input: tensor<1x2x!ttcore.tile<32x32, bf16>>,
      %output: tensor<1x3x!ttcore.tile<32x32, bf16>>) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{'ttl.tile_row_normalization_block' op input and output must have the same one-row tensor shape}}
    %result = ttl.tile_row_normalization_block
        %input, %input, %output scale = 1.000000e+00 epsilon = 1.000000e-05
        has_gamma = false num_tiles = 2 into dst[%c0]
        : tensor<1x2x!ttcore.tile<32x32, bf16>>,
          tensor<1x2x!ttcore.tile<32x32, bf16>>,
          tensor<1x3x!ttcore.tile<32x32, bf16>>
          -> tensor<1x3x!ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// Disabled gamma multiplication uses the input operand as the placeholder.
module {
  func.func @distinct_disabled_gamma(
      %input: !ttcore.tile<32x32, bf16>,
      %gamma: !ttcore.tile<32x32, bf16>,
      %output: !ttcore.tile<32x32, bf16>) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{'ttl.tile_row_normalization_block' op gamma must equal input when has_gamma is false}}
    %result = ttl.tile_row_normalization_block
        %input, %gamma, %output scale = 1.000000e+00 epsilon = 1.000000e-05
        has_gamma = false num_tiles = 1 into dst[%c0]
        : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>,
          !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

// Scalarization metadata must equal the tensor row width.
module {
  func.func @mismatched_num_tiles(
      %input: tensor<1x3x!ttcore.tile<32x32, bf16>>,
      %output: tensor<1x3x!ttcore.tile<32x32, bf16>>) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{'ttl.tile_row_normalization_block' op num_tiles must match the row tensor width}}
    %result = ttl.tile_row_normalization_block
        %input, %input, %output scale = 1.000000e+00 epsilon = 1.000000e-05
        has_gamma = false num_tiles = 2 into dst[%c0]
        : tensor<1x3x!ttcore.tile<32x32, bf16>>,
          tensor<1x3x!ttcore.tile<32x32, bf16>>,
          tensor<1x3x!ttcore.tile<32x32, bf16>>
          -> tensor<1x3x!ttcore.tile<32x32, bf16>>
    return
  }
}
