// Verifies structural and type invariants of the internal row-normalization
// block operation.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// Repeated gamma is undefined when gamma multiplication is disabled.
module {
  func.func @repeat_without_gamma(
      %input: !ttcore.tile<32x32, bf16>,
      %output: !ttcore.tile<32x32, bf16>) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{'ttl.tile_row_normalization_block' op repeat_gamma requires has_gamma}}
    %result = ttl.tile_row_normalization_block
        %input, %input, %output scale = 1.000000e+00 epsilon = 1.000000e-05
        has_gamma = false repeat_gamma = true into dst[%c0]
        : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>,
          !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    return
  }
}

// -----

// The specialized hardware schedule supports bf16 tiles only.
module {
  func.func @unsupported_dtype(
      %input: !ttcore.tile<32x32, f32>,
      %output: !ttcore.tile<32x32, f32>) {
    %c0 = arith.constant 0 : index
    // expected-error @below {{'ttl.tile_row_normalization_block' op supports bf16 tiles only}}
    %result = ttl.tile_row_normalization_block
        %input, %input, %output scale = 1.000000e+00 epsilon = 1.000000e-05
        has_gamma = false repeat_gamma = false into dst[%c0]
        : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>,
          !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
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
    // expected-error @below {{'ttl.tile_row_normalization_block' op gamma tensor shape does not match gamma mode}}
    %result = ttl.tile_row_normalization_block
        %input, %gamma, %output scale = 1.000000e+00 epsilon = 1.000000e-05
        has_gamma = true repeat_gamma = false into dst[%c0]
        : tensor<1x3x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>,
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
        has_gamma = false repeat_gamma = false into dst[%c0]
        : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>,
          !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    return
  }
}
