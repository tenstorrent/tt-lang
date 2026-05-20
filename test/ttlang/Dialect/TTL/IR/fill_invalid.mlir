// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// Summary: Verifier-level rejection cases for ttl.fill: non-tile result
// element type, non-positive shape entries.

// Row-major element type (not !ttcore.tile) is rejected.
func.func @fill_row_major_rejected() {
  // expected-error @below {{result element type must be !ttcore.tile, got 'f32'}}
  %r = ttl.fill 1.000000e+00 : tensor<4x4xf32>
  return
}

// -----

// Zero shape entry is rejected.
func.func @fill_zero_shape() {
  // expected-error @below {{result shape[1] = 0 must be positive}}
  %r = ttl.fill 1.000000e+00 : tensor<4x0x!ttcore.tile<32x32, f32>>
  return
}

// -----

// Dynamic shape entry is rejected.
func.func @fill_dynamic_shape() {
  // expected-error @below {{result must have a static shape}}
  %r = ttl.fill 1.000000e+00 : tensor<4x?x!ttcore.tile<32x32, f32>>
  return
}
