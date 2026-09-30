// Summary: Verify ttl.exp and ttl.tile_exp reject iteration counts that do not
// cover exactly one full tile.
//
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// Zero iterations leave the complete tile unevaluated.
func.func @exp_zero_iterations(
    %arg0: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  // expected-error @below {{attribute 'iterations' failed to satisfy constraint}}
  %0 = ttl.exp %arg0 {iterations = 0 : i32}
      : tensor<1x1x!ttcore.tile<32x32, f32>>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  return %0 : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

// Negative iterations are invalid loop bounds.
func.func @exp_negative_iterations(
    %arg0: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  // expected-error @below {{attribute 'iterations' failed to satisfy constraint}}
  %0 = ttl.exp %arg0 {iterations = -1 : i32}
      : tensor<1x1x!ttcore.tile<32x32, f32>>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  return %0 : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

// Four iterations evaluate only part of a full tile.
func.func @exp_unsupported_positive_iterations(
    %arg0: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  // expected-error @below {{attribute 'iterations' failed to satisfy constraint}}
  %0 = ttl.exp %arg0 {iterations = 4 : i32}
      : tensor<1x1x!ttcore.tile<32x32, f32>>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  return %0 : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

// Tile-level operations enforce the same zero-iteration constraint.
func.func @tile_exp_zero_iterations(
    %arg0: !ttcore.tile<32x32, f32>)
    -> !ttcore.tile<32x32, f32> {
  %c0 = arith.constant 0 : index
  // expected-error @below {{attribute 'iterations' failed to satisfy constraint}}
  %0 = ttl.tile_exp %arg0 into dst[%c0] {iterations = 0 : i32}
      : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  return %0 : !ttcore.tile<32x32, f32>
}

// -----

// Tile-level operations reject negative iteration counts.
func.func @tile_exp_negative_iterations(
    %arg0: !ttcore.tile<32x32, f32>)
    -> !ttcore.tile<32x32, f32> {
  %c0 = arith.constant 0 : index
  // expected-error @below {{attribute 'iterations' failed to satisfy constraint}}
  %0 = ttl.tile_exp %arg0 into dst[%c0] {iterations = -1 : i32}
      : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  return %0 : !ttcore.tile<32x32, f32>
}

// -----

// Tile-level operations reject unsupported positive iteration counts.
func.func @tile_exp_unsupported_positive_iterations(
    %arg0: !ttcore.tile<32x32, f32>)
    -> !ttcore.tile<32x32, f32> {
  %c0 = arith.constant 0 : index
  // expected-error @below {{attribute 'iterations' failed to satisfy constraint}}
  %0 = ttl.tile_exp %arg0 into dst[%c0] {iterations = 2147483647 : i32}
      : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  return %0 : !ttcore.tile<32x32, f32>
}
