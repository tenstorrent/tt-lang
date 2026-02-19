// Verifier tests for ttl.store op.
// RUN: ttlang-opt --verify-diagnostics --split-input-file %s

// -----

// Element type mismatch between tensor and view.
func.func @store_element_type_mismatch(
    %tensor: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %view: tensor<2x2x!ttcore.tile<32x32, bf16>>) {
  // expected-error @below {{tensor element type ('!ttcore.tile<32x32, f32>') must match view element type ('!ttcore.tile<32x32, bf16>')}}
  ttl.store %tensor, %view : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, bf16>>
  func.return
}

// -----

// Rank mismatch between tensor and view.
func.func @store_rank_mismatch(
    %tensor: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %view: tensor<2x2x1x!ttcore.tile<32x32, f32>>) {
  // expected-error @below {{tensor rank (2) must match view rank (3)}}
  ttl.store %tensor, %view : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x1x!ttcore.tile<32x32, f32>>
  func.return
}

// -----

// Shape mismatch between tensor and view.
func.func @store_shape_mismatch(
    %tensor: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %view: tensor<1x2x!ttcore.tile<32x32, f32>>) {
  // expected-error @below {{tensor shape dimension 0 (2) must match view shape dimension (1)}}
  ttl.store %tensor, %view : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<1x2x!ttcore.tile<32x32, f32>>
  func.return
}
