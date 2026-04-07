// Verifier tests for ttl.store op.
// RUN: ttlang-opt --verify-diagnostics --split-input-file %s

// -----

// Element type mismatch between tensor and view.
func.func @store_element_type_mismatch(
    %tensor: tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %view = ttl.cb_reserve %cb : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{tensor element type ('!ttcore.tile<32x32, f32>') must match view element type ('!ttcore.tile<32x32, bf16>')}}
  ttl.store %tensor, %view : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, bf16>>
  func.return
}

// -----

// Rank mismatch between tensor and view.
func.func @store_rank_mismatch(
    %tensor: tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2, 1], !ttcore.tile<32x32, f32>, 2>
  %view = ttl.cb_reserve %cb : <[2, 2, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x1x!ttcore.tile<32x32, f32>>
  // expected-error @below {{tensor rank (2) must match view rank (3)}}
  ttl.store %tensor, %view : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x1x!ttcore.tile<32x32, f32>>
  func.return
}

// -----

// Shape mismatch between tensor and view.
func.func @store_shape_mismatch(
    %tensor: tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  %view = ttl.cb_reserve %cb : <[1, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<1x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{tensor shape dimension 0 (2) must match view shape dimension (1)}}
  ttl.store %tensor, %view : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<1x2x!ttcore.tile<32x32, f32>>
  func.return
}

// -----

// View not from cb_reserve.
func.func @store_view_not_from_reserve(
    %tensor: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %view: tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  // expected-error @below {{view must come from ttl.cb_reserve}}
  ttl.store %tensor, %view : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return
}
