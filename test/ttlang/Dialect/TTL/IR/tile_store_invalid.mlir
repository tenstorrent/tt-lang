// Verifier tests for ttl.tile_store op.
// RUN: ttlang-opt --verify-diagnostics --split-input-file %s

// -----

// Tile operand must be !ttcore.tile type.
func.func @tile_store_non_tile_operand() {
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2], !ttcore.tile<32x32, f32>, 2>
  %view = ttl.cb_reserve %cb : <[2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x!ttcore.tile<32x32, f32>>
  %val = arith.constant 1.0 : f32
  // expected-error @below {{'ttl.tile_store' op tile operand must be !ttcore.tile, got 'f32'}}
  ttl.tile_store %val, %view[] : f32, tensor<2x!ttcore.tile<32x32, f32>>
  func.return
}

// -----

// View element type must match tile type.
func.func @tile_store_element_type_mismatch(
    %tile: !ttcore.tile<32x32, f32>) {
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2], !ttcore.tile<32x32, bf16>, 2>
  %view = ttl.cb_reserve %cb : <[2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x!ttcore.tile<32x32, bf16>>
  // expected-error @below {{'ttl.tile_store' op view element type ('!ttcore.tile<32x32, bf16>') must match tile type ('!ttcore.tile<32x32, f32>')}}
  ttl.tile_store %tile, %view[] : !ttcore.tile<32x32, f32>, tensor<2x!ttcore.tile<32x32, bf16>>
  func.return
}

// -----

// Wrong number of indices: 1 index for a rank-2 view.
func.func @tile_store_wrong_index_count(
    %tile: !ttcore.tile<32x32, f32>) {
  %c0 = arith.constant 0 : index
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %view = ttl.cb_reserve %cb : <[2, 3], !ttcore.tile<32x32, f32>, 2> -> tensor<2x3x!ttcore.tile<32x32, f32>>
  // expected-error @below {{'ttl.tile_store' op expected 0 or 2 indices, got 1}}
  ttl.tile_store %tile, %view[%c0] : !ttcore.tile<32x32, f32>, tensor<2x3x!ttcore.tile<32x32, f32>>
  func.return
}
