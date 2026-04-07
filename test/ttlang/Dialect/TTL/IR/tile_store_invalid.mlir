// Verifier tests for ttl.tile_store op.
// RUN: ttlang-opt --verify-diagnostics --split-input-file %s

// -----

// Tile operand must be !ttcore.tile type.
func.func @tile_store_non_tile_operand() {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2], !ttcore.tile<32x32, f32>, 2>
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
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2], !ttcore.tile<32x32, bf16>, 2>
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
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %view = ttl.cb_reserve %cb : <[2, 3], !ttcore.tile<32x32, f32>, 2> -> tensor<2x3x!ttcore.tile<32x32, f32>>
  // expected-error @below {{'ttl.tile_store' op expected 0 or 2 indices, got 1}}
  ttl.tile_store %tile, %view[%c0] : !ttcore.tile<32x32, f32>, tensor<2x3x!ttcore.tile<32x32, f32>>
  func.return
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Empty indices inside a compute body on a rank-2 view.
func.func @tile_store_empty_indices_in_compute(
    %in: tensor<2x2x!ttcore.tile<32x32, f32>>,
    %out: tensor<2x2x!ttcore.tile<32x32, f32>>) {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %in_cb = ttl.attach_cb %in, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_cb = ttl.attach_cb %out, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %view = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute ins(%in_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%out_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>):
    %exp = ttl.tile_exp %a : !ttcore.tile<32x32, f32>
    // expected-error @below {{'ttl.tile_store' op expected 2 indices inside compute body, got 0}}
    ttl.tile_store %exp, %view[] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return
}
