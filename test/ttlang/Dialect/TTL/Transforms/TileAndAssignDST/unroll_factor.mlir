// Summary: Verify unroll factor computation for DST utilization optimization.
// Tests that ttl.unroll_factor attribute is correctly computed based on:
//   unroll_factor = min(capacity / footprint_per_iteration, numTiles)
// where footprint_per_iteration = inputs_footprint + num_outputs for a single tile loop iteration.
//
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-tile-and-assign-dst{dst-capacity=8}))' --split-input-file | FileCheck %s --check-prefix=CAP8
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-tile-and-assign-dst{dst-capacity=4}))' --split-input-file | FileCheck %s --check-prefix=CAP4

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 2x2 tensor (4 tiles), 2 inputs, footprintPerTile=3, capacity=8
// Expected unroll_factor = min(8/3, 4) = min(2, 4) = 2
// CAP8-LABEL: func.func @test_2x2_binary
// CAP8: ttl.compute
// CAP8-SAME: ttl.unroll_factor = 2 : i32
// CAP4-LABEL: func.func @test_2x2_binary
// CAP4: ttl.compute
// CAP4-NOT: ttl.unroll_factor
// CAP4-SAME: ins
func.func @test_2x2_binary(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                           %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 1x1 tensor (single tile), no unrolling needed
// Expected: no ttl.unroll_factor attribute (numTiles=1, factor skipped)
// CAP8-LABEL: func.func @test_single_tile
// CAP8: ttl.compute
// CAP8-NOT: ttl.unroll_factor
// CAP8-SAME: ins
// CAP4-LABEL: func.func @test_single_tile
// CAP4: ttl.compute
// CAP4-NOT: ttl.unroll_factor
// CAP4-SAME: ins
func.func @test_single_tile(%a: tensor<1x1x!ttcore.tile<32x32, f32>>,
                            %b: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x1x!ttcore.tile<32x32, f32>>,
                         tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

#map1d = affine_map<(d0) -> (d0)>

// Purpose: 4x1 tensor (4 tiles), 2 inputs, footprintPerTile=3, capacity=8
// Expected unroll_factor = min(8/3, 4) = min(2, 4) = 2
// CAP8-LABEL: func.func @test_4x1_binary
// CAP8: ttl.compute
// CAP8-SAME: ttl.unroll_factor = 2 : i32
func.func @test_4x1_binary(%a: tensor<4x!ttcore.tile<32x32, f32>>,
                           %b: tensor<4x!ttcore.tile<32x32, f32>>)
    -> tensor<4x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<4x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<4x!ttcore.tile<32x32, f32>>,
                         tensor<4x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<4x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map1d, #map1d, #map1d],
       iterator_types = ["parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<4x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<4x!ttcore.tile<32x32, f32>>
}

// -----

#map1d = affine_map<(d0) -> (d0)>

// Purpose: 8 tiles total, 2 inputs, footprintPerTile=3, capacity=8
// Expected unroll_factor = min(8/3, 8) = min(2, 8) = 2
// CAP8-LABEL: func.func @test_max_unroll_factor
// CAP8: ttl.compute
// CAP8-SAME: ttl.unroll_factor = 2 : i32
func.func @test_max_unroll_factor(%a: tensor<8x!ttcore.tile<32x32, f32>>,
                                  %b: tensor<8x!ttcore.tile<32x32, f32>>)
    -> tensor<8x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<8x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[8], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[8], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[8], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<8x!ttcore.tile<32x32, f32>>, !ttl.cb<[8], !ttcore.tile<32x32, f32>, 2>) -> tensor<8x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<8x!ttcore.tile<32x32, f32>>, !ttl.cb<[8], !ttcore.tile<32x32, f32>, 2>) -> tensor<8x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<8x!ttcore.tile<32x32, f32>>, !ttl.cb<[8], !ttcore.tile<32x32, f32>, 2>) -> tensor<8x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<8x!ttcore.tile<32x32, f32>>,
                         tensor<8x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<8x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map1d, #map1d, #map1d],
       iterator_types = ["parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<8x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<8x!ttcore.tile<32x32, f32>>
}

// -----

#map1d = affine_map<(d0) -> (d0)>

// Purpose: 3 tiles with 1 input (unary op), footprintPerTile=2, capacity=8
// Expected unroll_factor = min(8/2, 3) = min(4, 3) = 3
// CAP8-LABEL: func.func @test_unary_op
// CAP8: ttl.compute
// CAP8-SAME: ttl.unroll_factor = 3 : i32
func.func @test_unary_op(%a: tensor<3x!ttcore.tile<32x32, f32>>)
    -> tensor<3x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<3x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[3], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[3], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<3x!ttcore.tile<32x32, f32>>, !ttl.cb<[3], !ttcore.tile<32x32, f32>, 2>) -> tensor<3x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<3x!ttcore.tile<32x32, f32>>, !ttl.cb<[3], !ttcore.tile<32x32, f32>, 2>) -> tensor<3x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb : tensor<3x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<3x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map1d, #map1d],
       iterator_types = ["parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %exp = ttl.tile_exp %a_tile : !ttcore.tile<32x32, f32>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<3x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<3x!ttcore.tile<32x32, f32>>
}
