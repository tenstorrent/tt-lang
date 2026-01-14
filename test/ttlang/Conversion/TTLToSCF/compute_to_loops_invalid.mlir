// Summary: Verify error diagnostics in ttl-lower-to-loops pass.
//
// Tests that the pass emits appropriate errors for IR that cannot be
// correctly lowered to SCF loops due to rank mismatches between
// ttl.linearized_index maps and the iteration domain.
//
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-lower-to-loops))' --verify-diagnostics --split-input-file

#map = affine_map<(d0) -> (d0)>

// Purpose: linearized_index with mismatched rank (2D map but 1D iteration domain).
// Expected error: index_map dimensions don't match iteration domain rank.
func.func @linearized_index_rank_mismatch(%a: tensor<4x!ttcore.tile<32x32, f32>>,
                                           %b: tensor<4x!ttcore.tile<32x32, f32>>)
    -> tensor<4x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<4x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>
  %b_att = ttl.attach_cb %b, %cbb : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x!ttcore.tile<32x32, f32>>

  %0 = ttl.compute ins(%a_att, %b_att : tensor<4x!ttcore.tile<32x32, f32>>, tensor<4x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<4x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    // expected-error @below {{index_map has 2 dimensions but iteration domain has 1}}
    %idx = ttl.linearized_index affine_map<(d0, d1) -> (d0 * 4 + d1)> : index
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x!ttcore.tile<32x32, f32>>
}

// -----

#map2d = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: linearized_index with 3D map in 2D iteration domain.
// Expected error: index_map dimensions don't match iteration domain rank.
func.func @linearized_index_3d_map_2d_domain(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                               %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_att = ttl.attach_cb %b, %cbb : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %0 = ttl.compute ins(%a_att, %b_att : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map2d, #map2d, #map2d], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    // expected-error @below {{index_map has 3 dimensions but iteration domain has 2}}
    %idx = ttl.linearized_index affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 4 + d2)> : index
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

#map1d = affine_map<(d0) -> (d0)>
#map2d_iter = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: linearized_index with 1D map in 2D iteration domain (fewer dims than expected).
// Expected error: index_map dimensions don't match iteration domain rank.
func.func @linearized_index_1d_map_2d_domain(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                               %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %cba = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cbb = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %a_att = ttl.attach_cb %a, %cba : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_att = ttl.attach_cb %b, %cbb : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %0 = ttl.compute ins(%a_att, %b_att : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map2d_iter, #map2d_iter, #map2d_iter], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
    // expected-error @below {{index_map has 1 dimensions but iteration domain has 2}}
    %idx = ttl.linearized_index affine_map<(d0) -> (d0)> : index
    %sum = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}
