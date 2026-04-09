// Summary: verify iter_index verifier catches invalid cases
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// -----
// Test: iter_index outside a ComputeOp body
func.func @outside_compute() {
  // expected-error @below {{'ttl.iter_index' op expects parent op 'ttl.compute'}}
  %idx = ttl.iter_index 0 : index
  return
}

// -----
// Test: dim out of range for the iteration domain
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @dim_out_of_range(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                             %cb_in: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
                             %cb_out: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cb_in
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cb_out
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %result = ttl.compute ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    // expected-error @below {{'ttl.iter_index' op dimension 3 is out of range for iteration domain of rank 2}}
    %idx = ttl.iter_index 3 : index
    %c0 = arith.constant 0 : index
    ttl.tile_store %arg0, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  return
}

// -----
// Test: negative dim (cast to unsigned wraps, so it is out of range)
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @negative_dim(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                         %cb_in: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
                         %cb_out: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cb_in
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cb_out
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %out_view = ttl.cb_reserve %cb_out : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %result = ttl.compute ins(%a_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    // expected-error @below {{'ttl.iter_index' op dimension -1 is out of range for iteration domain of rank 2}}
    %idx = ttl.iter_index -1 : index
    %c0 = arith.constant 0 : index
    ttl.tile_store %arg0, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  return
}
