// Summary: separate-output-region=1 with small capacity overflows when enough
// unary results are simultaneously live.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=2 separate-output-region=1}))' --verify-diagnostics

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: Three unary ops on separate block args all consumed later.
// With capacity=2 and separate-output-region=1, three simultaneously live
// unary results exceed the available registers.
func.func @separate_output_region_overflow(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                           %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                           %c: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %out_view = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{insufficient DST registers: all 2 registers in use (spilling not yet implemented)}}
  %result = ttl.compute
      ins(%a_cb, %b_cb, %c_cb :
          tensor<2x2x!ttcore.tile<32x32, f32>>,
          tensor<2x2x!ttcore.tile<32x32, f32>>,
          tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %c_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    // Three unary ops on separate block args - each needs copy_tile + DST.
    // All three results are used later, keeping them live simultaneously.
    %abs_a = ttl.tile_abs %a_tile : !ttcore.tile<32x32, f32>
    %abs_b = ttl.tile_abs %b_tile : !ttcore.tile<32x32, f32>
    %abs_c = ttl.tile_abs %c_tile : !ttcore.tile<32x32, f32>
    // Use all three results to keep them live simultaneously, exceeding capacity=2.
    %sum1 = ttl.tile_add %abs_a, %abs_b : !ttcore.tile<32x32, f32>
    %final = ttl.tile_add %sum1, %abs_c : !ttcore.tile<32x32, f32>
    ttl.tile_store %final, %out_view[%i, %j] : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
