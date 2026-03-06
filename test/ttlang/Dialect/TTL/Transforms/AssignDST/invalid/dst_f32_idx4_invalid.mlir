// Summary: f32 capacity overflow should fail before allocating dst_idx 4 (default with double buffering).
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-assign-dst))' --verify-diagnostics

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: Exceed default dst capacity (4) using unary ops on separate block args.
// Unary ops require copy_tile from CB to DST, so five simultaneously live
// unary results exceed f32 DST capacity of 4.
func.func @f32_capacity_overflow(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                 %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                 %c: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                 %d: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                 %e: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb4 = ttl.bind_cb {cb_index = 4, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cbout = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %d_cb = ttl.attach_cb %d, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %e_cb = ttl.attach_cb %e, %cb4 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cbout : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %out_view = ttl.cb_reserve %cbout : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // expected-error @below {{insufficient DST registers: all 4 registers in use (spilling not yet implemented)}}
  %result = ttl.compute
      ins(%a_cb, %b_cb, %c_cb, %d_cb, %e_cb :
          tensor<2x2x!ttcore.tile<32x32, f32>>,
          tensor<2x2x!ttcore.tile<32x32, f32>>,
          tensor<2x2x!ttcore.tile<32x32, f32>>,
          tensor<2x2x!ttcore.tile<32x32, f32>>,
          tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %c_tile: !ttcore.tile<32x32, f32>,
       %d_tile: !ttcore.tile<32x32, f32>,
       %e_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    // Five unary ops on separate block args - each needs copy_tile + DST.
    // All five results are used later, keeping them live simultaneously.
    %abs_a = ttl.tile_abs %a_tile : !ttcore.tile<32x32, f32>
    %abs_b = ttl.tile_abs %b_tile : !ttcore.tile<32x32, f32>
    %abs_c = ttl.tile_abs %c_tile : !ttcore.tile<32x32, f32>
    %abs_d = ttl.tile_abs %d_tile : !ttcore.tile<32x32, f32>
    %abs_e = ttl.tile_abs %e_tile : !ttcore.tile<32x32, f32>
    // Use all five results to keep them simultaneously live, exceeding capacity=4.
    %sum0 = ttl.tile_add %abs_a, %abs_b : !ttcore.tile<32x32, f32>
    %sum1 = ttl.tile_add %abs_c, %abs_d : !ttcore.tile<32x32, f32>
    %sum2 = ttl.tile_add %sum0, %abs_e : !ttcore.tile<32x32, f32>
    %final = ttl.tile_add %sum1, %sum2 : !ttcore.tile<32x32, f32>
    ttl.tile_store %final, %out_view : !ttcore.tile<32x32, f32>, tensor<2x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
