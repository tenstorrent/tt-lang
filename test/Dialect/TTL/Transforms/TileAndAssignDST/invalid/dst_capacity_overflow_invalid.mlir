// Summary: capacity overflow should emit a clear diagnostic.
// RUN: ttlang-opt %s --ttl-tile-and-assign-dst --split-input-file --verify-diagnostics

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: many block args make peak live tiles exceed default capacity (8).
func.func @capacity_overflow(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                             %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
                             %c: tensor<2x2x!ttcore.tile<32x32, f32>>,
                             %d: tensor<2x2x!ttcore.tile<32x32, f32>>,
                             %e: tensor<2x2x!ttcore.tile<32x32, f32>>,
                             %f: tensor<2x2x!ttcore.tile<32x32, f32>>,
                             %g: tensor<2x2x!ttcore.tile<32x32, f32>>,
                             %h: tensor<2x2x!ttcore.tile<32x32, f32>>,
                             %i: tensor<2x2x!ttcore.tile<32x32, f32>>,
                             %j: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb4 = ttl.bind_cb {cb_index = 4, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb5 = ttl.bind_cb {cb_index = 5, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb6 = ttl.bind_cb {cb_index = 6, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb7 = ttl.bind_cb {cb_index = 7, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb8 = ttl.bind_cb {cb_index = 8, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb9 = ttl.bind_cb {cb_index = 9, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb10 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %d_cb = ttl.attach_cb %d, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %e_cb = ttl.attach_cb %e, %cb4 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %f_cb = ttl.attach_cb %f, %cb5 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %g_cb = ttl.attach_cb %g, %cb6 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %h_cb = ttl.attach_cb %h, %cb7 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %i_cb = ttl.attach_cb %i, %cb8 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %j_cb = ttl.attach_cb %j, %cb9 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb10 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // expected-error @+1 {{operation chain requires}}
  %result = ttl.compute
      ins(%a_cb, %b_cb, %c_cb, %d_cb, %e_cb,
          %f_cb, %g_cb, %h_cb, %i_cb, %j_cb :
          tensor<2x2x!ttcore.tile<32x32, f32>>,
          tensor<2x2x!ttcore.tile<32x32, f32>>,
          tensor<2x2x!ttcore.tile<32x32, f32>>,
          tensor<2x2x!ttcore.tile<32x32, f32>>,
          tensor<2x2x!ttcore.tile<32x32, f32>>,
          tensor<2x2x!ttcore.tile<32x32, f32>>,
          tensor<2x2x!ttcore.tile<32x32, f32>>,
          tensor<2x2x!ttcore.tile<32x32, f32>>,
          tensor<2x2x!ttcore.tile<32x32, f32>>,
          tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map, #map,
                        #map, #map, #map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %c_tile: !ttcore.tile<32x32, f32>,
       %d_tile: !ttcore.tile<32x32, f32>,
       %e_tile: !ttcore.tile<32x32, f32>,
       %f_tile: !ttcore.tile<32x32, f32>,
       %g_tile: !ttcore.tile<32x32, f32>,
       %h_tile: !ttcore.tile<32x32, f32>,
       %i_tile: !ttcore.tile<32x32, f32>,
       %j_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    // Many live inputs force peak live tiles > capacity (10 inputs + temps).
    %sum1 = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %sum2 = ttl.tile_add %c_tile, %d_tile : !ttcore.tile<32x32, f32>
    %sum3 = ttl.tile_add %e_tile, %f_tile : !ttcore.tile<32x32, f32>
    %sum4 = ttl.tile_add %g_tile, %h_tile : !ttcore.tile<32x32, f32>
    %sum5 = ttl.tile_add %i_tile, %j_tile : !ttcore.tile<32x32, f32>
    %sum6 = ttl.tile_add %sum1, %sum2 : !ttcore.tile<32x32, f32>
    %sum7 = ttl.tile_add %sum3, %sum4 : !ttcore.tile<32x32, f32>
    %sum8 = ttl.tile_add %sum5, %sum6 : !ttcore.tile<32x32, f32>
    %sum9 = ttl.tile_add %sum7, %sum8 : !ttcore.tile<32x32, f32>
    ttl.yield %sum9 : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

