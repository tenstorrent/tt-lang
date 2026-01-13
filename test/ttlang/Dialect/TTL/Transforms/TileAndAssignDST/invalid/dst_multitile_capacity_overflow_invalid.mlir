// Summary: multi-tile capacity overflow should emit a clear diagnostic.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-tile-and-assign-dst))' --split-input-file --verify-diagnostics

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: Multi-tile capacity overflow. With 3x3 grid (9 tiles) and 2 inputs,
// total need is 2 inputs + 9 tiles = 11 but default capacity is only 8.
func.func @multitile_capacity_overflow(%a: tensor<3x3x!ttcore.tile<32x32, f32>>,
                                       %b: tensor<3x3x!ttcore.tile<32x32, f32>>)
    -> tensor<3x3x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<3x3x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[3, 3], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[3, 3], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[3, 3], !ttcore.tile<32x32, f32>, 1>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<3x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[3, 3], !ttcore.tile<32x32, f32>, 1>) -> tensor<3x3x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<3x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[3, 3], !ttcore.tile<32x32, f32>, 1>) -> tensor<3x3x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<3x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[3, 3], !ttcore.tile<32x32, f32>, 1>) -> tensor<3x3x!ttcore.tile<32x32, f32>>

  // expected-error @+1 {{multi-tile compute requires 11 DST registers (2 inputs + 9 tiles) but capacity is only 8}}
  %result = ttl.compute
      ins(%a_cb, %b_cb :
          tensor<3x3x!ttcore.tile<32x32, f32>>,
          tensor<3x3x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<3x3x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    // Two inputs + 9 tiles = 11 DST registers total. Exceeds capacity of 8.
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<3x3x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<3x3x!ttcore.tile<32x32, f32>>
}
