// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-tile-and-assign-dst{dst-capacity=2}))' -verify-diagnostics --split-input-file

#map = affine_map<(d0) -> (d0)>

// Test: Footprint exceeds small capacity
func.func @test_capacity_exceeded(
    %a: tensor<2x!ttcore.tile<32x32, f32>>,
    %b: tensor<2x!ttcore.tile<32x32, f32>>,
    %c: tensor<2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x!ttcore.tile<32x32, f32>> {

  %init = tensor.empty() : tensor<2x!ttcore.tile<32x32, f32>>
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2], !ttcore.tile<32x32, f32>, 2>
  %cb_out = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb_out : (tensor<2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x!ttcore.tile<32x32, f32>>

  // 3 inputs + 1 output = 4 DST slots required, exceeds capacity of 2
  // expected-error @+1 {{requires 4 DST slots but capacity is only 2}}
  %result = ttl.compute
      ins(%a_cb, %b_cb, %c_cb : tensor<2x!ttcore.tile<32x32, f32>>, tensor<2x!ttcore.tile<32x32, f32>>, tensor<2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} {
  ^bb0(%a_t: !ttcore.tile<32x32, f32>, %b_t: !ttcore.tile<32x32, f32>, %c_t: !ttcore.tile<32x32, f32>, %out_t: !ttcore.tile<32x32, f32>):
    %tmp = ttl.tile_add %a_t, %b_t : !ttcore.tile<32x32, f32>
    %res = ttl.tile_add %tmp, %c_t : !ttcore.tile<32x32, f32>
    ttl.yield %res : !ttcore.tile<32x32, f32>
  } -> tensor<2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x!ttcore.tile<32x32, f32>>
}
