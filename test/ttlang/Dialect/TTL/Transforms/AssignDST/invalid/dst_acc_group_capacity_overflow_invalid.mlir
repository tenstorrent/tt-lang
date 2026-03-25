// Summary: cross-compute accumulator allocation (Phase 5b) should emit a
// diagnostic when the accumulator register exceeds DST capacity.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=1}))' --verify-diagnostics

#map = affine_map<(d0, d1) -> (d0, d1)>

// Two computes in acc_group 0, each with one input passthrough. The input
// copy_tile consumes DST[0], so maxFootprint=1. The accumulator needs
// index 1, exceeding capacity=1.
func.func @acc_group_capacity_overflow(%a: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %out_view = ttl.cb_reserve %cb2 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // expected-error @below {{insufficient DST registers for cross-compute accumulator: accumulator requires register index 1 but only 1 registers are available (indices 0..0)}}
  %r0 = ttl.compute
      ins(%a_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"],
       ttl.acc_group = 0 : i32} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i0 = ttl.iter_index 0 : index
    %j0 = ttl.iter_index 1 : index
    ttl.tile_store %a_tile, %out_view[%i0, %j0] {acc = true} : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %r1 = ttl.compute
      ins(%a_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"],
       ttl.acc_group = 0 : i32} {
  ^bb0(%a_tile2: !ttcore.tile<32x32, f32>,
       %out_tile2: !ttcore.tile<32x32, f32>):
    %i1 = ttl.iter_index 0 : index
    %j1 = ttl.iter_index 1 : index
    ttl.tile_store %a_tile2, %out_view[%i1, %j1] {acc = true} : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  func.return %r1 : tensor<1x1x!ttcore.tile<32x32, f32>>
}
