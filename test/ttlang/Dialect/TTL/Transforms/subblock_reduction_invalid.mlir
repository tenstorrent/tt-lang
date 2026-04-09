// Verify that subblocking emits an error when reduction dimensions exceed
// DST capacity and cannot be partitioned.
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-subblock-compute-for-dst))' --verify-diagnostics

#map_in = affine_map<(d0, d1) -> (d0, d1)>
#map_out = affine_map<(d0, d1) -> (d0)>

func.func @reduction_exceeds_dst_budget(
    %a: tensor<2x10x!ttcore.tile<32x32, f32>>)
    -> tensor<2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[2, 10], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, block_count = 2}
      : !ttl.cb<[2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<2x10x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[2, 10], !ttcore.tile<32x32, f32>, 2>)
      -> tensor<2x10x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1
      : (tensor<2x!ttcore.tile<32x32, f32>>,
         !ttl.cb<[2], !ttcore.tile<32x32, f32>, 2>)
      -> tensor<2x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb1
      : <[2], !ttcore.tile<32x32, f32>, 2>
      -> tensor<2x!ttcore.tile<32x32, f32>>

  // expected-error @below {{'ttl.compute' op reduction dimensions require 10 DST tiles per iteration but only 8 are available; cannot subblock}}
  %result = ttl.compute
      ins(%a_cb : tensor<2x10x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map_in, #map_out],
       iterator_types = ["parallel", "reduction"],
       ttl.unroll_factor = 8 : i64} {
  ^bb0(%in: !ttcore.tile<32x32, f32>, %acc: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %add = ttl.tile_add %in, %acc into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %add, %reserve[%i] from dst[%c0]
        : !ttcore.tile<32x32, f32>, tensor<2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x!ttcore.tile<32x32, f32>>
}
