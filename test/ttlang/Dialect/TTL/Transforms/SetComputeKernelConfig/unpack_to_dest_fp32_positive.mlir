// Summary: Verify ttl.unpack_to_dest_fp32 is set for SFPU f32 CB-to-DST input.
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-set-compute-kernel-config))' --split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: An SFPU unary op that consumes an f32 input block argument requires
// unpacking that CB tile directly into DST as f32.
// CHECK-LABEL: func.func @f32_sfpu_unary_sets_unpack
// CHECK-SAME: ttl.unpack_to_dest_fp32 = true
func.func @f32_sfpu_unary_sets_unpack(
    %a: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>

  %out_view = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %res = ttl.compute
      ins(%a_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      %exp = ttl.tile_exp %a_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
      ttl.tile_store %exp, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, f32>>

  return %res : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: f32 -> bf16 typecast consumes an f32 CB0 tile through DST, so it
// requires unpack-to-DST mode even though the output tile is bf16.
// CHECK-LABEL: func.func @f32_to_bf16_typecast_sets_unpack
// CHECK-SAME: ttl.unpack_to_dest_fp32 = true
func.func @f32_to_bf16_typecast_sets_unpack(
    %a: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  %c0 = arith.constant 0 : index
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %out_view = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %res = ttl.compute
      ins(%a_cb : tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
    ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, bf16>):
      %i = ttl.iter_index 0 : index
      %j = ttl.iter_index 1 : index
      %cast = ttl.tile_typecast %a_tile into dst[%c0] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, bf16>
      ttl.tile_store %cast, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.yield
  } -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  return %res : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
