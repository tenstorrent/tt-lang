// Summary: SFPU unary chain should reuse the same DST register in-place.
// RUN: ttlang-opt %s --ttl-tile-and-assign-dst --split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @inplace_unary_chain
// Purpose: verify dst_idx reuse across multiple unary ops.
func.func @inplace_unary_chain(%a: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: %[[RESULT:.*]] = ttl.compute
  // CHECK-SAME: ins(%{{.*}} : {{.*}}) outs(%{{.*}} : {{.*}})
  // CHECK-SAME: {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
  // CHECK-NEXT: ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[EXP:.*]] = ttl.tile_exp %[[A]] {dst_idx = 0 : i32}
  // CHECK-NEXT:   %[[RELU:.*]] = ttl.tile_relu %[[EXP]] {dst_idx = 0 : i32}
  // CHECK-NEXT:   %[[SIGMOID:.*]] = ttl.tile_sigmoid %[[RELU]] {dst_idx = 0 : i32}
  // CHECK-NEXT:   ttl.yield %[[SIGMOID]]
  // CHECK-NEXT: }
  // CHECK-NOT: tile_batch_size
  %result = ttl.compute
      ins(%a_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %exp = ttl.tile_exp %a_tile : !ttcore.tile<32x32, f32>
    %relu = ttl.tile_relu %exp : !ttcore.tile<32x32, f32>
    %sigmoid = ttl.tile_sigmoid %relu : !ttcore.tile<32x32, f32>
    ttl.yield %sigmoid : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: return %[[RESULT]]
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

