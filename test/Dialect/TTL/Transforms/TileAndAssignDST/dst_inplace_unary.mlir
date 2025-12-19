// Summary: SFPU unary chain should flow via tokens with dst_idx debug attrs.
// RUN: ttlang-opt %s --ttl-tile-and-assign-dst --canonicalize --cse --split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: verify copy_tile emits token+tile, unary ops consume copied tile,
// and dst_idx annotations are added to math ops.
// CHECK-LABEL: func.func @inplace_unary_chain
func.func @inplace_unary_chain(%a: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

// CHECK: %[[RESULT:.*]] = ttl.compute
// CHECK: ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[A]], %{{.*}}, %{{.*}} : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[EXP:.*]] = ttl.tile_exp %[[DTILE]] {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[RELU:.*]] = ttl.tile_relu %[[EXP]] {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[SIG:.*]] = ttl.tile_sigmoid %[[RELU]] {dst_idx = 3 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   ttl.yield %[[SIG]] : !ttcore.tile<32x32, f32>
// CHECK: }
  %result = ttl.compute
      ins(%a_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %exp = ttl.tile_exp %a_tile : !ttcore.tile<32x32, f32>
    %relu = ttl.tile_relu %exp : !ttcore.tile<32x32, f32>
    %sigmoid = ttl.tile_sigmoid %relu : !ttcore.tile<32x32, f32>
    ttl.yield %sigmoid : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: return %[[RESULT]]
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
