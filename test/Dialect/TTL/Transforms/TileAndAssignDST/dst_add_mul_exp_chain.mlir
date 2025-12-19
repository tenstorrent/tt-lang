// Summary: three-op chain (add -> mul -> exp) should flow via tokens (no dst_idx).
// RUN: ttlang-opt %s --ttl-tile-and-assign-dst --split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @add_mul_exp_chain
// Purpose: verify copy_tile emits token+tile, tile ops consume copied tiles,
// and dst_idx/tile_batch_size are absent.
func.func @add_mul_exp_chain(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                             %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
                             %c: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: %[[RESULT:.*]] = ttl.compute
  // CHECK-SAME: ins(%{{.*}}, %{{.*}}, %{{.*}} : {{.*}}) outs(%{{.*}} : {{.*}})
  // CHECK-SAME: {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]}
  // CHECK: ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>, %[[C:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[DST0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[DTOK0:.*]], %[[DTILE0:.*]] = ttl.copy_tile %[[A]], %[[C0]], %[[DST0]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[C1:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[DST1:.*]] = arith.constant 1 : index
  // CHECK-NEXT: %[[DTOK1:.*]], %[[DTILE1:.*]] = ttl.copy_tile %[[B]], %[[C1]], %[[DST1]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[ADD:.*]] = ttl.tile_add %[[DTILE0]], %[[DTILE1]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[C2:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[DST2:.*]] = arith.constant 2 : index
  // CHECK-NEXT: %[[DTOK2:.*]], %[[DTILE2:.*]] = ttl.copy_tile %[[C]], %[[C2]], %[[DST2]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[MUL:.*]] = ttl.tile_mul %[[ADD]], %[[DTILE2]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[EXP:.*]] = ttl.tile_exp %[[MUL]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: ttl.yield %[[EXP]] : !ttcore.tile<32x32, f32>
  // CHECK: }
  // CHECK-NOT: dst_idx
  // CHECK-NOT: tile_batch_size
  %result = ttl.compute
      ins(%a_cb, %b_cb, %c_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                tensor<2x2x!ttcore.tile<32x32, f32>>,
                                tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %c_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %sum, %c_tile : !ttcore.tile<32x32, f32>
    %exp = ttl.tile_exp %mul : !ttcore.tile<32x32, f32>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: return %[[RESULT]]
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

