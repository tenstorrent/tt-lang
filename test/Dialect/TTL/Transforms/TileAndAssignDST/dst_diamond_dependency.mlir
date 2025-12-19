// Summary: Diamond-shaped use to ensure allocator keeps shared operand live.
// RUN: ttlang-opt %s --ttl-tile-and-assign-dst --canonicalize --cse --split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: verify that a value used by multiple tile ops is copied once and
// remains valid across both uses (no clobbering of live DST registers).
// CHECK-LABEL: func.func @diamond_two_uses
func.func @diamond_two_uses(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                           %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
                           %c: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: %[[RES:.*]] = ttl.compute
  // CHECK: ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>, %[[C:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT: %[[DTOKA:.*]], %[[DTILEA:.*]] = ttl.copy_tile %[[A]], %{{.*}}, %{{.*}} : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[DTOKB:.*]], %[[DTILEB:.*]] = ttl.copy_tile %[[B]], %{{.*}}, %{{.*}} : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[SUM:.*]] = ttl.tile_add %[[DTILEA]], %[[DTILEB]] {dst_idx = 2 : i32} : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[DTOKC:.*]], %[[DTILEC:.*]] = ttl.copy_tile %[[C]], %{{.*}}, %{{.*}} : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[DIFF:.*]] = ttl.tile_sub %[[DTILEA]], %[[DTILEC]] {dst_idx = 4 : i32} : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: %[[FINAL:.*]] = ttl.tile_add %[[SUM]], %[[DIFF]] {dst_idx = 5 : i32} : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: ttl.yield %[[FINAL]] : !ttcore.tile<32x32, f32>
  // CHECK: } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK: return %[[RES]]
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
    %diff = ttl.tile_sub %a_tile, %c_tile : !ttcore.tile<32x32, f32>
    %final = ttl.tile_add %sum, %diff : !ttcore.tile<32x32, f32>
    ttl.yield %final : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}
