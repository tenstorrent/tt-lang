// Summary: verify DST assignment and copy insertion on 4D tensors
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}),canonicalize,cse)' --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8 separate-output-region=1}),canonicalize,cse)' --split-input-file | FileCheck %s --check-prefix=SEPARATE
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8 enable-fpu-binary-ops=0}),canonicalize,cse)' --split-input-file | FileCheck %s --check-prefix=SFPU

// Verify no placeholder copies remain in final IR
// CHECK-NOT: placeholder

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Purpose: test that FPU binary add on 4D tensors works without linearized_index.
// FPU binary: both operands are block args, so no copy_tile or linearized_index needed.
// CHECK-LABEL: func.func @add_4d
func.func @add_4d(%a: tensor<3x6x4x2x!ttcore.tile<32x32, f32>>,
                  %b: tensor<3x6x4x2x!ttcore.tile<32x32, f32>>)
    -> tensor<3x6x4x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<3x6x4x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[3, 6, 4, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 1} : !ttl.cb<[3, 6, 4, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[3, 6, 4, 2], !ttcore.tile<32x32, f32>, 1>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<3x6x4x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[3, 6, 4, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<3x6x4x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<3x6x4x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[3, 6, 4, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<3x6x4x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<3x6x4x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[3, 6, 4, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<3x6x4x2x!ttcore.tile<32x32, f32>>

// CHECK: %[[RES:.*]] = ttl.compute
// CHECK-NEXT: ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// FPU binary: no copy_tile needed, no linearized_index needed.
// CHECK-NOT:  ttl.copy_tile
// CHECK-NEXT: %[[ADD:.*]] = ttl.tile_add %[[A]], %[[B]] {dst_idx = 0 : i32, ttl.fpu_binary}
// CHECK-NEXT: ttl.yield %[[ADD]]
// SEPARATE-LABEL: func.func @add_4d
// SEPARATE:      %[[ADDS:.*]] = ttl.tile_add {{.*}} {dst_idx = 0 : i32, ttl.fpu_binary}
// SEPARATE-NEXT: ttl.yield %[[ADDS]]
//
// SFPU path: verify 4D linearization (d0*48 + d1*8 + d2*2 + d3) and copy_tile
// Map appears at module scope, so check before the label.
// SFPU-DAG:     [[$MAP:#map[0-9]*]] = affine_map<(d0, d1, d2, d3) -> (d0 * 48 + d1 * 8 + d2 * 2 + d3)>
// SFPU-LABEL: func.func @add_4d
// SFPU:         ttl.compute
// SFPU:         ^bb0(%[[AS:.*]]: !ttcore.tile<32x32, f32>, %[[BS:.*]]: !ttcore.tile<32x32, f32>, %{{.*}}: !ttcore.tile<32x32, f32>):
// SFPU-NEXT:      %[[LIN:.*]] = ttl.linearized_index [[$MAP]]
// SFPU-NEXT:      %{{.*}}, %[[DA:.*]] = ttl.copy_tile %[[AS]], %[[LIN]], %{{.*}} {dst_idx = 0 : i32}
// SFPU-NEXT:      %{{.*}}, %[[DB:.*]] = ttl.copy_tile %[[BS]], %[[LIN]], %{{.*}} {dst_idx = 1 : i32}
// SFPU-NEXT:      %[[ADDS:.*]] = ttl.tile_add %[[DA]], %[[DB]] {dst_idx = 0 : i32}
// SFPU-NEXT:      ttl.yield %[[ADDS]]
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<3x6x4x2x!ttcore.tile<32x32, f32>>,
                         tensor<3x6x4x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<3x6x4x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel", "parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<3x6x4x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<3x6x4x2x!ttcore.tile<32x32, f32>>
}
