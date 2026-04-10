// Summary: verify DST assignment and copy insertion on 4D tensors
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}),canonicalize,cse)' --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8 separate-output-region=1}),canonicalize,cse)' --split-input-file | FileCheck %s --check-prefix=SEPARATE
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8 enable-fpu-binary-ops=0}),canonicalize,cse)' --split-input-file | FileCheck %s --check-prefix=SFPU

// Verify no placeholder copies remain in final IR
// CHECK-NOT: placeholder

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Purpose: test that FPU binary add on 4D tensors works correctly with iter_index.
// FPU binary: both operands are block args, so no copy_tile needed.
// iter_index ops provide iteration coordinates for CB indexing.
// CHECK-LABEL: func.func @add_4d
func.func @add_4d(%a: tensor<3x6x4x2x!ttcore.tile<32x32, f32>>,
                  %b: tensor<3x6x4x2x!ttcore.tile<32x32, f32>>)
    -> tensor<3x6x4x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<3x6x4x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[3, 6, 4, 2], !ttcore.tile<32x32, f32>, 1>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 1} : !ttl.cb<[3, 6, 4, 2], !ttcore.tile<32x32, f32>, 1>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[3, 6, 4, 2], !ttcore.tile<32x32, f32>, 1>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<3x6x4x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[3, 6, 4, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<3x6x4x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<3x6x4x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[3, 6, 4, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<3x6x4x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<3x6x4x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[3, 6, 4, 2], !ttcore.tile<32x32, f32>, 1>) -> tensor<3x6x4x2x!ttcore.tile<32x32, f32>>

// CHECK: %[[RES:.*]] = ttl.compute
// CHECK-NEXT: ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:   %[[I0:.*]] = ttl.iter_index 0 : index
// CHECK-NEXT:   %[[I1:.*]] = ttl.iter_index 1 : index
// CHECK-NEXT:   %[[I2:.*]] = ttl.iter_index 2 : index
// CHECK-NEXT:   %[[I3:.*]] = ttl.iter_index 3 : index
// FPU binary: no copy_tile needed.
// CHECK-NOT:  ttl.copy_tile
// CHECK:      %[[ADD:.*]] = ttl.tile_add %[[A]], %[[B]] into dst[%c0] {ttl.fpu_binary}
// CHECK-NEXT: ttl.tile_store %[[ADD]], %{{.*}}[%[[I0]], %[[I1]], %[[I2]], %[[I3]]]
// CHECK-NEXT: ttl.yield
// SEPARATE-LABEL: func.func @add_4d
// SEPARATE:      %[[ADDS:.*]] = ttl.tile_add {{.*}} into dst[%c0] {ttl.fpu_binary}
// SEPARATE:      ttl.tile_store
// SEPARATE-NEXT: ttl.yield
//
// SFPU path: copy_tile uses iter_index for CB indexing
// SFPU-LABEL: func.func @add_4d
// SFPU:         ttl.compute
// SFPU:         ^bb0(%[[AS:.*]]: !ttcore.tile<32x32, f32>, %[[BS:.*]]: !ttcore.tile<32x32, f32>, %{{.*}}: !ttcore.tile<32x32, f32>):
// SFPU-NEXT:      %[[SI0:.*]] = ttl.iter_index 0 : index
// SFPU-NEXT:      %[[SI1:.*]] = ttl.iter_index 1 : index
// SFPU-NEXT:      %[[SI2:.*]] = ttl.iter_index 2 : index
// SFPU-NEXT:      %[[SI3:.*]] = ttl.iter_index 3 : index
// SFPU:           %{{.*}}, %[[DA:.*]] = ttl.copy_tile %[[AS]][%[[SI0]], %[[SI1]], %[[SI2]], %[[SI3]]] into dst[%c0]
// SFPU-NEXT:      %{{.*}}, %[[DB:.*]] = ttl.copy_tile %[[BS]][%[[SI0]], %[[SI1]], %[[SI2]], %[[SI3]]] into dst[%c1]
// SFPU-NEXT:      %[[ADDS:.*]] = ttl.tile_add %[[DA]], %[[DB]] into dst[%c0]
// SFPU:           ttl.tile_store %[[ADDS]], %{{.*}}[%[[SI0]], %[[SI1]], %[[SI2]], %[[SI3]]]
// SFPU-NEXT:      ttl.yield
  %out_view = ttl.cb_reserve %cb2 : <[3, 6, 4, 2], !ttcore.tile<32x32, f32>, 1> -> tensor<3x6x4x2x!ttcore.tile<32x32, f32>>
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<3x6x4x2x!ttcore.tile<32x32, f32>>,
                         tensor<3x6x4x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<3x6x4x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel", "parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %k = ttl.iter_index 2 : index
    %l = ttl.iter_index 3 : index
    %c0 = arith.constant 0 : index
    %sum = ttl.tile_add %a_tile, %b_tile into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %sum, %out_view[%i, %j, %k, %l] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<3x6x4x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<3x6x4x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<3x6x4x2x!ttcore.tile<32x32, f32>>
}
