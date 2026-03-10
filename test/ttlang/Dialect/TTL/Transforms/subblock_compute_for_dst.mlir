// Tests for ttl-subblock-compute-for-dst pass: partitioning ttl.compute into
// DST-sized subblocks. Verifies that ttl-assign-dst computes unroll_factor and
// ttl-subblock-compute-for-dst partitions the compute into subblocks.
// Multi-dimensional tensors are tiled across multiple dimensions.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}))' --split-input-file | FileCheck %s --check-prefix=ASSIGN
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8},ttl-subblock-compute-for-dst))' --split-input-file | FileCheck %s --check-prefix=TILED
// Actual DST capacity (no override) for broadcast tests where FPU detection matters:
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst))' --split-input-file | FileCheck %s --check-prefix=BCAST-ASSIGN
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst,ttl-subblock-compute-for-dst))' --split-input-file | FileCheck %s --check-prefix=BCAST-TILED

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 1x8 tensor with unary chain (1 DST register per iteration).
// DST capacity=8, dstPerIteration=1, totalTiles=8.
// unroll_factor = min(8/1, 8) = 8 = totalTiles -> attribute set but no
// DST subblock partitioning (all tiles fit in one subblock).
// ASSIGN-LABEL: func.func @no_tiling_when_all_fit
// ASSIGN:         ttl.compute
// ASSIGN-SAME:    ttl.unroll_factor = 8 : i64
// TILED-LABEL:  func.func @no_tiling_when_all_fit
// TILED-NOT:    scf.for
// TILED:        ttl.compute
// TILED-SAME:   ttl.full_linearization_strides
// TILED-SAME:   ttl.unroll_factor = 8 : i64
func.func @no_tiling_when_all_fit(%a: tensor<1x8x!ttcore.tile<32x32, f32>>)
    -> tensor<1x8x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x8x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 8], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 8], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x8x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 8], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x8x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<1x8x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 8], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x8x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb1 : <[1, 8], !ttcore.tile<32x32, f32>, 2> -> tensor<1x8x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb : tensor<1x8x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x8x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %exp = ttl.tile_exp %a_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %reserve : !ttcore.tile<32x32, f32>, tensor<1x8x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x8x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<1x8x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 1x8 tensor with FPU binary op (1 DST register per iteration since
// both operands are block args). DST capacity=8, dstPerIteration=1,
// totalTiles=8. unroll_factor = min(8/1, 8) = 8 = totalTiles -> attribute
// set but no DST subblock partitioning (all tiles fit in one subblock).
// ASSIGN-LABEL: func.func @tile_binary_1x8
// ASSIGN:         ttl.compute
// ASSIGN-SAME:    ttl.unroll_factor = 8 : i64
// TILED-LABEL:  func.func @tile_binary_1x8
// TILED-NOT:    scf.for
// TILED:        ttl.compute
// TILED-SAME:   ttl.full_linearization_strides
// TILED-SAME:   ttl.unroll_factor = 8 : i64
func.func @tile_binary_1x8(
    %a: tensor<1x8x!ttcore.tile<32x32, f32>>,
    %b: tensor<1x8x!ttcore.tile<32x32, f32>>)
    -> tensor<1x8x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x8x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 8], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 8], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 8], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x8x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 8], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x8x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<1x8x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 8], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x8x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<1x8x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 8], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x8x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb2 : <[1, 8], !ttcore.tile<32x32, f32>, 2> -> tensor<1x8x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x8x!ttcore.tile<32x32, f32>>, tensor<1x8x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x8x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %sum, %reserve : !ttcore.tile<32x32, f32>, tensor<1x8x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x8x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<1x8x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 2x8 multi-dimensional tensor with unary (1 DST register per
// iteration). DST capacity=8, dstPerIteration=1, totalTiles=16.
// unroll_factor = min(8/1, 16) = 8.
// Multi-dim tiling: tileSizes=[1,8], product=8. Loop on dim 0 (0 to 2 step 1).
// Inner compute on tensor<1x8x...>.
// ASSIGN-LABEL: func.func @tile_multidim_2x8
// ASSIGN:         ttl.compute
// ASSIGN-SAME:    ttl.unroll_factor = 8 : i64
// TILED-LABEL:  func.func @tile_multidim_2x8
// TILED:        %[[C0:.*]] = arith.constant 0 : index
// TILED-NEXT:   %[[C2:.*]] = arith.constant 2 : index
// TILED-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// TILED-NEXT:   scf.for %[[IV:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// TILED-NEXT:     {{.*}} = tensor.extract_slice {{.*}}[%[[IV]], 0] [1, 8] [1, 1] : tensor<2x8x!ttcore.tile<32x32, f32>> to tensor<1x8x!ttcore.tile<32x32, f32>>
// TILED-NEXT:     {{.*}} = tensor.extract_slice {{.*}}[%[[IV]], 0] [1, 8] [1, 1] : tensor<2x8x!ttcore.tile<32x32, f32>> to tensor<1x8x!ttcore.tile<32x32, f32>>
// TILED-NEXT:     {{.*}} = ttl.compute
// TILED-SAME:     tensor<1x8x!ttcore.tile<32x32, f32>>
// TILED-SAME:     ttl.full_linearization_strides
// TILED:            ttl.linearized_index
// Stride 8 for dim 0: arith.muli(iv, 8) then arith.addi.
// TILED:            arith.muli %[[IV]],
// TILED-NEXT:       arith.addi
// TILED:            ttl.tile_exp
// TILED-NEXT:       ttl.tile_store
// TILED-NEXT:       ttl.yield
// TILED-NEXT:     } -> tensor<1x8x!ttcore.tile<32x32, f32>>
// TILED-NEXT:   }
func.func @tile_multidim_2x8(%a: tensor<2x8x!ttcore.tile<32x32, f32>>)
    -> tensor<2x8x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x8x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 8], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 8], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x8x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 8], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x8x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<2x8x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 8], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x8x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb1 : <[2, 8], !ttcore.tile<32x32, f32>, 2> -> tensor<2x8x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb : tensor<2x8x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x8x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %exp = ttl.tile_exp %a_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %reserve : !ttcore.tile<32x32, f32>, tensor<2x8x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x8x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x8x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 2x4 multi-dimensional tensor where all tiles fit in DST.
// DST capacity=8, dstPerIteration=1, totalTiles=8.
// unroll_factor = min(8/1, 8) = 8 = totalTiles -> no DST subblock
// partitioning needed.
// ASSIGN-LABEL: func.func @no_subblocking_multidim
// ASSIGN:         ttl.compute
// ASSIGN-SAME:    ttl.unroll_factor = 8 : i64
// TILED-LABEL:  func.func @no_subblocking_multidim
// TILED-NOT:    scf.for
// TILED:        ttl.compute
// TILED-SAME:   ttl.full_linearization_strides
// TILED-SAME:   ttl.unroll_factor = 8 : i64
func.func @no_subblocking_multidim(%a: tensor<2x4x!ttcore.tile<32x32, f32>>)
    -> tensor<2x4x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x4x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 4], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x4x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<2x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x4x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb1 : <[2, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<2x4x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb : tensor<2x4x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x4x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %exp = ttl.tile_exp %a_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %reserve : !ttcore.tile<32x32, f32>, tensor<2x4x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x4x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x4x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 4x4 square tensor with unary (1 DST register per iteration).
// DST capacity=8, dstPerIteration=1, totalTiles=16.
// unroll_factor = min(8/1, 16) = 8.
// Multi-dim tiling: tileSizes=[2,4], product=8. Loop on dim 0 (0 to 4 step 2).
// Inner compute on tensor<2x4x...>, two subblocks of 8 tiles each.
// ASSIGN-LABEL: func.func @subblock_multidim_4x4
// ASSIGN:         ttl.compute
// ASSIGN-SAME:    ttl.unroll_factor = 8 : i64
// TILED-LABEL:  func.func @subblock_multidim_4x4
// TILED:        %[[C0:.*]] = arith.constant 0 : index
// TILED-NEXT:   %[[C4:.*]] = arith.constant 4 : index
// TILED-NEXT:   %[[C2:.*]] = arith.constant 2 : index
// TILED-NEXT:   scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C2]] {
// TILED-NEXT:     {{.*}} = tensor.extract_slice {{.*}}[%[[IV]], 0] [2, 4] [1, 1] : tensor<4x4x!ttcore.tile<32x32, f32>> to tensor<2x4x!ttcore.tile<32x32, f32>>
// TILED-NEXT:     {{.*}} = tensor.extract_slice {{.*}}[%[[IV]], 0] [2, 4] [1, 1] : tensor<4x4x!ttcore.tile<32x32, f32>> to tensor<2x4x!ttcore.tile<32x32, f32>>
// TILED-NEXT:     {{.*}} = ttl.compute
// TILED-SAME:     tensor<2x4x!ttcore.tile<32x32, f32>>
// TILED-SAME:     ttl.full_linearization_strides
// TILED:            ttl.linearized_index
// Stride 4 for dim 0: arith.muli(iv, 4) then arith.addi.
// TILED:            arith.muli %[[IV]],
// TILED-NEXT:       arith.addi
// TILED:            ttl.tile_exp
// TILED-NEXT:       ttl.tile_store
// TILED-NEXT:       ttl.yield
// TILED-NEXT:     } -> tensor<2x4x!ttcore.tile<32x32, f32>>
// TILED-NEXT:   }
func.func @subblock_multidim_4x4(%a: tensor<4x4x!ttcore.tile<32x32, f32>>)
    -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb : tensor<4x4x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<4x4x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %exp = ttl.tile_exp %a_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %reserve : !ttcore.tile<32x32, f32>, tensor<4x4x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<4x4x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 2x4 multi-dimensional tensor with FPU binary op (1 DST register
// per iteration since both operands are block args). DST capacity=8,
// dstPerIteration=1, totalTiles=8. unroll_factor = min(8/1, 8) = 8 =
// totalTiles -> attribute set but no DST subblock partitioning.
// ASSIGN-LABEL: func.func @no_subblocking_binary
// ASSIGN:         ttl.compute
// ASSIGN-SAME:    ttl.unroll_factor = 8 : i64
// TILED-LABEL:  func.func @no_subblocking_binary
// TILED-NOT:    scf.for
// TILED:        ttl.compute
// TILED-SAME:   ttl.full_linearization_strides
// TILED-SAME:   ttl.unroll_factor = 8 : i64
func.func @no_subblocking_binary(
    %a: tensor<2x4x!ttcore.tile<32x32, f32>>,
    %b: tensor<2x4x!ttcore.tile<32x32, f32>>)
    -> tensor<2x4x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x4x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 4], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 4], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x4x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x4x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x4x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb2 : <[2, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<2x4x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x4x!ttcore.tile<32x32, f32>>, tensor<2x4x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x4x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %sum, %reserve : !ttcore.tile<32x32, f32>, tensor<2x4x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x4x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x4x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 3x3 tensor with unary -- totalTiles not evenly divisible by
// initial unroll_factor. DST capacity=8, dstPerIteration=1, totalTiles=9.
// unroll_factor = min(8/1, 9) = 8.
// Multi-dim tiling: best tileSizes=[1,3] (product=3, largest evenly-dividing
// subblock <= 8). Loop on dim 0 (0 to 3 step 1), 3 subblocks of 3 tiles.
// ASSIGN-LABEL: func.func @tile_multidim_remainder_3x3
// ASSIGN:         ttl.compute
// ASSIGN-SAME:    ttl.unroll_factor = 8 : i64
// TILED-LABEL:  func.func @tile_multidim_remainder_3x3
// TILED:        %[[C0:.*]] = arith.constant 0 : index
// TILED-NEXT:   %[[C3:.*]] = arith.constant 3 : index
// TILED-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// TILED-NEXT:   scf.for %[[IV:.*]] = %[[C0]] to %[[C3]] step %[[C1]] {
// TILED-NEXT:     {{.*}} = tensor.extract_slice {{.*}}[%[[IV]], 0] [1, 3] [1, 1] : tensor<3x3x!ttcore.tile<32x32, f32>> to tensor<1x3x!ttcore.tile<32x32, f32>>
// TILED-NEXT:     {{.*}} = tensor.extract_slice {{.*}}[%[[IV]], 0] [1, 3] [1, 1] : tensor<3x3x!ttcore.tile<32x32, f32>> to tensor<1x3x!ttcore.tile<32x32, f32>>
// TILED-NEXT:     {{.*}} = ttl.compute
// TILED-SAME:     tensor<1x3x!ttcore.tile<32x32, f32>>
// TILED-SAME:     ttl.full_linearization_strides
// TILED:            ttl.linearized_index
// Stride 3 for dim 0: arith.muli(iv, 3) then arith.addi.
// TILED:            arith.muli %[[IV]],
// TILED-NEXT:       arith.addi
// TILED:            ttl.tile_exp
// TILED-NEXT:       ttl.tile_store
// TILED-NEXT:       ttl.yield
// TILED-NEXT:     } -> tensor<1x3x!ttcore.tile<32x32, f32>>
// TILED-NEXT:   }
func.func @tile_multidim_remainder_3x3(%a: tensor<3x3x!ttcore.tile<32x32, f32>>)
    -> tensor<3x3x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<3x3x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[3, 3], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[3, 3], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<3x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[3, 3], !ttcore.tile<32x32, f32>, 2>) -> tensor<3x3x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<3x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[3, 3], !ttcore.tile<32x32, f32>, 2>) -> tensor<3x3x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb1 : <[3, 3], !ttcore.tile<32x32, f32>, 2> -> tensor<3x3x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb : tensor<3x3x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<3x3x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %exp = ttl.tile_exp %a_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %reserve : !ttcore.tile<32x32, f32>, tensor<3x3x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<3x3x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<3x3x!ttcore.tile<32x32, f32>>
}

// -----

#map_identity = affine_map<(d0, d1) -> (d0, d1)>
#map_col_bcast = affine_map<(d0, d1) -> (d0, 0)>

// Purpose: 4x4 output with col-broadcast input B (4x1). Exercises
// mapOffsetsAndSizes with broadcast indexing map containing a constant
// expression for the column dimension. Verifies that the broadcast dim
// retains its original size (1) in extract_slice rather than being
// incorrectly set to the iteration-domain size (4).
// The two inputs have different indexing maps (identity vs broadcast), so
// FPU binary detection is skipped and the op uses the SFPU path with
// dstPerIteration=2 (copy_tile for both operands).
// DST capacity=4 (f32, actual), dstPerIteration=2, totalTiles=16.
// unroll_factor = min(4/2, 16) = 2.
// Multi-dim tiling: tileSizes=[1,2], product=2.
// Two nested loops: dim 0 (0 to 4 step 1), dim 1 (0 to 4 step 2).
//
// Uses BCAST-ASSIGN/BCAST-TILED prefixes.
// BCAST-ASSIGN-LABEL: func.func @subblock_broadcast_col
// BCAST-ASSIGN:         ttl.compute
// BCAST-ASSIGN-SAME:    ttl.unroll_factor = 2 : i64
// BCAST-TILED-LABEL:  func.func @subblock_broadcast_col
// BCAST-TILED:        scf.for %[[IV0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// BCAST-TILED-NEXT:     scf.for %[[IV1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// A: identity map, 1x2 subblock slice.
// BCAST-TILED-NEXT:       {{.*}} = tensor.extract_slice {{.*}}[%[[IV0]], %[[IV1]]] [1, 2] [1, 1] : tensor<4x4x!ttcore.tile<32x32, f32>> to tensor<1x2x!ttcore.tile<32x32, f32>>
// B: col broadcast map -- broadcast dim (col) keeps original size 1.
// BCAST-TILED-NEXT:       {{.*}} = tensor.extract_slice {{.*}}[%[[IV0]], 0] [1, 1] [1, 1] : tensor<4x1x!ttcore.tile<32x32, f32>> to tensor<1x1x!ttcore.tile<32x32, f32>>
// Output: identity map, 1x2 subblock slice.
// BCAST-TILED-NEXT:       {{.*}} = tensor.extract_slice {{.*}}[%[[IV0]], %[[IV1]]] [1, 2] [1, 1] : tensor<4x4x!ttcore.tile<32x32, f32>> to tensor<1x2x!ttcore.tile<32x32, f32>>
// Tiled compute has broadcast-aware operand shapes.
// BCAST-TILED-NEXT:       {{.*}} = ttl.compute
// BCAST-TILED-SAME:       tensor<1x2x!ttcore.tile<32x32, f32>>
// BCAST-TILED-SAME:       tensor<1x1x!ttcore.tile<32x32, f32>>
// BCAST-TILED-SAME:       ttl.full_linearization_strides
// BCAST-TILED:              ttl.tile_add
// BCAST-TILED-NEXT:         ttl.tile_store
// BCAST-TILED-NEXT:         ttl.yield
// BCAST-TILED-NEXT:       } -> tensor<1x2x!ttcore.tile<32x32, f32>>
// BCAST-TILED:          } {ttl.subblock_stride
// BCAST-TILED:        } {ttl.subblock_stride
func.func @subblock_broadcast_col(
    %a: tensor<4x4x!ttcore.tile<32x32, f32>>,
    %b: tensor<4x1x!ttcore.tile<32x32, f32>>)
    -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<4x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb2 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<4x4x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map_identity, #map_col_bcast, #map_identity],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %sum, %reserve : !ttcore.tile<32x32, f32>, tensor<4x4x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<4x4x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<4x4x!ttcore.tile<32x32, f32>>
}
