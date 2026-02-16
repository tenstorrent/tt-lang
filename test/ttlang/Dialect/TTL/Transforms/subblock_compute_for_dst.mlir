// Tests for ttl-subblock-compute-for-dst pass: partitioning ttl.compute into
// DST-sized subblocks. Verifies that ttl-assign-dst computes unroll_factor and
// ttl-subblock-compute-for-dst partitions the compute into subblocks.
// Multi-dimensional tensors are flattened to 1D before partitioning.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}))' --split-input-file | FileCheck %s --check-prefix=ASSIGN
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8},ttl-subblock-compute-for-dst))' --split-input-file | FileCheck %s --check-prefix=TILED

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
    ttl.yield %exp : !ttcore.tile<32x32, f32>
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
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<1x8x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<1x8x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 2x8 multi-dimensional tensor with unary (1 DST register per
// iteration). DST capacity=8, dstPerIteration=1, totalTiles=16.
// unroll_factor = min(8/1, 16) = 8.
// Iteration space flattening: tensor<2x8x...> -> tensor<16x...> because
// outer dim (2) > 1 -- without flattening, innermost-dim-only partitioning
// would produce subblocks of 2*8=16 tiles instead of 8.
// After DST subblock partitioning: scf.for with step 8, inner compute on
// tensor<8x...>.
// ASSIGN-LABEL: func.func @flatten_and_tile_2x8
// ASSIGN:         ttl.compute
// ASSIGN-SAME:    ttl.unroll_factor = 8 : i64
// TILED-LABEL:  func.func @flatten_and_tile_2x8
// TILED:        {{.*}} = tensor.collapse_shape {{.*}} : tensor<2x8x!ttcore.tile<32x32, f32>> into tensor<16x!ttcore.tile<32x32, f32>>
// TILED-NEXT:   {{.*}} = tensor.collapse_shape {{.*}} : tensor<2x8x!ttcore.tile<32x32, f32>> into tensor<16x!ttcore.tile<32x32, f32>>
// TILED-NEXT:   %[[C0:.*]] = arith.constant 0 : index
// TILED-NEXT:   %[[C16:.*]] = arith.constant 16 : index
// TILED-NEXT:   %[[C8:.*]] = arith.constant 8 : index
// TILED-NEXT:   scf.for %[[IV:.*]] = %[[C0]] to %[[C16]] step %[[C8]] {
// TILED-NEXT:     {{.*}} = tensor.extract_slice {{.*}}[%[[IV]]] [8] [1] : tensor<16x!ttcore.tile<32x32, f32>> to tensor<8x!ttcore.tile<32x32, f32>>
// TILED-NEXT:     {{.*}} = tensor.extract_slice {{.*}}[%[[IV]]] [8] [1] : tensor<16x!ttcore.tile<32x32, f32>> to tensor<8x!ttcore.tile<32x32, f32>>
// TILED-NEXT:     {{.*}} = ttl.compute
// TILED-SAME:     tensor<8x!ttcore.tile<32x32, f32>>
// TILED:            ttl.tile_exp
// TILED-NEXT:       ttl.tile_store
// TILED-NEXT:       ttl.yield
// TILED-NEXT:     } -> tensor<8x!ttcore.tile<32x32, f32>>
// TILED-NEXT:   }
func.func @flatten_and_tile_2x8(%a: tensor<2x8x!ttcore.tile<32x32, f32>>)
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
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<2x8x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x8x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 2x4 multi-dimensional tensor where all tiles fit in DST.
// DST capacity=8, dstPerIteration=1, totalTiles=8.
// unroll_factor = min(8/1, 8) = 8 = totalTiles -> no DST subblock
// partitioning needed. No flattening either (tiling won't happen).
// ASSIGN-LABEL: func.func @no_subblocking_multidim
// ASSIGN:         ttl.compute
// ASSIGN-SAME:    ttl.unroll_factor = 8 : i64
// TILED-LABEL:  func.func @no_subblocking_multidim
// TILED-NOT:    tensor.collapse_shape
// TILED-NOT:    scf.for
// TILED:        ttl.compute
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
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<2x4x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x4x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 4x4 square tensor with unary (1 DST register per iteration).
// DST capacity=8, dstPerIteration=1, totalTiles=16.
// unroll_factor = min(8/1, 16) = 8.
// Iteration space flattening: tensor<4x4x...> -> tensor<16x...> because
// outer dim (4) > 1.
// After DST subblock partitioning: scf.for with step 8, two subblocks of
// 8 tiles each.
// ASSIGN-LABEL: func.func @flatten_and_subblock_4x4
// ASSIGN:         ttl.compute
// ASSIGN-SAME:    ttl.unroll_factor = 8 : i64
// TILED-LABEL:  func.func @flatten_and_subblock_4x4
// TILED:        {{.*}} = tensor.collapse_shape {{.*}} : tensor<4x4x!ttcore.tile<32x32, f32>> into tensor<16x!ttcore.tile<32x32, f32>>
// TILED-NEXT:   {{.*}} = tensor.collapse_shape {{.*}} : tensor<4x4x!ttcore.tile<32x32, f32>> into tensor<16x!ttcore.tile<32x32, f32>>
// TILED-NEXT:   %[[C0:.*]] = arith.constant 0 : index
// TILED-NEXT:   %[[C16:.*]] = arith.constant 16 : index
// TILED-NEXT:   %[[C8:.*]] = arith.constant 8 : index
// TILED-NEXT:   scf.for %[[IV:.*]] = %[[C0]] to %[[C16]] step %[[C8]] {
// TILED-NEXT:     {{.*}} = tensor.extract_slice {{.*}}[%[[IV]]] [8] [1] : tensor<16x!ttcore.tile<32x32, f32>> to tensor<8x!ttcore.tile<32x32, f32>>
// TILED-NEXT:     {{.*}} = tensor.extract_slice {{.*}}[%[[IV]]] [8] [1] : tensor<16x!ttcore.tile<32x32, f32>> to tensor<8x!ttcore.tile<32x32, f32>>
// TILED-NEXT:     {{.*}} = ttl.compute
// TILED-SAME:     tensor<8x!ttcore.tile<32x32, f32>>
// TILED:            ttl.tile_exp
// TILED-NEXT:       ttl.tile_store
// TILED-NEXT:       ttl.yield
// TILED-NEXT:     } -> tensor<8x!ttcore.tile<32x32, f32>>
// TILED-NEXT:   }
func.func @flatten_and_subblock_4x4(%a: tensor<4x4x!ttcore.tile<32x32, f32>>)
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
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<4x4x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 2x4 multi-dimensional tensor with FPU binary op (1 DST register
// per iteration since both operands are block args). DST capacity=8,
// dstPerIteration=1, totalTiles=8. unroll_factor = min(8/1, 8) = 8 =
// totalTiles -> attribute set but no DST subblock partitioning.
// No flattening either (tiling won't happen).
// ASSIGN-LABEL: func.func @flatten_and_subblock_binary
// ASSIGN:         ttl.compute
// ASSIGN-SAME:    ttl.unroll_factor = 8 : i64
// TILED-LABEL:  func.func @flatten_and_subblock_binary
// TILED-NOT:    tensor.collapse_shape
// TILED-NOT:    scf.for
// TILED:        ttl.compute
// TILED-SAME:   ttl.unroll_factor = 8 : i64
func.func @flatten_and_subblock_binary(
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
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x4x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x4x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 3x3 tensor with unary -- totalTiles not evenly divisible by
// unroll_factor, requiring remainder handling after flattening.
// DST capacity=8, dstPerIteration=1, totalTiles=9.
// unroll_factor = min(8/1, 9) = 8.
// Iteration space flattening: tensor<3x3x...> -> tensor<9x...>.
// After DST subblock partitioning: scf.for with step 8 over 9 tiles.
// First subblock: 8 tiles. Second subblock: 1 tile (remainder).
// The loop uses arith.minsi to compute min(8, 9 - iv) for the block size.
// ASSIGN-LABEL: func.func @flatten_with_remainder
// ASSIGN:         ttl.compute
// ASSIGN-SAME:    ttl.unroll_factor = 8 : i64
// TILED-LABEL:  func.func @flatten_with_remainder
// TILED:        {{.*}} = tensor.collapse_shape {{.*}} : tensor<3x3x!ttcore.tile<32x32, f32>> into tensor<9x!ttcore.tile<32x32, f32>>
// TILED-NEXT:   {{.*}} = tensor.collapse_shape {{.*}} : tensor<3x3x!ttcore.tile<32x32, f32>> into tensor<9x!ttcore.tile<32x32, f32>>
// TILED-NEXT:   %[[C0:.*]] = arith.constant 0 : index
// TILED-NEXT:   %[[C9:.*]] = arith.constant 9 : index
// TILED-NEXT:   %[[C8:.*]] = arith.constant 8 : index
// TILED-NEXT:   scf.for %[[IV:.*]] = %[[C0]] to %[[C9]] step %[[C8]] {
// TILED-NEXT:     %[[C9_1:.*]] = arith.constant 9 : index
// TILED-NEXT:     %[[REM:.*]] = arith.subi %[[C9_1]], %[[IV]] : index
// TILED-NEXT:     %[[C8_2:.*]] = arith.constant 8 : index
// TILED-NEXT:     %[[BLOCK:.*]] = arith.minsi %[[C8_2]], %[[REM]] : index
// TILED-NEXT:     {{.*}} = tensor.extract_slice {{.*}}[%[[IV]]] [%[[BLOCK]]] [1] : tensor<9x!ttcore.tile<32x32, f32>> to tensor<?x!ttcore.tile<32x32, f32>>
// TILED-NEXT:     {{.*}} = tensor.extract_slice {{.*}}[%[[IV]]] [%[[BLOCK]]] [1] : tensor<9x!ttcore.tile<32x32, f32>> to tensor<?x!ttcore.tile<32x32, f32>>
// TILED-NEXT:     {{.*}} = ttl.compute
// TILED-SAME:     tensor<?x!ttcore.tile<32x32, f32>>
// TILED:            ttl.tile_exp
// TILED-NEXT:       ttl.tile_store
// TILED-NEXT:       ttl.yield
// TILED-NEXT:     } -> tensor<?x!ttcore.tile<32x32, f32>>
// TILED-NEXT:   }
func.func @flatten_with_remainder(%a: tensor<3x3x!ttcore.tile<32x32, f32>>)
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
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<3x3x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<3x3x!ttcore.tile<32x32, f32>>
}
