// Tests for ttl-subblock-compute-for-dst pass: partitioning ttl.compute into
// DST-sized subblocks. Verifies that ttl-assign-dst computes unroll_factor and
// ttl-subblock-compute-for-dst partitions the compute into subblocks.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8}))' --split-input-file | FileCheck %s --check-prefix=ASSIGN
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8},ttl-subblock-compute-for-dst))' --split-input-file | FileCheck %s --check-prefix=TILED

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 1x8 tensor with unary chain (1 DST register per iteration).
// DST capacity=8, dstPerIteration=1, totalTiles=8.
// unroll_factor = min(8/1, 8) = 8 = totalTiles -> attribute set but no tiling.
// ASSIGN-LABEL: func.func @no_tiling_when_all_fit
// ASSIGN: ttl.unroll_factor = 8
// TILED-LABEL: func.func @no_tiling_when_all_fit
// TILED-NOT: scf.for
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

// Purpose: 1x8 tensor with binary op (2 DST registers per iteration: 1 for
// copy_tile of lhs, 1 for add result). DST capacity=8, dstPerIteration=2,
// totalTiles=8. unroll_factor = min(8/2, 8) = 4.
// After tiling: scf.for with step 4, inner compute on tensor<1x4x...>.
// ASSIGN-LABEL: func.func @tile_binary_1x8
// ASSIGN: ttl.unroll_factor = 4
// TILED-LABEL: func.func @tile_binary_1x8
// TILED: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %[[STEP:.*]]
// TILED:   tensor.extract_slice
// TILED:   tensor.extract_slice
// TILED:   tensor.extract_slice
// TILED:   ttl.compute
// TILED:     ttl.tile_add
// TILED:     ttl.tile_store
// TILED:     ttl.yield
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
