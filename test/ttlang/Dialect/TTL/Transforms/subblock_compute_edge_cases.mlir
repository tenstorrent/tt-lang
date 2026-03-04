// Tests for subblock compute edge cases not covered by the main subblock test.
//
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=4},ttl-subblock-compute-for-dst))' --split-input-file | FileCheck %s --check-prefix=TILED

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 1: 6x6 tensor -> 2D subblocking (both dimensions get outer loops)
// =============================================================================
// DST capacity=4, dstPerIteration=1 (exp unary).
// unroll_factor = min(4/1, 36) = 4. dimSizes=[6,6].
// computeMultiDimSubblockSizes([6,6], 4) = [2,2] (product=4, best fit).
// Both dims need outer loops: dim0 from 0..6 step 2, dim1 from 0..6 step 2.
// This is the ONLY test that exercises 2D nested subblock outer loops.

// TILED-LABEL: func.func @subblock_2d_6x6
// Two nested scf.for loops (order: dim0 then dim1):
// TILED:        scf.for %[[I:.*]] =
// TILED:          scf.for %[[J:.*]] =
// Extract 2x2 subblock:
// TILED:            tensor.extract_slice {{.*}}[%[[I]], %[[J]]] [2, 2] [1, 1]
// TILED:            tensor.extract_slice {{.*}}[%[[I]], %[[J]]] [2, 2] [1, 1]
// TILED:            ttl.compute
// TILED-SAME:       tensor<2x2x!ttcore.tile<32x32, f32>>
// TILED-SAME:       ttl.full_linearization_strides
// TILED:              ttl.tile_exp
// TILED:              ttl.yield
// TILED:            } -> tensor<2x2x!ttcore.tile<32x32, f32>>

func.func @subblock_2d_6x6(%a: tensor<6x6x!ttcore.tile<32x32, f32>>)
    -> tensor<6x6x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<6x6x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[6, 6], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[6, 6], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<6x6x!ttcore.tile<32x32, f32>>, !ttl.cb<[6, 6], !ttcore.tile<32x32, f32>, 2>) -> tensor<6x6x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<6x6x!ttcore.tile<32x32, f32>>, !ttl.cb<[6, 6], !ttcore.tile<32x32, f32>, 2>) -> tensor<6x6x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb1 : <[6, 6], !ttcore.tile<32x32, f32>, 2> -> tensor<6x6x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb : tensor<6x6x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<6x6x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %exp = ttl.tile_exp %a_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %reserve : !ttcore.tile<32x32, f32>, tensor<6x6x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<6x6x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<6x6x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// =============================================================================
// Test 2: Consecutive computes in the same function
// =============================================================================
// Two independent ttl.compute ops on different tensors. Both should be
// independently processed by the subblock pass. The first gets subblocked
// (4x4, unroll=4 -> [1,4] subblock via prefersInner), the second fits
// without subblocking (1x2, totalTiles=2 <= unroll=2).

// TILED-LABEL: func.func @consecutive_computes
// First compute: subblocked with outer loop (dim0: 0..4 step 1)
// TILED:        scf.for
// TILED:          ttl.compute
// TILED-SAME:     tensor<1x4x!ttcore.tile<32x32, f32>>
// TILED-SAME:     ttl.full_linearization_strides
// TILED:            ttl.tile_exp
// TILED:            ttl.yield
// Second compute: no subblocking (fits in DST)
// TILED:        ttl.compute
// TILED-SAME:   tensor<1x2x!ttcore.tile<32x32, f32>>
// TILED-SAME:   ttl.full_linearization_strides
// TILED-SAME:   ttl.unroll_factor = 2
// TILED:          ttl.tile_log
// TILED:          ttl.yield

func.func @consecutive_computes(
    %a: tensor<4x4x!ttcore.tile<32x32, f32>>,
    %b: tensor<1x2x!ttcore.tile<32x32, f32>>)
    -> (tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<1x2x!ttcore.tile<32x32, f32>>) {
  %init0 = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  %init1 = tensor.empty() : tensor<1x2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 17, buffer_factor = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %init0_cb = ttl.attach_cb %init0, %cb1 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb2 : (tensor<1x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x2x!ttcore.tile<32x32, f32>>
  %init1_cb = ttl.attach_cb %init1, %cb3 : (tensor<1x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x2x!ttcore.tile<32x32, f32>>

  %r0 = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %r1 = ttl.cb_reserve %cb3 : <[1, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<1x2x!ttcore.tile<32x32, f32>>

  // First compute: 4x4 exp -> gets subblocked (16 tiles, unroll=4)
  %res0 = ttl.compute
      ins(%a_cb : tensor<4x4x!ttcore.tile<32x32, f32>>)
      outs(%init0_cb : tensor<4x4x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %o0: !ttcore.tile<32x32, f32>):
    %exp = ttl.tile_exp %a_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %r0 : !ttcore.tile<32x32, f32>, tensor<4x4x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // Second compute: 1x2 log -> fits without subblocking (2 tiles <= unroll=2)
  %res1 = ttl.compute
      ins(%b_cb : tensor<1x2x!ttcore.tile<32x32, f32>>)
      outs(%init1_cb : tensor<1x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%b_tile: !ttcore.tile<32x32, f32>, %o1: !ttcore.tile<32x32, f32>):
    %log = ttl.tile_log %b_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %log, %r1 : !ttcore.tile<32x32, f32>, tensor<1x2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<1x2x!ttcore.tile<32x32, f32>>

  func.return %res0, %res1 : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<1x2x!ttcore.tile<32x32, f32>>
}

// -----

#map_identity = affine_map<(d0, d1) -> (d0, d1)>
#map_row_bcast = affine_map<(d0, d1) -> (0, d1)>

// =============================================================================
// Test 3: Row broadcast with 2D subblocking
// =============================================================================
// Exercises mapOffsetsAndSizes where the FIRST result is a constant (0) and
// the second is a dim expression. The col broadcast test in the main file
// covers the reverse case (first dim, second constant).
// DST capacity=4, FPU binary (tile_add both block args) dstPerIteration=1.
// unroll_factor = min(4/1, 36) = 4. subblock = [2,2].
// Both dims tiled: dim0 0..6 step 2, dim1 0..6 step 2.
// B (1x6 row bcast): dim0 constant -> offset=0, size=1; dim1 mapped -> offset=%j, size=2.

// TILED-LABEL: func.func @subblock_row_broadcast
// TILED:        scf.for %[[I:.*]] =
// TILED:          scf.for %[[J:.*]] =
// A (identity): subblock slice
// TILED:            tensor.extract_slice {{.*}}[%[[I]], %[[J]]] [2, 2] [1, 1] : tensor<6x6x!ttcore.tile<32x32, f32>>
// B (row bcast): dim0 always 0 with original size 1, dim1 varies with subblock
// TILED:            tensor.extract_slice {{.*}}[0, %[[J]]] [1, 2] [1, 1] : tensor<1x6x!ttcore.tile<32x32, f32>>
// Output (identity): subblock slice
// TILED:            tensor.extract_slice {{.*}}[%[[I]], %[[J]]] [2, 2] [1, 1] : tensor<6x6x!ttcore.tile<32x32, f32>>
// TILED:            ttl.compute
// TILED-SAME:       tensor<2x2x!ttcore.tile<32x32, f32>>
// TILED-SAME:       tensor<1x2x!ttcore.tile<32x32, f32>>

func.func @subblock_row_broadcast(
    %a: tensor<6x6x!ttcore.tile<32x32, f32>>,
    %b: tensor<1x6x!ttcore.tile<32x32, f32>>)
    -> tensor<6x6x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<6x6x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[6, 6], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 6], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[6, 6], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<6x6x!ttcore.tile<32x32, f32>>, !ttl.cb<[6, 6], !ttcore.tile<32x32, f32>, 2>) -> tensor<6x6x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<1x6x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 6], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x6x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<6x6x!ttcore.tile<32x32, f32>>, !ttl.cb<[6, 6], !ttcore.tile<32x32, f32>, 2>) -> tensor<6x6x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb2 : <[6, 6], !ttcore.tile<32x32, f32>, 2> -> tensor<6x6x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<6x6x!ttcore.tile<32x32, f32>>, tensor<1x6x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<6x6x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map_identity, #map_row_bcast, #map_identity],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %sum, %reserve : !ttcore.tile<32x32, f32>, tensor<6x6x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<6x6x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<6x6x!ttcore.tile<32x32, f32>>
}

// -----

#map_identity = affine_map<(d0, d1) -> (d0, d1)>
#map_scalar_bcast = affine_map<(d0, d1) -> (0, 0)>

// =============================================================================
// Test 4: Scalar broadcast with 2D subblocking
// =============================================================================
// Exercises mapOffsetsAndSizes where BOTH results are constants. The
// broadcast input (1x1) should always extract the full tensor regardless
// of loop IVs. With 2D subblocking, neither outer loop variable appears
// in the broadcast input's extract_slice.
// DST capacity=4, FPU binary dstPerIteration=1. subblock=[2,2].

// TILED-LABEL: func.func @subblock_scalar_broadcast
// TILED:        scf.for %[[I:.*]] =
// TILED:          scf.for %[[J:.*]] =
// A (identity): varies with both loop IVs
// TILED:            tensor.extract_slice {{.*}}[%[[I]], %[[J]]] [2, 2] [1, 1] : tensor<6x6x!ttcore.tile<32x32, f32>>
// C (scalar bcast): always the full 1x1 tensor, no loop IV dependency
// TILED:            tensor.extract_slice {{.*}}[0, 0] [1, 1] [1, 1] : tensor<1x1x!ttcore.tile<32x32, f32>>
// Output (identity): varies with both loop IVs
// TILED:            tensor.extract_slice {{.*}}[%[[I]], %[[J]]] [2, 2] [1, 1] : tensor<6x6x!ttcore.tile<32x32, f32>>
// TILED:            ttl.compute
// TILED-SAME:       tensor<2x2x!ttcore.tile<32x32, f32>>
// TILED-SAME:       tensor<1x1x!ttcore.tile<32x32, f32>>

func.func @subblock_scalar_broadcast(
    %a: tensor<6x6x!ttcore.tile<32x32, f32>>,
    %c: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<6x6x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<6x6x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[6, 6], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[6, 6], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<6x6x!ttcore.tile<32x32, f32>>, !ttl.cb<[6, 6], !ttcore.tile<32x32, f32>, 2>) -> tensor<6x6x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<6x6x!ttcore.tile<32x32, f32>>, !ttl.cb<[6, 6], !ttcore.tile<32x32, f32>, 2>) -> tensor<6x6x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb2 : <[6, 6], !ttcore.tile<32x32, f32>, 2> -> tensor<6x6x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb, %c_cb : tensor<6x6x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<6x6x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map_identity, #map_scalar_bcast, #map_identity],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %c_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %c_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %sum, %reserve : !ttcore.tile<32x32, f32>, tensor<6x6x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<6x6x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<6x6x!ttcore.tile<32x32, f32>>
}
