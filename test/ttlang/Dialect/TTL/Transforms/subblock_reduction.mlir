// Tests for ttl-subblock-compute-for-dst with reduction iterators.
// Verifies that only parallel dimensions are subblocked while reduction
// dimensions are always fully included in each subblock.
//
// TODO: Add a test with an actual reduction op once one is implemented.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-subblock-compute-for-dst))' --split-input-file | FileCheck %s --check-prefix=SUBBLOCK

// -----

// Purpose: Reduction that fits entirely in DST -- no subblocking needed.
// Shape: 2x3 iteration domain (parallel=2, reduction=3).
// totalTiles=6, unroll_factor=6=totalTiles -> all fit in one subblock.

#map_in = affine_map<(d0, d1) -> (d0, d1)>
#map_out = affine_map<(d0, d1) -> (d0)>

// SUBBLOCK-LABEL: func.func @reduction_fits_in_dst
// No loop -- everything fits in one subblock.
// SUBBLOCK-NOT:   scf.for
// SUBBLOCK:       ttl.compute
// SUBBLOCK-SAME:  ins({{.*}} : tensor<2x3x!ttcore.tile<32x32, f32>>)
// SUBBLOCK-SAME:  outs({{.*}} : tensor<2x!ttcore.tile<32x32, f32>>)
// SUBBLOCK-SAME:  iterator_types = ["parallel", "reduction"]
// SUBBLOCK-SAME:  ttl.full_linearization_strides = array<i64: 3, 1>
// SUBBLOCK-SAME:  ttl.unroll_factor = 6
// Original body preserved.
// SUBBLOCK:         ttl.tile_add
// SUBBLOCK:         ttl.tile_store
// SUBBLOCK:         ttl.yield
func.func @reduction_fits_in_dst(
    %a: tensor<2x3x!ttcore.tile<32x32, f32>>)
    -> tensor<2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x3x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb1 : <[2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb : tensor<2x3x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map_in, #map_out],
       iterator_types = ["parallel", "reduction"],
       ttl.unroll_factor = 6 : i64} {
  ^bb0(%in: !ttcore.tile<32x32, f32>, %acc: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %add = ttl.tile_add %in, %acc : !ttcore.tile<32x32, f32>
    ttl.tile_store %add, %reserve[%i] : !ttcore.tile<32x32, f32>, tensor<2x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x!ttcore.tile<32x32, f32>>
}

// -----

// Purpose: Mixed parallel+reduction where only the parallel dim is subblocked.
// Shape: 8x3 iteration domain (parallel=8, reduction=3).
// totalTiles=24, unroll_factor=8.
// reductionProduct=3, parallelBudget=8/3=2.
// Subblock parallel dim: [8] with budget 2 -> subblock size [2].
// Full subblock sizes: [2, 3]. Loop on dim 0 (0 to 8 step 2).

#map_in2 = affine_map<(d0, d1) -> (d0, d1)>
#map_out2 = affine_map<(d0, d1) -> (d0)>

// SUBBLOCK-LABEL: func.func @reduction_subblock_parallel_only
// Outer loop on parallel dim 0: 0 to 8 step 2.
// SUBBLOCK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// SUBBLOCK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// SUBBLOCK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// SUBBLOCK:       scf.for %[[IV:.*]] = %[[C0]] to %[[C8]] step %[[C2]] {
// Input sliced on parallel dim, reduction dim kept whole: [iv, 0] [2, 3].
// SUBBLOCK:         tensor.extract_slice {{.*}}[%[[IV]], 0] [2, 3] [1, 1]
// SUBBLOCK-SAME:    tensor<8x3x!ttcore.tile<32x32, f32>> to tensor<2x3x!ttcore.tile<32x32, f32>>
// Output sliced on parallel dim only: [iv] [2].
// SUBBLOCK:         tensor.extract_slice {{.*}}[%[[IV]]] [2] [1]
// SUBBLOCK-SAME:    tensor<8x!ttcore.tile<32x32, f32>> to tensor<2x!ttcore.tile<32x32, f32>>
// Inner compute on the subblocked shapes.
// SUBBLOCK:         ttl.compute
// SUBBLOCK-SAME:    ins({{.*}} : tensor<2x3x!ttcore.tile<32x32, f32>>)
// SUBBLOCK-SAME:    outs({{.*}} : tensor<2x!ttcore.tile<32x32, f32>>)
// SUBBLOCK-SAME:    iterator_types = ["parallel", "reduction"]
// SUBBLOCK-SAME:    ttl.full_linearization_strides = array<i64: 3, 1>
// unroll_factor removed from subblocked compute.
// SUBBLOCK-NOT:     ttl.unroll_factor
// SUBBLOCK:           ttl.tile_add
// SUBBLOCK:           ttl.tile_store
// SUBBLOCK:           ttl.yield
// No second loop -- reduction dim is NOT subblocked.
// SUBBLOCK-NOT:     scf.for
// Loop annotated with subblock dim and stride.
// SUBBLOCK:       } {ttl.subblock_dim = 0 : index, ttl.subblock_loop_stride = 3 : index}
func.func @reduction_subblock_parallel_only(
    %a: tensor<8x3x!ttcore.tile<32x32, f32>>)
    -> tensor<8x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<8x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[8, 3], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[8], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<8x3x!ttcore.tile<32x32, f32>>, !ttl.cb<[8, 3], !ttcore.tile<32x32, f32>, 2>) -> tensor<8x3x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<8x!ttcore.tile<32x32, f32>>, !ttl.cb<[8], !ttcore.tile<32x32, f32>, 2>) -> tensor<8x!ttcore.tile<32x32, f32>>

  %reserve2 = ttl.cb_reserve %cb1 : <[8], !ttcore.tile<32x32, f32>, 2> -> tensor<8x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb : tensor<8x3x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<8x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map_in2, #map_out2],
       iterator_types = ["parallel", "reduction"],
       ttl.unroll_factor = 8 : i64} {
  ^bb0(%in: !ttcore.tile<32x32, f32>, %acc: !ttcore.tile<32x32, f32>):
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %add = ttl.tile_add %in, %acc : !ttcore.tile<32x32, f32>
    ttl.tile_store %add, %reserve2[%i] : !ttcore.tile<32x32, f32>, tensor<8x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<8x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<8x!ttcore.tile<32x32, f32>>
}
