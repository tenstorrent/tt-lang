// Tests for integrated unrolling in lower-to-loops: when a compute is marked
// with ttl.fully_unroll, lower-to-loops emits N unrolled copies of the body
// with incrementing DST indices and tile offsets, instead of creating scf.for.

// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8},ttl-subblock-compute-for-dst,ttl-insert-tile-regs-sync,ttl-lower-to-loops))' --split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 1x4 unary -- 4 tiles all fit in DST. Verify all 4 unrolled copies
// have correct dst_idx (0-3), tile_offset (0-3), and no scf.for loops.
// One sync region wraps all copies (acquire before, commit/wait/release after).
// CHECK-LABEL: func.func @unroll_unary_1x4
// CHECK-NOT:   scf.for
// CHECK:       ttl.tile_regs_acquire
// CHECK:       ttl.copy_tile {{.*}} {dst_idx = 0 : i32, ttl.tile_offset = 0
// CHECK:       ttl.tile_exp {{.*}} {dst_idx = 0 : i32, ttl.tile_offset = 0
// CHECK:       ttl.copy_tile {{.*}} {dst_idx = 1 : i32, ttl.tile_offset = 1
// CHECK:       ttl.tile_exp {{.*}} {dst_idx = 1 : i32, ttl.tile_offset = 1
// CHECK:       ttl.copy_tile {{.*}} {dst_idx = 2 : i32, ttl.tile_offset = 2
// CHECK:       ttl.tile_exp {{.*}} {dst_idx = 2 : i32, ttl.tile_offset = 2
// CHECK:       ttl.copy_tile {{.*}} {dst_idx = 3 : i32, ttl.tile_offset = 3
// CHECK:       ttl.tile_exp {{.*}} {dst_idx = 3 : i32, ttl.tile_offset = 3
// CHECK:       ttl.tile_regs_commit
// CHECK:       ttl.tile_regs_wait
// CHECK:       ttl.tile_store {{.*}} {ttl.tile_offset = 0
// CHECK:       ttl.tile_store {{.*}} {ttl.tile_offset = 1
// CHECK:       ttl.tile_store {{.*}} {ttl.tile_offset = 2
// CHECK:       ttl.tile_store {{.*}} {ttl.tile_offset = 3
// CHECK:       ttl.tile_regs_release
func.func @unroll_unary_1x4(%a: tensor<1x4x!ttcore.tile<32x32, f32>>)
    -> tensor<1x4x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x4x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 4], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x4x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb1 : (tensor<1x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x4x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb1 : <[1, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<1x4x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb : tensor<1x4x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x4x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %out_tile: !ttcore.tile<32x32, f32>):
    %exp = ttl.tile_exp %a_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %exp, %reserve : !ttcore.tile<32x32, f32>, tensor<1x4x!ttcore.tile<32x32, f32>>
    ttl.yield %exp : !ttcore.tile<32x32, f32>
  } -> tensor<1x4x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<1x4x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 1x4 FPU binary -- 4 tiles fit in DST. Verify 4 unrolled tile_add
// copies with correct dst_idx (0-3) and tile_offset (0-3). No scf.for loops.
// CHECK-LABEL: func.func @unroll_binary_1x4
// CHECK-NOT:   scf.for
// CHECK:       ttl.tile_add {{.*}} {dst_idx = 0 : i32{{.*}}ttl.tile_offset = 0
// CHECK:       ttl.tile_add {{.*}} {dst_idx = 1 : i32{{.*}}ttl.tile_offset = 1
// CHECK:       ttl.tile_add {{.*}} {dst_idx = 2 : i32{{.*}}ttl.tile_offset = 2
// CHECK:       ttl.tile_add {{.*}} {dst_idx = 3 : i32{{.*}}ttl.tile_offset = 3
// CHECK:       ttl.tile_regs_commit
// CHECK:       ttl.tile_regs_wait
// CHECK:       ttl.tile_store {{.*}} {ttl.tile_offset = 0
// CHECK:       ttl.tile_store {{.*}} {ttl.tile_offset = 1
// CHECK:       ttl.tile_store {{.*}} {ttl.tile_offset = 2
// CHECK:       ttl.tile_store {{.*}} {ttl.tile_offset = 3
func.func @unroll_binary_1x4(
    %a: tensor<1x4x!ttcore.tile<32x32, f32>>,
    %b: tensor<1x4x!ttcore.tile<32x32, f32>>)
    -> tensor<1x4x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<1x4x!ttcore.tile<32x32, f32>>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 4], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 4], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<1x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x4x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<1x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x4x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<1x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x4x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb2 : <[1, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<1x4x!ttcore.tile<32x32, f32>>

  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<1x4x!ttcore.tile<32x32, f32>>, tensor<1x4x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<1x4x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>, %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.tile_store %sum, %reserve : !ttcore.tile<32x32, f32>, tensor<1x4x!ttcore.tile<32x32, f32>>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<1x4x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<1x4x!ttcore.tile<32x32, f32>>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: 4x4 subblocked -- totalTiles=16, unroll_factor=8.
// Subblock sizes [2,4], outer scf.for on dim 0 (step 2).
// Inner body has 8 unrolled copies. One sync region per subblock iteration.
// CHECK-LABEL: func.func @unroll_subblocked_4x4
// CHECK:       scf.for %[[IV:.*]] =
// CHECK:         ttl.tile_regs_acquire
// CHECK:         arith.muli %[[IV]],
// CHECK:         ttl.copy_tile {{.*}} {dst_idx = 0
// CHECK:         ttl.tile_exp {{.*}} {dst_idx = 0
// CHECK:         ttl.copy_tile {{.*}} {dst_idx = 1
// CHECK:         ttl.tile_exp {{.*}} {dst_idx = 1
// Verify last unrolled copy in subblock has dst_idx = 7.
// CHECK:         ttl.copy_tile {{.*}} {dst_idx = 7
// CHECK:         ttl.tile_exp {{.*}} {dst_idx = 7
// CHECK:         ttl.tile_regs_commit
// CHECK:         ttl.tile_regs_wait
// CHECK:         ttl.tile_store
// CHECK:         ttl.tile_store
// CHECK:         ttl.tile_regs_release
func.func @unroll_subblocked_4x4(%a: tensor<4x4x!ttcore.tile<32x32, f32>>)
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

// Purpose: 2x4 non-subblocked -- all 8 tiles fit in DST.
// Verify 8 unrolled copies with 2D linearized offsets and no loops.
// Tile positions: (0,0)=0, (0,1)=1, ..., (0,3)=3, (1,0)=4, ..., (1,3)=7.
// CHECK-LABEL: func.func @unroll_2d_offsets_2x4
// CHECK-NOT:   scf.for
// CHECK:       ttl.copy_tile {{.*}} {dst_idx = 0 : i32, ttl.tile_offset = 0
// CHECK:       ttl.copy_tile {{.*}} {dst_idx = 1 : i32, ttl.tile_offset = 1
// CHECK:       ttl.copy_tile {{.*}} {dst_idx = 2 : i32, ttl.tile_offset = 2
// CHECK:       ttl.copy_tile {{.*}} {dst_idx = 3 : i32, ttl.tile_offset = 3
// CHECK:       ttl.copy_tile {{.*}} {dst_idx = 4 : i32, ttl.tile_offset = 4
// CHECK:       ttl.copy_tile {{.*}} {dst_idx = 5 : i32, ttl.tile_offset = 5
// CHECK:       ttl.copy_tile {{.*}} {dst_idx = 6 : i32, ttl.tile_offset = 6
// CHECK:       ttl.copy_tile {{.*}} {dst_idx = 7 : i32, ttl.tile_offset = 7
func.func @unroll_2d_offsets_2x4(%a: tensor<2x4x!ttcore.tile<32x32, f32>>)
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
