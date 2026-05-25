// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),cse,canonicalize)' --split-input-file | FileCheck %s

// Summary: ND lowering coverage for ttl.block.broadcast. The op accepts any
// rank; the lowering builds an N-dim affine map (constant 0 in broadcast
// dims, identity elsewhere) and derives the tile-level BcastType from
// whether the innermost two dims participate in the broadcast.

// 3D outer-only broadcast: input (1, K, N) -> output (I, K, N), dims = [0].
// Innermost two dims (K, N) are unchanged; the broadcast is purely
// inter-tile and the body passes the input tile through (no tile_bcast).
// CHECK-LABEL: func.func @bcast_3d_outer
// CHECK: %[[IN:.*]] = ttl.attach_cb %arg0
// CHECK: %[[OUT:.*]] = ttl.cb_reserve
// CHECK: %[[INIT:.*]] = ttl.attach_cb {{.*}}
// CHECK: %[[COMPUTE:.*]] = ttl.compute ins(%[[IN]]
// CHECK-SAME: outs(%[[INIT]]
// CHECK-NEXT: ^bb0(%[[IN_TILE:.*]]: !ttcore.tile<32x32, f32>, %[[OUT_TILE:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT: %[[DIM0:.*]] = ttl.iter_index 0
// CHECK-NEXT: %[[DIM1:.*]] = ttl.iter_index 1
// CHECK-NEXT: %[[DIM2:.*]] = ttl.iter_index 2
// CHECK-NOT: tile_bcast
// CHECK-NEXT: ttl.tile_store %[[IN_TILE]], %[[OUT]][%[[DIM0]], %[[DIM1]], %[[DIM2]]] from dst
// CHECK-NEXT: ttl.yield
// CHECK: return %[[COMPUTE]]
func.func @bcast_3d_outer(%arg0: tensor<1x2x2x!ttcore.tile<32x32, f32>>) -> tensor<3x2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[3, 2, 2], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<1x2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x2x2x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : !ttl.cb<[3, 2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<3x2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.block.broadcast %arg0_cb dims = [0], shape = [3, 2, 2] : tensor<1x2x2x!ttcore.tile<32x32, f32>> -> tensor<3x2x2x!ttcore.tile<32x32, f32>>
  ttl.store %result, %reserve : tensor<3x2x2x!ttcore.tile<32x32, f32>>, tensor<3x2x2x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<3x2x2x!ttcore.tile<32x32, f32>>
}

// -----

// 3D outer + innermost broadcast: input (1, 2, 1) -> output (3, 2, 4),
// dims = [0, -1]. Innermost dim is broadcast => Col (1); outer dim 0 is
// purely inter-tile (constant 0 in input affine map).
// CHECK-LABEL: func.func @bcast_3d_outer_and_col
// CHECK: %[[IN:.*]] = ttl.attach_cb %arg0
// CHECK: %[[OUT:.*]] = ttl.cb_reserve
// CHECK: %[[INIT:.*]] = ttl.attach_cb {{.*}}
// CHECK: %[[COMPUTE:.*]] = ttl.compute ins(%[[IN]]
// CHECK-SAME: outs(%[[INIT]]
// CHECK-NEXT: ^bb0(%[[IN_TILE:.*]]: !ttcore.tile<32x32, f32>, %[[OUT_TILE:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT: %[[DIM0:.*]] = ttl.iter_index 0
// CHECK-NEXT: %[[DIM1:.*]] = ttl.iter_index 1
// CHECK-NEXT: %[[DIM2:.*]] = ttl.iter_index 2
// CHECK-NEXT: %[[BCASTED:.*]] = ttl.tile_bcast %[[IN_TILE]], %[[OUT_TILE]] 1 : i32
// CHECK-NEXT: ttl.tile_store %[[BCASTED]], %[[OUT]][%[[DIM0]], %[[DIM1]], %[[DIM2]]] from dst
// CHECK-NEXT: ttl.yield
// CHECK: return %[[COMPUTE]]
func.func @bcast_3d_outer_and_col(%arg0: tensor<1x2x1x!ttcore.tile<32x32, f32>>) -> tensor<3x2x4x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 2, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[3, 2, 4], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<1x2x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 2, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x2x1x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : !ttl.cb<[3, 2, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<3x2x4x!ttcore.tile<32x32, f32>>
  %result = ttl.block.broadcast %arg0_cb dims = [0, -1], shape = [3, 2, 4] : tensor<1x2x1x!ttcore.tile<32x32, f32>> -> tensor<3x2x4x!ttcore.tile<32x32, f32>>
  ttl.store %result, %reserve : tensor<3x2x4x!ttcore.tile<32x32, f32>>, tensor<3x2x4x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<3x2x4x!ttcore.tile<32x32, f32>>
}

// -----

// 3D scalar-tile broadcast across innermost two dims: input (2, 1, 1) ->
// output (2, 3, 4), dims = [-1, -2]. Outer dim 0 has matching size; both
// innermost dims broadcast => Scalar (3).
// CHECK-LABEL: func.func @bcast_3d_innermost_scalar
// CHECK: %[[IN:.*]] = ttl.attach_cb %arg0
// CHECK: %[[OUT:.*]] = ttl.cb_reserve
// CHECK: %[[INIT:.*]] = ttl.attach_cb {{.*}}
// CHECK: %[[COMPUTE:.*]] = ttl.compute ins(%[[IN]]
// CHECK-SAME: outs(%[[INIT]]
// CHECK-NEXT: ^bb0(%[[IN_TILE:.*]]: !ttcore.tile<32x32, f32>, %[[OUT_TILE:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT: %[[DIM0:.*]] = ttl.iter_index 0
// CHECK-NEXT: %[[DIM1:.*]] = ttl.iter_index 1
// CHECK-NEXT: %[[DIM2:.*]] = ttl.iter_index 2
// CHECK-NEXT: %[[BCASTED:.*]] = ttl.tile_bcast %[[IN_TILE]], %[[OUT_TILE]] 3 : i32
// CHECK-NEXT: ttl.tile_store %[[BCASTED]], %[[OUT]][%[[DIM0]], %[[DIM1]], %[[DIM2]]] from dst
// CHECK-NEXT: ttl.yield
// CHECK: return %[[COMPUTE]]
func.func @bcast_3d_innermost_scalar(%arg0: tensor<2x1x1x!ttcore.tile<32x32, f32>>) -> tensor<2x3x4x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 3, 4], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<2x1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x1x1x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : !ttl.cb<[2, 3, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<2x3x4x!ttcore.tile<32x32, f32>>
  %result = ttl.block.broadcast %arg0_cb dims = [-1, -2], shape = [2, 3, 4] : tensor<2x1x1x!ttcore.tile<32x32, f32>> -> tensor<2x3x4x!ttcore.tile<32x32, f32>>
  ttl.store %result, %reserve : tensor<2x3x4x!ttcore.tile<32x32, f32>>, tensor<2x3x4x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<2x3x4x!ttcore.tile<32x32, f32>>
}
