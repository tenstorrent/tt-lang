// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),cse,canonicalize)' | FileCheck %s

// Summary: A ttl.block.broadcast whose only consumer is add/sub/mul folds into
// a single ttl.tile_binary_bcast, so the broadcast is never materialized into
// DST. The broadcast operand is always second because the FPU only broadcasts
// its SRCB unpack source. Attribute encodings: EltwiseBinaryType Add=0, Sub=1,
// Mul=2; BcastType Col=1, Row=2, Scalar=3.

// Row broadcast (dims=[-2]) folded into an add.
// CHECK-LABEL: func.func @row_add
func.func @row_add(%arg0: tensor<1x2x!ttcore.tile<32x32, f32>>, %arg1: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<1x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x2x!ttcore.tile<32x32, f32>>
  %arg1_cb = ttl.attach_cb %arg1, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ^bb0(%[[BCAST_TILE:.*]]: !ttcore.tile<32x32, f32>, %[[DATA_TILE:.*]]: !ttcore.tile<32x32, f32>, %[[OUT_TILE:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK: ttl.tile_binary_bcast %[[DATA_TILE]], %[[BCAST_TILE]], %[[OUT_TILE]] 0 : i32 2 : i32 into dst
  %reserve = ttl.cb_reserve %cb2 : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %bcast = ttl.block.broadcast %arg0_cb dims = [-2], shape = [2, 2] : tensor<1x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %add = ttl.add %bcast, %arg1_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %add, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %add : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Column broadcast (dims=[-1]) folded into a mul.
// CHECK-LABEL: func.func @col_mul
func.func @col_mul(%arg0: tensor<2x1x!ttcore.tile<32x32, f32>>, %arg1: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<2x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x1x!ttcore.tile<32x32, f32>>
  %arg1_cb = ttl.attach_cb %arg1, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ^bb0(%[[BCAST_TILE:.*]]: !ttcore.tile<32x32, f32>, %[[DATA_TILE:.*]]: !ttcore.tile<32x32, f32>, %[[OUT_TILE:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK: ttl.tile_binary_bcast %[[DATA_TILE]], %[[BCAST_TILE]], %[[OUT_TILE]] 2 : i32 1 : i32 into dst
  %reserve = ttl.cb_reserve %cb2 : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %bcast = ttl.block.broadcast %arg0_cb dims = [-1], shape = [2, 2] : tensor<2x1x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %mul = ttl.mul %arg1_cb, %bcast : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %mul, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %mul : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Scalar broadcast (dims=[-1, -2]) folded into a sub with the broadcast on the
// right, which is the only orientation the FPU supports for sub.
// CHECK-LABEL: func.func @scalar_sub
func.func @scalar_sub(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>>, %arg1: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %arg1_cb = ttl.attach_cb %arg1, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ^bb0(%[[BCAST_TILE:.*]]: !ttcore.tile<32x32, f32>, %[[DATA_TILE:.*]]: !ttcore.tile<32x32, f32>, %[[OUT_TILE:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK: ttl.tile_binary_bcast %[[DATA_TILE]], %[[BCAST_TILE]], %[[OUT_TILE]] 1 : i32 3 : i32 into dst
  %reserve = ttl.cb_reserve %cb2 : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %bcast = ttl.block.broadcast %arg0_cb dims = [-1, -2], shape = [2, 2] : tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %sub = ttl.sub %arg1_cb, %bcast : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %sub, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %sub : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Fallback: `broadcast(B) - A` cannot be expressed by the FPU, so the pair
// stays as tile_bcast followed by tile_sub.
// CHECK-LABEL: func.func @scalar_sub_reversed
func.func @scalar_sub_reversed(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>>, %arg1: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %arg1_cb = ttl.attach_cb %arg1, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK-NOT: ttl.tile_binary_bcast
  // CHECK: ttl.tile_bcast
  // CHECK: ttl.tile_sub
  %reserve = ttl.cb_reserve %cb2 : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %bcast = ttl.block.broadcast %arg0_cb dims = [-1, -2], shape = [2, 2] : tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %sub = ttl.sub %bcast, %arg1_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %sub, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %sub : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Fallback: an inter-tile-only broadcast has no within-tile hardware kind, so
// there is nothing for the FPU to fold.
// CHECK-LABEL: func.func @inter_tile_broadcast_not_fused
func.func @inter_tile_broadcast_not_fused(%arg0: tensor<1x2x2x!ttcore.tile<32x32, f32>>, %arg1: tensor<2x2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2, 2], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<1x2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x2x2x!ttcore.tile<32x32, f32>>
  %arg1_cb = ttl.attach_cb %arg1, %cb1 : (tensor<2x2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x2x!ttcore.tile<32x32, f32>>

  // CHECK-NOT: ttl.tile_binary_bcast
  // CHECK: ttl.tile_add
  %reserve = ttl.cb_reserve %cb2 : !ttl.cb<[2, 2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x2x!ttcore.tile<32x32, f32>>
  %bcast = ttl.block.broadcast %arg0_cb dims = [-3], shape = [2, 2, 2] : tensor<1x2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x2x!ttcore.tile<32x32, f32>>
  %add = ttl.add %bcast, %arg1_cb : tensor<2x2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x2x!ttcore.tile<32x32, f32>>
  ttl.store %add, %reserve : tensor<2x2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x2x!ttcore.tile<32x32, f32>>
  func.return %add : tensor<2x2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Fallback: the broadcast result has two consumers, so it must be materialized
// into DST once and shared.
// CHECK-LABEL: func.func @multi_use_broadcast_not_fused
func.func @multi_use_broadcast_not_fused(%arg0: tensor<1x2x!ttcore.tile<32x32, f32>>, %arg1: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<1x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x2x!ttcore.tile<32x32, f32>>
  %arg1_cb = ttl.attach_cb %arg1, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK-NOT: ttl.tile_binary_bcast
  // CHECK: ttl.tile_bcast
  %reserve = ttl.cb_reserve %cb2 : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %bcast = ttl.block.broadcast %arg0_cb dims = [-2], shape = [2, 2] : tensor<1x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %add = ttl.add %bcast, %arg1_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %mul = ttl.mul %add, %bcast : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %mul, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %mul : tensor<2x2x!ttcore.tile<32x32, f32>>
}
