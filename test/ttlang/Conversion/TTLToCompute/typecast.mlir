// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute))' --split-input-file | FileCheck %s

// Test ttl.typecast lowers to a ttl.compute with ttl.tile_typecast in the
// body. The compute has block arguments of different element types (input is
// bf16, output is f32) since typecast changes the element data type.

// CHECK-LABEL: func.func @typecast_bf16_to_f32
func.func @typecast_bf16_to_f32(%a: tensor<2x2x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  // CHECK:      %[[RES:.*]] = ttl.compute
  // CHECK-SAME:   ins(%{{.*}} : tensor<2x2x!ttcore.tile<32x32, bf16>>)
  // CHECK-SAME:   outs(%{{.*}} : tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK:      ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, bf16>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK:        ttl.tile_typecast %[[IN]] into dst[%{{.*}}] {{.*}} : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
  // CHECK:        ttl.tile_store
  // CHECK:        ttl.yield
  // CHECK:      } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.typecast %a_cb
       : (tensor<2x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test that fusing a chain containing ttl.typecast produces a single
// ttl.compute whose block arguments retain each root input's true element
// type (bf16 for `a`, f32 for `b`) and whose body emits tile_typecast +
// SFPU tile_add/tile_mul on the correctly typed intermediates. Regression
// for "fusion failed: unsupported op type" when TypecastOp appeared in a
// fused chain.

// CHECK-LABEL: func.func @fuse_typecast_with_f32_binary
func.func @fuse_typecast_with_f32_binary(
    %a: tensor<3x5x!ttcore.tile<32x32, bf16>>,
    %b: tensor<3x5x!ttcore.tile<32x32, f32>>)
    -> tensor<3x5x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[3, 5], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[3, 5], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[3, 5], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<3x5x!ttcore.tile<32x32, bf16>>, !ttl.cb<[3, 5], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<3x5x!ttcore.tile<32x32, bf16>>
  %b_cb = ttl.attach_cb %b, %cb1
      : (tensor<3x5x!ttcore.tile<32x32, f32>>, !ttl.cb<[3, 5], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<3x5x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb2 : <[3, 5], !ttcore.tile<32x32, f32>, 2> -> tensor<3x5x!ttcore.tile<32x32, f32>>

  // CHECK:      ttl.compute
  // CHECK-SAME:   ins(%{{.*}}, %{{.*}} : tensor<3x5x!ttcore.tile<32x32, bf16>>, tensor<3x5x!ttcore.tile<32x32, f32>>)
  // CHECK-SAME:   outs(%{{.*}} : tensor<3x5x!ttcore.tile<32x32, f32>>)
  // CHECK:      ^bb0(%[[A_BF16:.*]]: !ttcore.tile<32x32, bf16>, %[[B_F32:.*]]: !ttcore.tile<32x32, f32>, %{{.*}}: !ttcore.tile<32x32, f32>):
  // CHECK:        %[[A_F32:.*]] = ttl.tile_typecast %[[A_BF16]] into dst[%{{.*}}] {{.*}} : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
  // CHECK:        %[[ADD:.*]] = ttl.tile_add %[[A_F32]], %[[B_F32]] into dst[%{{.*}}]
  // CHECK:        %[[MUL:.*]] = ttl.tile_mul %[[ADD]], %[[B_F32]] into dst[%{{.*}}]
  // CHECK:        %[[OUT:.*]] = ttl.tile_add %[[MUL]], %[[B_F32]] into dst[%{{.*}}]
  // CHECK:        ttl.tile_store %[[OUT]], %{{.*}}
  %t = ttl.typecast %a_cb
       : (tensor<3x5x!ttcore.tile<32x32, bf16>>) -> tensor<3x5x!ttcore.tile<32x32, f32>>
  %0 = ttl.add %t, %b_cb
       : tensor<3x5x!ttcore.tile<32x32, f32>>, tensor<3x5x!ttcore.tile<32x32, f32>> -> tensor<3x5x!ttcore.tile<32x32, f32>>
  %1 = ttl.mul %0, %b_cb
       : tensor<3x5x!ttcore.tile<32x32, f32>>, tensor<3x5x!ttcore.tile<32x32, f32>> -> tensor<3x5x!ttcore.tile<32x32, f32>>
  %2 = ttl.add %1, %b_cb
       : tensor<3x5x!ttcore.tile<32x32, f32>>, tensor<3x5x!ttcore.tile<32x32, f32>> -> tensor<3x5x!ttcore.tile<32x32, f32>>
  ttl.store %2, %reserve : tensor<3x5x!ttcore.tile<32x32, f32>>, tensor<3x5x!ttcore.tile<32x32, f32>>

  return %2 : tensor<3x5x!ttcore.tile<32x32, f32>>
}

// -----

// Test that fusing typecast(exp(x)) lowers correctly: the SFPU unary keeps
// its input element type, and the typecast at the end converts to the
// destination tile type.

// CHECK-LABEL: func.func @fuse_exp_then_typecast
func.func @fuse_exp_then_typecast(%a: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>

  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  // CHECK:      ttl.compute
  // CHECK-SAME:   ins(%{{.*}} : tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-SAME:   outs(%{{.*}} : tensor<2x2x!ttcore.tile<32x32, bf16>>)
  // CHECK:      ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, f32>, %{{.*}}: !ttcore.tile<32x32, bf16>):
  // CHECK:        %[[E:.*]] = ttl.tile_exp %[[IN]] into dst[%{{.*}}] {{.*}} : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  // CHECK:        %[[C:.*]] = ttl.tile_typecast %[[E]] into dst[%{{.*}}] {{.*}} : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, bf16>
  // CHECK:        ttl.tile_store %[[C]], %{{.*}}
  %e = ttl.exp %a_cb
       : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %t = ttl.typecast %e
       : (tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.store %t, %reserve : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>

  return %t : tensor<2x2x!ttcore.tile<32x32, bf16>>
}

// -----

// Test that fusing typecast(block.broadcast(x)) preserves the broadcast's
// own dtype on the tile result so the trailing tile_typecast performs the
// conversion. Regression for using the final fused sink dtype on the bcast
// result, which would have produced an f32 tile_bcast (incorrect: bcast
// preserves input dtype) and a no-op typecast that drops the bf16 -> f32
// conversion.

// CHECK-LABEL: func.func @fuse_bcast_then_typecast
func.func @fuse_bcast_then_typecast(%a: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb_out = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %reserve = ttl.cb_reserve %cb_out : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK:      ttl.compute
  // CHECK-SAME:   ins(%{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>)
  // CHECK-SAME:   outs(%{{.*}} : tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK:      ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, bf16>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK:        %[[B:.*]] = ttl.tile_bcast %[[IN]], %[[OUT]]{{.*}}-> !ttcore.tile<32x32, bf16>
  // CHECK:        %[[C:.*]] = ttl.tile_typecast %[[B]]{{.*}}: !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
  // CHECK:        ttl.tile_store %[[C]], %{{.*}}
  %b = ttl.block.broadcast %a_cb dims = [-2, -1], shape = [2, 2]
       : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %t = ttl.typecast %b
       : (tensor<2x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %t, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  return %t : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test that fusing typecast(block.broadcast(a) + b) keeps both the bcast and
// add results in their own dtype (bf16) and emits a trailing tile_typecast
// for the bf16 -> f32 conversion. Combines the bcast and elementwise paths.

// CHECK-LABEL: func.func @fuse_bcast_add_then_typecast
func.func @fuse_bcast_add_then_typecast(
    %a: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %b: tensor<2x2x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb_out = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %b_cb = ttl.attach_cb %b, %cb1
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  %reserve = ttl.cb_reserve %cb_out : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK:      ttl.compute
  // CHECK-SAME:   ins(%{{.*}}, %{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>)
  // CHECK-SAME:   outs(%{{.*}} : tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK:      ^bb0(%[[A:.*]]: !ttcore.tile<32x32, bf16>, %[[B:.*]]: !ttcore.tile<32x32, bf16>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK:        %[[BC:.*]] = ttl.tile_bcast %[[A]], %[[OUT]]{{.*}}-> !ttcore.tile<32x32, bf16>
  // CHECK:        %[[ADD:.*]] = ttl.tile_add %[[BC]], %[[B]]{{.*}}: !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
  // CHECK:        %[[C:.*]] = ttl.tile_typecast %[[ADD]]{{.*}}: !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
  // CHECK:        ttl.tile_store %[[C]], %{{.*}}
  %bc = ttl.block.broadcast %a_cb dims = [-2, -1], shape = [2, 2]
       : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %sum = ttl.add %bc, %b_cb
       : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>
         -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %t = ttl.typecast %sum
       : (tensor<2x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %t, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  return %t : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test same-type ttl.typecast lowers as a no-op before store lowering builds
// the compute that copies the input tile to the output CB.

// CHECK-LABEL: func.func @typecast_identity_to_input
// CHECK: %[[IN_CB:.*]] = ttl.attach_cb
// CHECK-NOT: ttl.typecast
// CHECK-NOT: ttl.tile_typecast
// CHECK: ttl.tile_store
// CHECK: return %[[IN_CB]]
func.func @typecast_identity_to_input(
    %a: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.typecast %a_cb
       : (tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}
