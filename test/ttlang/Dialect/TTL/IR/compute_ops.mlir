// RUN: ttlang-opt --split-input-file %s | FileCheck %s
// Summary: Test TTL compute operation parsing and verification.

// -----

// TileAddOp: verify operands from tensor.extract flow to tile_add and result is returned.
// CHECK-LABEL: func.func @tile_add_basic
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1x!ttcore.tile<32x32, bf16>>, %[[ARG1:.*]]: tensor<1x1x!ttcore.tile<32x32, bf16>>)
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[LHS:.*]] = tensor.extract %[[ARG0]][%[[C0]], %[[C0]]] : tensor<1x1x!ttcore.tile<32x32, bf16>>
// CHECK: %[[RHS:.*]] = tensor.extract %[[ARG1]][%[[C0]], %[[C0]]] : tensor<1x1x!ttcore.tile<32x32, bf16>>
// CHECK: %[[RESULT:.*]] = ttl.tile_add %[[LHS]], %[[RHS]] : <32x32, bf16>, <32x32, bf16> -> <32x32, bf16>
// CHECK: return %[[RESULT]] : !ttcore.tile<32x32, bf16>
module {
  func.func @tile_add_basic(%lhs_view: tensor<1x1x!ttcore.tile<32x32, bf16>>,
                             %rhs_view: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> !ttcore.tile<32x32, bf16> {
    %c0 = arith.constant 0 : index
    %lhs = tensor.extract %lhs_view[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %rhs = tensor.extract %rhs_view[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %result = ttl.tile_add %lhs, %rhs : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    return %result : !ttcore.tile<32x32, bf16>
  }
}

// -----

// TileAddOp: verify f32 tile datatype is preserved.
// CHECK-LABEL: func.func @tile_add_f32
// CHECK: %[[LHS:.*]] = tensor.extract {{.*}} : tensor<1x1x!ttcore.tile<32x32, f32>>
// CHECK: %[[RHS:.*]] = tensor.extract {{.*}} : tensor<1x1x!ttcore.tile<32x32, f32>>
// CHECK: %[[RESULT:.*]] = ttl.tile_add %[[LHS]], %[[RHS]] : <32x32, f32>, <32x32, f32> -> <32x32, f32>
// CHECK: return %[[RESULT]] : !ttcore.tile<32x32, f32>
module {
  func.func @tile_add_f32(%lhs_view: tensor<1x1x!ttcore.tile<32x32, f32>>,
                           %rhs_view: tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttcore.tile<32x32, f32> {
    %c0 = arith.constant 0 : index
    %lhs = tensor.extract %lhs_view[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>>
    %rhs = tensor.extract %rhs_view[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>>
    %result = ttl.tile_add %lhs, %rhs : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    return %result : !ttcore.tile<32x32, f32>
  }
}

// -----

// StoreOp: verify value flows to store dest from cb_reserve.
// CHECK-LABEL: func.func @store_basic
// CHECK-SAME: (%[[CB:.*]]: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>, %[[TILE:.*]]: !ttcore.tile<32x32, bf16>)
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[VIEW:.*]] = ttl.cb_reserve %[[CB]], %[[C1]] : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
// CHECK: ttl.store %[[TILE]], %[[VIEW]] : <32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
// CHECK: ttl.cb_push %[[CB]], %[[C1]] : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
module {
  func.func @store_basic(%cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
                          %tile: !ttcore.tile<32x32, bf16>) {
    %c1 = arith.constant 1 : i32
    %view = ttl.cb_reserve %cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %tile, %view : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Complete compute pattern: verify tile_add result flows to store.
// CHECK-LABEL: func.func @compute_add_pattern
// CHECK-SAME: (%[[LHS_CB:.*]]: !ttl.cb<{{.*}}>, %[[RHS_CB:.*]]: !ttl.cb<{{.*}}>, %[[OUT_CB:.*]]: !ttl.cb<{{.*}}>)
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[LHS_VIEW:.*]] = ttl.cb_wait %[[LHS_CB]], %[[C1]]
// CHECK: %[[RHS_VIEW:.*]] = ttl.cb_wait %[[RHS_CB]], %[[C1]]
// CHECK: %[[LHS:.*]] = tensor.extract %[[LHS_VIEW]][%[[C0]], %[[C0]]]
// CHECK: %[[RHS:.*]] = tensor.extract %[[RHS_VIEW]][%[[C0]], %[[C0]]]
// CHECK: %[[RESULT:.*]] = ttl.tile_add %[[LHS]], %[[RHS]] : <32x32, bf16>, <32x32, bf16> -> <32x32, bf16>
// CHECK: %[[OUT_VIEW:.*]] = ttl.cb_reserve %[[OUT_CB]], %[[C1]]
// CHECK: ttl.store %[[RESULT]], %[[OUT_VIEW]] : <32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
// CHECK: ttl.cb_push %[[OUT_CB]], %[[C1]]
// CHECK: ttl.cb_pop %[[LHS_CB]], %[[C1]]
// CHECK: ttl.cb_pop %[[RHS_CB]], %[[C1]]
module {
  func.func @compute_add_pattern(
      %lhs_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %rhs_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
      %out_cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32

    %lhs_view = ttl.cb_wait %lhs_cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %rhs_view = ttl.cb_wait %rhs_cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>

    %lhs = tensor.extract %lhs_view[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %rhs = tensor.extract %rhs_view[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>

    %result = ttl.tile_add %lhs, %rhs : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>

    %out_view = ttl.cb_reserve %out_cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %out_view : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %out_cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>

    ttl.cb_pop %lhs_cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %rhs_cb, %c1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>

    return
  }
}
