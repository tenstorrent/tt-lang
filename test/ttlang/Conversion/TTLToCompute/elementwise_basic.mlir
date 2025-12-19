// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-tile-and-assign-dst),cse,canonicalize)' | FileCheck %s

// Basic elementwise operations lowered to ttl.compute with tile ops and DST assignment.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @binary_add
func.func @binary_add(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>, %arg1: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[EMPTY:.*]] = tensor.empty
  // CHECK-DAG: %[[CB0:.*]] = ttl.bind_cb
  // CHECK-DAG: %[[CB1:.*]] = ttl.bind_cb
  // CHECK-DAG: %[[CB2:.*]] = ttl.bind_cb
  // CHECK-DAG: %[[ARG0_CB:.*]] = ttl.attach_cb %arg0, %[[CB0]]
  // CHECK-DAG: %[[ARG1_CB:.*]] = ttl.attach_cb %arg1, %[[CB1]]
  // CHECK-DAG: %[[EMPTY_CB:.*]] = ttl.attach_cb %[[EMPTY]], %[[CB2]]
  // CHECK: %[[COMPUTE:.*]] = ttl.compute ins(%[[ARG0_CB]], %[[ARG1_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[EMPTY_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) {
  // CHECK: ^bb0(%[[LHS:.*]]: !ttcore.tile<32x32, f32>, %[[RHS:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK:   %[[DTOK0:.*]], %[[DTILE0:.*]] = ttl.copy_tile %[[LHS]], %[[C0]], %[[C0]]
  // CHECK:   %[[DTOK1:.*]], %[[DTILE1:.*]] = ttl.copy_tile %[[RHS]], %[[C0]], %[[C1]]
  // CHECK:   %[[ADD:.*]] = ttl.tile_add %[[DTILE0]], %[[DTILE1]]
  // CHECK:   ttl.yield %[[ADD]]
  // CHECK: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK: return %[[COMPUTE]]
  %0 = ttl.add %arg0, %arg1 : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// CHECK-LABEL: func.func @unary_exp
func.func @unary_exp(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[EMPTY:.*]] = tensor.empty
  // CHECK-DAG: %[[CB0:.*]] = ttl.bind_cb
  // CHECK-DAG: %[[CB1:.*]] = ttl.bind_cb
  // CHECK-DAG: %[[ARG0_CB:.*]] = ttl.attach_cb %arg0, %[[CB0]]
  // CHECK-DAG: %[[EMPTY_CB:.*]] = ttl.attach_cb %[[EMPTY]], %[[CB1]]
  // CHECK: %[[COMPUTE:.*]] = ttl.compute ins(%[[ARG0_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[EMPTY_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) {
  // CHECK: ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[IN]], %[[C0]], %[[C0]]
  // CHECK:   %[[EXP:.*]] = ttl.tile_exp %[[DTILE]]
  // CHECK:   ttl.yield %[[EXP]]
  // CHECK: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK: return %[[COMPUTE]]
  %0 = ttl.exp %arg0 : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// CHECK-LABEL: func.func @chain_binary_unary
func.func @chain_binary_unary(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>, %arg1: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK: ttl.tile_add
  // CHECK: ttl.tile_relu
  // CHECK: return
  %0 = ttl.add %arg0, %arg1 : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %1 = ttl.relu %0 : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  func.return %1 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// CHECK-LABEL: func.func @multiple_binary
func.func @multiple_binary(%a: tensor<4x4x!ttcore.tile<32x32, f32>>, %b: tensor<4x4x!ttcore.tile<32x32, f32>>, %c: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK: ttl.tile_add
  // CHECK: ttl.tile_mul
  // CHECK: return
  %0 = ttl.add %a, %b : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %1 = ttl.mul %0, %c : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  func.return %1 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// CHECK-LABEL: func.func @unary_chain
func.func @unary_chain(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK: ttl.tile_abs
  // CHECK: ttl.tile_sqrt
  // CHECK: ttl.tile_log
  // CHECK: return
  %0 = ttl.abs %arg0 : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %1 = ttl.sqrt %0 : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %2 = ttl.log %1 : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  func.return %2 : tensor<4x4x!ttcore.tile<32x32, f32>>
}
