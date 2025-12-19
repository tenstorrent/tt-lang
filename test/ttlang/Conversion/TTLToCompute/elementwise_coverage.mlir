// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-tile-and-assign-dst),cse,canonicalize)' | FileCheck %s

// Test: Binary elementwise operations lower to ttl.compute with tile ops

// CHECK-LABEL: func.func @binary_add
func.func @binary_add(%arg0: tensor<2x2x!ttcore.tile<32x32, f32>>, %arg1: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[EMPTY:.*]] = tensor.empty
// CHECK-DAG: %[[CB0:.*]] = ttl.bind_cb
// CHECK-DAG: %[[CB1:.*]] = ttl.bind_cb
// CHECK-DAG: %[[CB2:.*]] = ttl.bind_cb
// CHECK-DAG: %[[ARG0_CB:.*]] = ttl.attach_cb %arg0, %[[CB0]]
// CHECK-DAG: %[[ARG1_CB:.*]] = ttl.attach_cb %arg1, %[[CB1]]
// CHECK-DAG: %[[EMPTY_CB:.*]] = ttl.attach_cb %[[EMPTY]], %[[CB2]]
// CHECK: %[[COMPUTE:.*]] = ttl.compute ins(%[[ARG0_CB]], %[[ARG1_CB]] : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%[[EMPTY_CB]] : tensor<2x2x!ttcore.tile<32x32, f32>>) {
// CHECK: ^bb0(%[[LHS:.*]]: !ttcore.tile<32x32, f32>, %[[RHS:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// CHECK:   %[[DTOK0:.*]], %[[DTILE0:.*]] = ttl.copy_tile %[[LHS]], %[[C0]], %[[C0]]
// CHECK:   %[[DTOK1:.*]], %[[DTILE1:.*]] = ttl.copy_tile %[[RHS]], %[[C0]], %[[C1]]
// CHECK:   %[[ADD:.*]] = ttl.tile_add %[[DTILE0]], %[[DTILE1]]
// CHECK:   ttl.yield %[[ADD]]
// CHECK: } -> tensor<2x2x!ttcore.tile<32x32, f32>>
// CHECK: return %[[COMPUTE]]
  %0 = ttl.add %arg0, %arg1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Unary elementwise operations (SFPU)

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

// -----

// Test: Chained elementwise operations produce multiple ttl.compute ops

// CHECK-LABEL: func.func @chained_ops
func.func @chained_ops(%arg0: tensor<2x2x!ttcore.tile<32x32, f32>>, %arg1: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
// CHECK: ttl.tile_add
// CHECK: ttl.tile_relu
// CHECK: return
  %0 = ttl.add %arg0, %arg1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %1 = ttl.relu %0 : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %1 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: All binary operations

// CHECK-LABEL: func.func @all_binary_ops
func.func @all_binary_ops(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // CHECK: ttl.tile_add
  %add = ttl.add %a, %b : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_sub
  %sub = ttl.sub %add, %b : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_mul
  %mul = ttl.mul %sub, %a : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_max
  %max = ttl.max %mul, %b : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %max : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: All unary operations

// CHECK-LABEL: func.func @all_unary_ops
func.func @all_unary_ops(%x: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // CHECK: ttl.tile_exp
  %exp = ttl.exp %x : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_log
  %log = ttl.log %exp : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_sqrt
  %sqrt = ttl.sqrt %log : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_rsqrt
  %rsqrt = ttl.rsqrt %sqrt : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_tanh
  %tanh = ttl.tanh %rsqrt : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_sigmoid
  %sigmoid = ttl.sigmoid %tanh : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_neg
  %neg = ttl.neg %sigmoid : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_abs
  %abs = ttl.abs %neg : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_relu
  %relu = ttl.relu %abs : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %relu : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: DST assignment on chain of binary and unary ops

// CHECK-LABEL: func.func @dst_assignment_chain
func.func @dst_assignment_chain(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x2x!ttcore.tile<32x32, f32>>, %c: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // First compute: binary add
  // CHECK: ttl.tile_add
  %0 = ttl.add %a, %b : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Second compute: unary exp
  // CHECK: ttl.tile_exp
  %1 = ttl.exp %0 : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Third compute: binary mul
  // CHECK: ttl.tile_mul {{.*}} {dst_idx = 2 : i32}
  %2 = ttl.mul %1, %c : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: return
  func.return %2 : tensor<2x2x!ttcore.tile<32x32, f32>>
}
