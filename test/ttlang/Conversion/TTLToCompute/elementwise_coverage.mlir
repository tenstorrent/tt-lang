// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-assign-dst-registers),canonicalize)' | FileCheck %s

// Test: Binary elementwise operations lower to ttl.compute with tile ops

// CHECK-LABEL: func.func @binary_add
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[ARG1:.+]]: tensor<2x2x!ttcore.tile<32x32, f32>>)
func.func @binary_add(%arg0: tensor<2x2x!ttcore.tile<32x32, f32>>, %arg1: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK: %[[RESULT:.+]] = ttl.compute
  // CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-SAME: outs(%[[INIT]] : tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN0:.+]]: !ttcore.tile<32x32, f32>, %[[IN1:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT: %[[ADD:.+]] = ttl.tile_add %[[IN0]], %[[IN1]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: ttl.yield %[[ADD]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %0 = ttl.add %arg0, %arg1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Unary elementwise operations (SFPU)

// CHECK-LABEL: func.func @unary_exp
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x4x!ttcore.tile<32x32, f32>>)
func.func @unary_exp(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK: %[[RESULT:.+]] = ttl.compute
  // CHECK-SAME: ins(%[[ARG0]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-SAME: outs(%[[INIT]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT: %[[EXP:.+]] = ttl.tile_exp %[[IN]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: ttl.yield %[[EXP]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %0 = ttl.exp %arg0 : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Chained elementwise operations produce multiple ttl.compute ops

// CHECK-LABEL: func.func @chained_ops
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[ARG1:.+]]: tensor<2x2x!ttcore.tile<32x32, f32>>)
func.func @chained_ops(%arg0: tensor<2x2x!ttcore.tile<32x32, f32>>, %arg1: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // CHECK: %[[INIT0:.+]] = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK: %[[ADD:.+]] = ttl.compute
  // CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-SAME: outs(%[[INIT0]] : tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN0:.+]]: !ttcore.tile<32x32, f32>, %[[IN1:.+]]: !ttcore.tile<32x32, f32>, %[[OUT0:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT: %[[TILE_ADD:.+]] = ttl.tile_add %[[IN0]], %[[IN1]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: ttl.yield %[[TILE_ADD]] : !ttcore.tile<32x32, f32>
  %0 = ttl.add %arg0, %arg1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: %[[INIT1:.+]] = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK: %[[RESULT:.+]] = ttl.compute
  // CHECK-SAME: ins(%[[ADD]] : tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-SAME: outs(%[[INIT1]] : tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN2:.+]]: !ttcore.tile<32x32, f32>, %[[OUT1:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT: %[[TILE_RELU:.+]] = ttl.tile_relu %[[IN2]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: ttl.yield %[[TILE_RELU]] : !ttcore.tile<32x32, f32>
  %1 = ttl.relu %0 : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: return %[[RESULT]]
  func.return %1 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: All binary operations

// CHECK-LABEL: func.func @all_binary_ops
func.func @all_binary_ops(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // CHECK: ttl.compute
  // CHECK: ttl.tile_add
  %add = ttl.add %a, %b : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_sub
  %sub = ttl.sub %add, %b : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_mul
  %mul = ttl.mul %sub, %a : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_max
  %max = ttl.max %mul, %b : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %max : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: All unary operations

// CHECK-LABEL: func.func @all_unary_ops
func.func @all_unary_ops(%x: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // CHECK: ttl.compute
  // CHECK: ttl.tile_exp
  %exp = ttl.exp %x : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_log
  %log = ttl.log %exp : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_sqrt
  %sqrt = ttl.sqrt %log : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_rsqrt
  %rsqrt = ttl.rsqrt %sqrt : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_tanh
  %tanh = ttl.tanh %rsqrt : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_sigmoid
  %sigmoid = ttl.sigmoid %tanh : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_neg
  %neg = ttl.neg %sigmoid : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_abs
  %abs = ttl.abs %neg : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_relu
  %relu = ttl.relu %abs : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %relu : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: DST assignment on complex chain

// CHECK-LABEL: func.func @dst_assignment_chain
// CHECK-SAME: (%[[A:.+]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[B:.+]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[C:.+]]: tensor<2x2x!ttcore.tile<32x32, f32>>)
func.func @dst_assignment_chain(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x2x!ttcore.tile<32x32, f32>>, %c: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // First compute: binary add
  // CHECK: %[[COMP0:.+]] = ttl.compute
  // CHECK-SAME: ins(%[[A]], %[[B]]
  // CHECK: ttl.tile_add{{.*}}{dst_idx = 0 : i32}
  %0 = ttl.add %a, %b : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Second compute: unary exp
  // CHECK: %[[COMP1:.+]] = ttl.compute
  // CHECK-SAME: ins(%[[COMP0]]
  // CHECK: ttl.tile_exp{{.*}}{dst_idx = 0 : i32}
  %1 = ttl.exp %0 : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Third compute: binary mul
  // CHECK: %[[COMP2:.+]] = ttl.compute
  // CHECK-SAME: ins(%[[COMP1]], %[[C]]
  // CHECK: ttl.tile_mul{{.*}}{dst_idx = 0 : i32}
  %2 = ttl.mul %1, %c : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: return %[[COMP2]]
  func.return %2 : tensor<2x2x!ttcore.tile<32x32, f32>>
}
