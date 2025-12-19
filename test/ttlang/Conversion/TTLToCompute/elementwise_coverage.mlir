// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-tile-and-assign-dst),canonicalize)' | FileCheck %s

// Test: Binary elementwise operations lower to ttl.compute with tile ops

// CHECK-LABEL: func.func @binary_add
// CHECK-SAME: (%[[ARG0:.+]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[ARG1:.+]]: tensor<2x2x!ttcore.tile<32x32, f32>>)
func.func @binary_add(%arg0: tensor<2x2x!ttcore.tile<32x32, f32>>, %arg1: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // CHECK-NEXT: %[[INIT:.+]] = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[CB0:.+]] = ttl.bind_cb
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb
  // CHECK-NEXT: %[[CB2:.+]] = ttl.bind_cb
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK-NEXT: %[[ARG1_CB:.+]] = ttl.attach_cb %[[ARG1]], %[[CB1]]
  // CHECK-NEXT: %[[INIT_CB:.+]] = ttl.attach_cb %[[INIT]], %[[CB2]]
  // CHECK-NEXT: %[[RESULT:.+]] = ttl.compute ins(%[[ARG0_CB]], %[[ARG1_CB]] : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN0:.+]]: !ttcore.tile<32x32, f32>, %[[IN1:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[ADD:.+]] = ttl.tile_add %[[IN0]], %[[IN1]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   ttl.yield %[[ADD]] : !ttcore.tile<32x32, f32>
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
  // CHECK-NEXT: %[[INIT:.+]] = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[CB0:.+]] = ttl.bind_cb
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK-NEXT: %[[INIT_CB:.+]] = ttl.attach_cb %[[INIT]], %[[CB1]]
  // CHECK-NEXT: %[[RESULT:.+]] = ttl.compute ins(%[[ARG0_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[EXP:.+]] = ttl.tile_exp %[[IN]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   ttl.yield %[[EXP]] : !ttcore.tile<32x32, f32>
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
  // CHECK-NEXT: %[[INIT0:.+]] = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[CB0:.+]] = ttl.bind_cb
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb
  // CHECK-NEXT: %[[CB2:.+]] = ttl.bind_cb
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK-NEXT: %[[ARG1_CB:.+]] = ttl.attach_cb %[[ARG1]], %[[CB1]]
  // CHECK-NEXT: %[[INIT0_CB:.+]] = ttl.attach_cb %[[INIT0]], %[[CB2]]
  // CHECK-NEXT: %[[ADD:.+]] = ttl.compute ins(%[[ARG0_CB]], %[[ARG1_CB]] : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%[[INIT0_CB]] : tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN0:.+]]: !ttcore.tile<32x32, f32>, %[[IN1:.+]]: !ttcore.tile<32x32, f32>, %[[OUT0:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[TILE_ADD:.+]] = ttl.tile_add %[[IN0]], %[[IN1]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   ttl.yield %[[TILE_ADD]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.add %arg0, %arg1 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK-NEXT: %[[INIT1:.+]] = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[CB3:.+]] = ttl.bind_cb
  // CHECK-NEXT: %[[CB4:.+]] = ttl.bind_cb
  // CHECK-NEXT: %[[ADD_CB:.+]] = ttl.attach_cb %[[ADD]], %[[CB3]]
  // CHECK-NEXT: %[[INIT1_CB:.+]] = ttl.attach_cb %[[INIT1]], %[[CB4]]
  // CHECK-NEXT: %[[RESULT:.+]] = ttl.compute ins(%[[ADD_CB]] : tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%[[INIT1_CB]] : tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN2:.+]]: !ttcore.tile<32x32, f32>, %[[OUT1:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[TILE_RELU:.+]] = ttl.tile_relu %[[IN2]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   ttl.yield %[[TILE_RELU]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %1 = ttl.relu %0 : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK-NEXT: return %[[RESULT]]
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
  // CHECK: %[[A_CB:.+]] = ttl.attach_cb %[[A]]
  // CHECK: %[[B_CB:.+]] = ttl.attach_cb %[[B]]
  // CHECK: %[[COMP0:.+]] = ttl.compute
  // CHECK: ttl.tile_add
  %0 = ttl.add %a, %b : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Second compute: unary exp
  // CHECK: %[[COMP0_CB:.+]] = ttl.attach_cb %[[COMP0]]
  // CHECK: %[[COMP1:.+]] = ttl.compute
  // CHECK: ttl.tile_exp
  %1 = ttl.exp %0 : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Third compute: binary mul
  // CHECK: %[[COMP1_CB:.+]] = ttl.attach_cb %[[COMP1]]
  // CHECK: %[[C_CB:.+]] = ttl.attach_cb %[[C]]
  // CHECK: %[[COMP2:.+]] = ttl.compute
  // CHECK: ttl.tile_mul
  // CHECK-NOT: dst_idx
  %2 = ttl.mul %1, %c : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: return %[[COMP2]]
  func.return %2 : tensor<2x2x!ttcore.tile<32x32, f32>>
}
