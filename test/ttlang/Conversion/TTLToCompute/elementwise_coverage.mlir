// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-assign-dst),cse,canonicalize)' | FileCheck %s

// Test: Binary elementwise operations lower to ttl.compute with tile ops
// Input provides explicit bind_cb and attach_cb ops.
// This test verifies the full CB attachment pattern for all arguments.

// CHECK-LABEL: func.func @binary_add
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[ARG1:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
func.func @binary_add(%arg0: tensor<2x2x!ttcore.tile<32x32, f32>>, %arg1: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-DAG: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-DAG: %[[CB2:.+]] = ttl.bind_cb{cb_index = 2
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK-NEXT: %[[ARG1_CB:.+]] = ttl.attach_cb %[[ARG1]], %[[CB1]]
  // CHECK-NEXT: %[[INIT:.+]] = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[INIT_CB:.+]] = ttl.attach_cb %[[INIT]], %[[CB2]]
  // CHECK-NEXT: %[[RESULT:.+]] = ttl.compute ins(%[[ARG0_CB]], %[[ARG1_CB]] : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<2x2x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN0:.+]]: !ttcore.tile<32x32, f32>, %[[IN1:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[LIN_IDX:.*]] = ttl.linearized_index
  // CHECK-NEXT:   %[[DTOK0:.*]], %[[DTILE0:.*]] = ttl.copy_tile %[[IN0]], %[[LIN_IDX]], %[[C0]]
  // CHECK-NEXT:   %[[DTOK1:.*]], %[[DTILE1:.*]] = ttl.copy_tile %[[IN1]], %[[LIN_IDX]], %[[C1]]
  // CHECK-NEXT:   %[[ADD:.+]] = ttl.tile_add %[[DTILE0]], %[[DTILE1]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   %[[VIEW:.+]] = ttl.cb_reserve %[[CB2]] : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT:   ttl.store %[[ADD]], %[[VIEW]]
  // CHECK-NEXT:   ttl.yield %[[ADD]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.add %a, %b : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %0, %cb2 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Unary elementwise operations (SFPU)

// CHECK-LABEL: func.func @unary_exp
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
func.func @unary_exp(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG:  %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK-NEXT: %[[INIT:.+]] = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[INIT_CB:.+]] = ttl.attach_cb %[[INIT]], %[[CB1]]
  // CHECK-NEXT: %[[RESULT:.+]] = ttl.compute ins(%[[ARG0_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-NEXT: ^bb0(%[[IN:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[LIN_IDX:.*]] = ttl.linearized_index
  // CHECK-NEXT:   %[[DTOK:.*]], %[[DTILE:.*]] = ttl.copy_tile %[[IN]], %[[LIN_IDX]], %[[C0]]
  // CHECK-NEXT:   %[[EXP:.+]] = ttl.tile_exp %[[DTILE]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   %[[VIEW:.+]] = ttl.cb_reserve %[[CB1]] : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT:   ttl.store %[[EXP]], %[[VIEW]]
  // CHECK-NEXT:   ttl.yield %[[EXP]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.exp %a : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %0, %cb1 : tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// Test: Chained elementwise operations produce multiple ttl.compute ops

// CHECK-LABEL: func.func @chained_ops
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[ARG1:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
func.func @chained_ops(%arg0: tensor<2x2x!ttcore.tile<32x32, f32>>, %arg1: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // CHECK-DAG:  %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG:  %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-NEXT: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-NEXT: %[[CB2:.+]] = ttl.bind_cb{cb_index = 2
  // CHECK-NEXT: %[[CB3:.+]] = ttl.bind_cb{cb_index = 3
  // CHECK-NEXT: %[[ARG0_CB:.+]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK-NEXT: %[[ARG1_CB:.+]] = ttl.attach_cb %[[ARG1]], %[[CB1]]
  // CHECK-NEXT: %[[EMPTY:.+]] = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: %[[EMPTY_CB:.+]] = ttl.attach_cb %[[EMPTY]], %[[CB2]]
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a = ttl.attach_cb %arg0, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // First compute: binary add
  // CHECK-NEXT: %[[ADD:.+]] = ttl.compute ins(%[[ARG0_CB]], %[[ARG1_CB]] : {{.*}}) outs(%[[EMPTY_CB]] : {{.*}}) {
  // CHECK-NEXT: ^bb0(%[[IN0:.+]]: !ttcore.tile<32x32, f32>, %[[IN1:.+]]: !ttcore.tile<32x32, f32>, %[[OUT0:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[LIN_IDX:.*]] = ttl.linearized_index
  // CHECK-NEXT:   %[[DTOK0:.*]], %[[DTILE0:.*]] = ttl.copy_tile %[[IN0]], %[[LIN_IDX]], %[[C0]]
  // CHECK-NEXT:   %[[DTOK1:.*]], %[[DTILE1:.*]] = ttl.copy_tile %[[IN1]], %[[LIN_IDX]], %[[C1]]
  // CHECK-NEXT:   %[[TILE_ADD:.+]] = ttl.tile_add %[[DTILE0]], %[[DTILE1]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   %[[VIEW0:.+]] = ttl.cb_reserve %[[CB2]] : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT:   ttl.store %[[TILE_ADD]], %[[VIEW0]]
  // CHECK-NEXT:   ttl.yield %[[TILE_ADD]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.add %a, %b : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %0, %cb2 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  // CHECK-NEXT: %[[ADD_CB:.+]] = ttl.attach_cb %[[ADD]], %[[CB2]]
  %add_cb = ttl.attach_cb %0, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Second compute: unary relu
  // CHECK-NEXT: %[[EMPTY_CB2:.+]] = ttl.attach_cb %[[EMPTY]], %[[CB3]]
  // CHECK-NEXT: %[[RESULT:.+]] = ttl.compute ins(%[[ADD_CB]] : {{.*}}) outs(%[[EMPTY_CB2]] : {{.*}}) {
  // CHECK-NEXT: ^bb0(%[[IN2:.+]]: !ttcore.tile<32x32, f32>, %[[OUT1:.+]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[LIN_IDX:.*]] = ttl.linearized_index
  // CHECK-NEXT:   %[[DTOK2:.*]], %[[DTILE2:.*]] = ttl.copy_tile %[[IN2]], %[[LIN_IDX]], %[[C0]]
  // CHECK-NEXT:   %[[TILE_RELU:.+]] = ttl.tile_relu %[[DTILE2]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
  // CHECK-NEXT:   %[[VIEW1:.+]] = ttl.cb_reserve %[[CB3]] : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT:   ttl.store %[[TILE_RELU]], %[[VIEW1]]
  // CHECK-NEXT:   ttl.yield %[[TILE_RELU]] : !ttcore.tile<32x32, f32>
  // CHECK-NEXT: } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %1 = ttl.relu %add_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %1, %cb3 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  func.return %1 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: All binary operations

// CHECK-LABEL: func.func @all_binary_ops
func.func @all_binary_ops(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb4 = ttl.bind_cb {cb_index = 4, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb5 = ttl.bind_cb {cb_index = 5, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb6 = ttl.bind_cb {cb_index = 6, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb7 = ttl.bind_cb {cb_index = 7, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb8 = ttl.bind_cb {cb_index = 8, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_add
  %add = ttl.add %a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %add, %cb2 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %add_cb = ttl.attach_cb %add, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb2 = ttl.attach_cb %b, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_sub
  %sub = ttl.sub %add_cb, %b_cb2 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %sub, %cb4 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %sub_cb = ttl.attach_cb %sub, %cb4 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_cb2 = ttl.attach_cb %a, %cb5 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_mul
  %mul = ttl.mul %sub_cb, %a_cb2 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %mul, %cb6 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %mul_cb = ttl.attach_cb %mul, %cb6 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb3 = ttl.attach_cb %b, %cb7 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_max
  %max = ttl.max %mul_cb, %b_cb3 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %max, %cb8 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  func.return %max : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: All unary operations

// CHECK-LABEL: func.func @all_unary_ops
func.func @all_unary_ops(%x: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb4 = ttl.bind_cb {cb_index = 4, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb5 = ttl.bind_cb {cb_index = 5, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb6 = ttl.bind_cb {cb_index = 6, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb7 = ttl.bind_cb {cb_index = 7, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb8 = ttl.bind_cb {cb_index = 8, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb9 = ttl.bind_cb {cb_index = 9, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %x_cb = ttl.attach_cb %x, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_exp
  %exp = ttl.exp %x_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %exp, %cb1 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %exp_cb = ttl.attach_cb %exp, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_log
  %log = ttl.log %exp_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %log, %cb2 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %log_cb = ttl.attach_cb %log, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_sqrt
  %sqrt = ttl.sqrt %log_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %sqrt, %cb3 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %sqrt_cb = ttl.attach_cb %sqrt, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_rsqrt
  %rsqrt = ttl.rsqrt %sqrt_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %rsqrt, %cb4 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %rsqrt_cb = ttl.attach_cb %rsqrt, %cb4 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_tanh
  %tanh = ttl.tanh %rsqrt_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %tanh, %cb5 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %tanh_cb = ttl.attach_cb %tanh, %cb5 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_sigmoid
  %sigmoid = ttl.sigmoid %tanh_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %sigmoid, %cb6 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %sigmoid_cb = ttl.attach_cb %sigmoid, %cb6 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_neg
  %neg = ttl.neg %sigmoid_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %neg, %cb7 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %neg_cb = ttl.attach_cb %neg, %cb7 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_abs
  %abs = ttl.abs %neg_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %abs, %cb8 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %abs_cb = ttl.attach_cb %abs, %cb8 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.tile_relu
  %relu = ttl.relu %abs_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %relu, %cb9 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  func.return %relu : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Test: DST assignment on chain of binary and unary ops

// CHECK-LABEL: func.func @dst_assignment_chain
// CHECK-SAME:  (%[[A:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[B:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>, %[[C:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
func.func @dst_assignment_chain(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x2x!ttcore.tile<32x32, f32>>, %c: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb3 = ttl.bind_cb {cb_index = 3, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb4 = ttl.bind_cb {cb_index = 4, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb5 = ttl.bind_cb {cb_index = 5, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
  // CHECK-DAG: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
  // CHECK-DAG: %[[CB2:.+]] = ttl.bind_cb{cb_index = 2
  // CHECK-DAG: %[[CB3:.+]] = ttl.bind_cb{cb_index = 3
  // CHECK-DAG: %[[CB4:.+]] = ttl.bind_cb{cb_index = 4
  // CHECK-DAG: %[[CB5:.+]] = ttl.bind_cb{cb_index = 5
  // CHECK: %[[A_CB:.+]] = ttl.attach_cb %[[A:arg0]], %[[CB0]]
  // CHECK: %[[B_CB:.+]] = ttl.attach_cb %[[B:arg1]], %[[CB1]]
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // First compute: binary add
  // CHECK: %[[COMP0:.+]] = ttl.compute
  // CHECK: ttl.tile_add{{.*}}{dst_idx = 0 : i32}
  %0 = ttl.add %a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %0, %cb2 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  // CHECK: %[[COMP0_CB:.+]] = ttl.attach_cb %[[COMP0]]
  %add_cb = ttl.attach_cb %0, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Second compute: unary exp
  // CHECK: %[[COMP1:.+]] = ttl.compute
  // CHECK: ttl.tile_exp{{.*}}{dst_idx = 0 : i32}
  %1 = ttl.exp %add_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %1, %cb3 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  // CHECK: %[[COMP1_CB:.+]] = ttl.attach_cb %[[COMP1]]
  // CHECK: %[[C_CB:.+]] = ttl.attach_cb %[[C]]
  %exp_cb = ttl.attach_cb %1, %cb3 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %c_cb = ttl.attach_cb %c, %cb4 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Third compute: binary mul
  // CHECK: %[[COMP2:.+]] = ttl.compute
  // CHECK: ttl.tile_mul{{.*}}{dst_idx = 0 : i32}
  // CHECK: return %[[COMP2]]
  %2 = ttl.mul %exp_cb, %c_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.tensor_store %2, %cb5 : tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  func.return %2 : tensor<2x2x!ttcore.tile<32x32, f32>>
}
