// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-assign-dst-registers),canonicalize)' | FileCheck %s

// Basic elementwise operations lowered to ttl.compute with tile ops and DST assignment.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @binary_add
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>, %[[ARG1:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>)
func.func @binary_add(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>, %arg1: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-NEXT: %[[INIT:.*]] = tensor.empty
  // CHECK-NEXT: %[[CB0:.*]] = ttl.bind_cb
  // CHECK-NEXT: %[[CB1:.*]] = ttl.bind_cb
  // CHECK-NEXT: %[[CB2:.*]] = ttl.bind_cb
  // CHECK-NEXT: %[[ARG0_CB:.*]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK-NEXT: %[[ARG1_CB:.*]] = ttl.attach_cb %[[ARG1]], %[[CB1]]
  // CHECK-NEXT: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT]], %[[CB2]]
  // CHECK-NEXT: %[[RESULT:.*]] = ttl.compute ins(%[[ARG0_CB]], %[[ARG1_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} {
  // CHECK-NEXT: ^bb0(%[[LHS:.*]]: !ttcore.tile<32x32, f32>, %[[RHS:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[SUM:.*]] = ttl.tile_add %[[LHS]], %[[RHS]] {dst_idx = 0 : i32}
  // CHECK-NEXT:   ttl.yield %[[SUM]]
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %0 = ttl.add %arg0, %arg1 : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// CHECK-LABEL: func.func @unary_exp
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>)
func.func @unary_exp(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK-NEXT: %[[INIT:.*]] = tensor.empty
  // CHECK-NEXT: %[[CB0:.*]] = ttl.bind_cb
  // CHECK-NEXT: %[[CB1:.*]] = ttl.bind_cb
  // CHECK-NEXT: %[[ARG0_CB:.*]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK-NEXT: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT]], %[[CB1]]
  // CHECK-NEXT: %[[RESULT:.*]] = ttl.compute ins(%[[ARG0_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) outs(%[[INIT_CB]] : tensor<4x4x!ttcore.tile<32x32, f32>>) {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} {
  // CHECK-NEXT: ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[EXP:.*]] = ttl.tile_exp %[[IN]] {dst_idx = 0 : i32}
  // CHECK-NEXT:   ttl.yield %[[EXP]]
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %0 = ttl.exp %arg0 : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// CHECK-LABEL: func.func @chain_binary_unary
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>, %[[ARG1:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>)
func.func @chain_binary_unary(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>, %arg1: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // First compute: binary add
  // CHECK-NEXT: %[[INIT0:.*]] = tensor.empty
  // CHECK-NEXT: %[[CB0:.*]] = ttl.bind_cb
  // CHECK-NEXT: %[[CB1:.*]] = ttl.bind_cb
  // CHECK-NEXT: %[[CB2:.*]] = ttl.bind_cb
  // CHECK-NEXT: %[[ARG0_CB:.*]] = ttl.attach_cb %[[ARG0]], %[[CB0]]
  // CHECK-NEXT: %[[ARG1_CB:.*]] = ttl.attach_cb %[[ARG1]], %[[CB1]]
  // CHECK-NEXT: %[[INIT0_CB:.*]] = ttl.attach_cb %[[INIT0]], %[[CB2]]
  // CHECK-NEXT: %[[ADD_RESULT:.*]] = ttl.compute ins(%[[ARG0_CB]], %[[ARG1_CB]] : {{.*}}) outs(%[[INIT0_CB]] : {{.*}}) {
  // CHECK-NEXT: ^bb0(%[[LHS:.*]]: !ttcore.tile<32x32, f32>, %[[RHS:.*]]: !ttcore.tile<32x32, f32>, %[[OUT0:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[SUM:.*]] = ttl.tile_add %[[LHS]], %[[RHS]] {dst_idx = 0 : i32}
  // CHECK-NEXT:   ttl.yield %[[SUM]]
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.add %arg0, %arg1 : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // Second compute: unary relu
  // CHECK-NEXT: %[[INIT1:.*]] = tensor.empty
  // CHECK-NEXT: %[[CB3:.*]] = ttl.bind_cb
  // CHECK-NEXT: %[[CB4:.*]] = ttl.bind_cb
  // CHECK-NEXT: %[[ADD_RESULT_CB:.*]] = ttl.attach_cb %[[ADD_RESULT]], %[[CB3]]
  // CHECK-NEXT: %[[INIT1_CB:.*]] = ttl.attach_cb %[[INIT1]], %[[CB4]]
  // CHECK-NEXT: %[[RELU_RESULT:.*]] = ttl.compute ins(%[[ADD_RESULT_CB]] : {{.*}}) outs(%[[INIT1_CB]] : {{.*}}) {
  // CHECK-NEXT: ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, f32>, %[[OUT1:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[ACT:.*]] = ttl.tile_relu %[[IN]] {dst_idx = 0 : i32}
  // CHECK-NEXT:   ttl.yield %[[ACT]]
  // CHECK-NEXT: } -> tensor<4x4x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RELU_RESULT]]
  %1 = ttl.relu %0 : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  func.return %1 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// CHECK-LABEL: func.func @multiple_binary
// CHECK-SAME: (%[[A:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>, %[[B:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>, %[[C:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>)
func.func @multiple_binary(%a: tensor<4x4x!ttcore.tile<32x32, f32>>, %b: tensor<4x4x!ttcore.tile<32x32, f32>>, %c: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK: %[[A_CB:.*]] = ttl.attach_cb %[[A]]
  // CHECK: %[[B_CB:.*]] = ttl.attach_cb %[[B]]
  // CHECK: %[[ADD:.*]] = ttl.compute
  // CHECK: ttl.tile_add{{.*}}dst_idx = 0
  %0 = ttl.add %a, %b : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: %[[ADD_CB:.*]] = ttl.attach_cb %[[ADD]]
  // CHECK: %[[C_CB:.*]] = ttl.attach_cb %[[C]]
  // CHECK: ttl.compute
  // CHECK: ttl.tile_mul{{.*}}dst_idx = 0
  %1 = ttl.mul %0, %c : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  func.return %1 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// CHECK-LABEL: func.func @unary_chain
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x!ttcore.tile<32x32, f32>>)
func.func @unary_chain(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK: %[[ARG0_CB:.*]] = ttl.attach_cb %[[ARG0]]
  // CHECK: %[[ABS:.*]] = ttl.compute
  // CHECK: ttl.tile_abs{{.*}}dst_idx = 0
  %0 = ttl.abs %arg0 : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: %[[ABS_CB:.*]] = ttl.attach_cb %[[ABS]]
  // CHECK: %[[SQRT:.*]] = ttl.compute
  // CHECK: ttl.tile_sqrt{{.*}}dst_idx = 0
  %1 = ttl.sqrt %0 : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  // CHECK: %[[SQRT_CB:.*]] = ttl.attach_cb %[[SQRT]]
  // CHECK: ttl.compute
  // CHECK: ttl.tile_log{{.*}}dst_idx = 0
  %2 = ttl.log %1 : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>

  func.return %2 : tensor<4x4x!ttcore.tile<32x32, f32>>
}
