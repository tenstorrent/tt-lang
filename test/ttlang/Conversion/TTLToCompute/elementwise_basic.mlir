// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-assign-dst-registers),canonicalize)' | FileCheck %s

// Basic elementwise operations lowered to ttl.compute with tile ops and DST assignment.

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @binary_add
func.func @binary_add(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: %[[INIT:.*]] = tensor.empty
  // CHECK: %[[RESULT:.*]] = ttl.compute
  // CHECK-SAME: ins(%arg0, %arg1 : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-SAME: outs(%[[INIT]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-SAME: indexing_maps = [#map, #map, #map]
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]
  // CHECK: ^bb0(%[[LHS:.*]]: !ttcore.tile<32x32, f32>, %[[RHS:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK: %[[SUM:.*]] = ttl.tile_add %[[LHS]], %[[RHS]] {dst_idx = 0 : i32}
  // CHECK: ttl.yield %[[SUM]]
  %0 = ttl.add %arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @unary_exp
func.func @unary_exp(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: %[[INIT:.*]] = tensor.empty
  // CHECK: %[[RESULT:.*]] = ttl.compute
  // CHECK-SAME: ins(%arg0 : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-SAME: outs(%[[INIT]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK: ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK: %[[EXP:.*]] = ttl.tile_exp %[[IN]] {dst_idx = 0 : i32}
  // CHECK: ttl.yield %[[EXP]]
  %0 = ttl.exp %arg0 : tensor<4x4xf32> -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @chain_binary_unary
func.func @chain_binary_unary(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // First compute: binary add
  // CHECK: %[[INIT0:.*]] = tensor.empty
  // CHECK: %[[ADD_RESULT:.*]] = ttl.compute
  // CHECK-SAME: ins(%arg0, %arg1 : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-SAME: outs(%[[INIT0]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK: ^bb0(%[[LHS:.*]]: !ttcore.tile<32x32, f32>, %[[RHS:.*]]: !ttcore.tile<32x32, f32>, %[[OUT0:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK: %[[SUM:.*]] = ttl.tile_add %[[LHS]], %[[RHS]] {dst_idx = 0 : i32}
  // CHECK: ttl.yield %[[SUM]]
  %0 = ttl.add %arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>

  // Second compute: unary relu
  // CHECK: %[[INIT1:.*]] = tensor.empty
  // CHECK: %[[RELU_RESULT:.*]] = ttl.compute
  // CHECK-SAME: ins(%[[ADD_RESULT]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK-SAME: outs(%[[INIT1]] : tensor<4x4x!ttcore.tile<32x32, f32>>)
  // CHECK: ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, f32>, %[[OUT1:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK: %[[ACT:.*]] = ttl.tile_relu %[[IN]] {dst_idx = 0 : i32}
  // CHECK: ttl.yield %[[ACT]]
  %1 = ttl.relu %0 : tensor<4x4xf32> -> tensor<4x4xf32>

  func.return %1 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @multiple_binary
func.func @multiple_binary(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: ttl.compute
  // CHECK: ttl.tile_add{{.*}}dst_idx = 0
  %0 = ttl.add %a, %b : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_mul{{.*}}dst_idx = 0
  %1 = ttl.mul %0, %c : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>

  func.return %1 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @unary_chain
func.func @unary_chain(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: ttl.compute
  // CHECK: ttl.tile_abs{{.*}}dst_idx = 0
  %0 = ttl.abs %arg0 : tensor<4x4xf32> -> tensor<4x4xf32>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_sqrt{{.*}}dst_idx = 0
  %1 = ttl.sqrt %0 : tensor<4x4xf32> -> tensor<4x4xf32>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_log{{.*}}dst_idx = 0
  %2 = ttl.log %1 : tensor<4x4xf32> -> tensor<4x4xf32>

  func.return %2 : tensor<4x4xf32>
}
