// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-assign-dst-registers),canonicalize)' | FileCheck %s

// Basic elementwise chain lowered to ttl.compute with tile ops and DST assignment.

// CHECK-LABEL: func.func @eltwise
func.func @eltwise(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: ttl.compute
  // CHECK: ttl.tile_add{{.*}}dst_idx = 0
  // CHECK: ttl.yield
  %0 = ttl.add %arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_relu{{.*}}dst_idx = 0
  // CHECK: ttl.yield
  %1 = ttl.relu %0 : tensor<4x4xf32> -> tensor<4x4xf32>

  func.return %1 : tensor<4x4xf32>
}
