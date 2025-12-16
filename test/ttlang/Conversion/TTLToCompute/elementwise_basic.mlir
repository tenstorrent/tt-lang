// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-assign-dst-registers),canonicalize)' | FileCheck %s

// Basic elementwise chain lowered to ttl.compute with tile ops and DST annotation.

// CHECK-LABEL: func.func @eltwise
func.func @eltwise(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: ttl.compute
  // CHECK-SAME: ttl.dst_required = 16 : i32
  // CHECK: ttl.tile_add
  // CHECK: ttl.yield
  %0 = ttl.add %arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>

  // CHECK: ttl.compute
  // CHECK-SAME: ttl.dst_required = 16 : i32
  // CHECK: ttl.tile_relu
  // CHECK: ttl.yield
  %1 = ttl.relu %0 : tensor<4x4xf32> -> tensor<4x4xf32>

  func.return %1 : tensor<4x4xf32>
}
