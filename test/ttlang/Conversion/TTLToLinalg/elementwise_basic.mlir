// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-linalg,ttl-assign-dst-registers),canonicalize)' | FileCheck %s

// Basic elementwise chain lowered to linalg.generic with DST annotation.

func.func @eltwise(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = ttl.add %arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  %1 = ttl.relu %0 : tensor<4x4xf32> -> tensor<4x4xf32>
  func.return %1 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @eltwise
// CHECK: arith.constant 0.000000e+00 : f32
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK-SAME: attrs = {ttl.dst_required = 16 : i32}
// CHECK: linalg.yield
// CHECK: arith.select
