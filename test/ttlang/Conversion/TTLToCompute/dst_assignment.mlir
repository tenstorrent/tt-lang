// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-tile-and-assign-dst),canonicalize)' | FileCheck %s

// Test: token-based lowering without dst_idx attributes.

func.func @ok(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = ttl.add %a, %b : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @ok
// CHECK: tensor.empty
// CHECK: ttl.compute
// CHECK: ttl.tile_add
// CHECK-NOT: dst_idx
// CHECK: ttl.yield
