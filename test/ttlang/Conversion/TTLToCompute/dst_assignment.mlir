// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-assign-dst-registers),canonicalize)' | FileCheck %s

// Test: dst annotation succeeds within capacity (2x2 tiles = 4 <= 64).

func.func @ok(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = ttl.add %a, %b : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @ok
// CHECK: tensor.empty
// CHECK-NEXT: ttl.compute
// CHECK-SAME: ttl.dst_required = 4 : i32
// CHECK: ttl.tile_add
// CHECK: ttl.yield
