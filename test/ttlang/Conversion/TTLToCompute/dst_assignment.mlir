// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute,ttl-assign-dst-registers),canonicalize)' | FileCheck %s

// Test: DST assignment assigns dst_idx attributes to tile ops. This is
// mostly a placeholder, there will be extensive tests after the DST pass is added.

func.func @ok(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = ttl.add %a, %b : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @ok
// CHECK: tensor.empty
// CHECK-NEXT: ttl.compute
// CHECK: ttl.tile_add{{.*}}dst_idx = 0
// CHECK: ttl.yield
