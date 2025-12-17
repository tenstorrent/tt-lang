// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),canonicalize)' | FileCheck %s

// Test all elementwise operations lower correctly to ttl.compute with tile ops.

// CHECK-LABEL: func.func @binary_ops
func.func @binary_ops(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: ttl.compute
  // CHECK: ttl.tile_add
  %add = ttl.add %a, %b : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_sub
  %sub = ttl.sub %add, %b : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_mul
  %mul = ttl.mul %sub, %a : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_max
  %max = ttl.max %mul, %b : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>

  func.return %max : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @unary_simple
func.func @unary_simple(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: ttl.compute
  // CHECK: ttl.tile_exp
  %exp = ttl.exp %x : tensor<2x2xf32> -> tensor<2x2xf32>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_log
  %log = ttl.log %exp : tensor<2x2xf32> -> tensor<2x2xf32>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_sqrt
  %sqrt = ttl.sqrt %log : tensor<2x2xf32> -> tensor<2x2xf32>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_rsqrt
  %rsqrt = ttl.rsqrt %sqrt : tensor<2x2xf32> -> tensor<2x2xf32>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_tanh
  %tanh = ttl.tanh %rsqrt : tensor<2x2xf32> -> tensor<2x2xf32>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_neg
  %neg = ttl.neg %tanh : tensor<2x2xf32> -> tensor<2x2xf32>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_abs
  %abs = ttl.abs %neg : tensor<2x2xf32> -> tensor<2x2xf32>

  func.return %abs : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @unary_custom
func.func @unary_custom(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: ttl.compute
  // CHECK: ttl.tile_relu
  %relu = ttl.relu %x : tensor<2x2xf32> -> tensor<2x2xf32>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_sigmoid
  %sigmoid = ttl.sigmoid %relu : tensor<2x2xf32> -> tensor<2x2xf32>

  func.return %sigmoid : tensor<2x2xf32>
}
