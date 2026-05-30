// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),canonicalize)' | FileCheck %s

// Tests fused tensor elementwise lowering to ttl.compute.

// Verifies that exp(a) + sqrt(b) lowers into one ttl.compute.
// CHECK-LABEL: func.func @fused_exp_sqrt_add
func.func @fused_exp_sqrt_add(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // These should be fused into a single ttl.compute
  %reserve = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %exp = ttl.exp %a_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %sqrt = ttl.sqrt %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %add = ttl.add %exp, %sqrt : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %add, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK-SAME: ins(%{{.*}}, %{{.*}}:
  // CHECK: ttl.tile_exp
  // CHECK: ttl.tile_sqrt
  // CHECK: ttl.tile_add
  // CHECK: ttl.tile_store

  func.return %add : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Verifies that exp(a) + b lowers into one ttl.compute.
// CHECK-LABEL: func.func @fused_exp_add_direct
func.func @fused_exp_add_direct(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %exp = ttl.exp %a_cb : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %add = ttl.add %exp, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %add, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK: ttl.tile_exp
  // CHECK: ttl.tile_add
  // CHECK: ttl.tile_store

  func.return %add : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Verifies that c * (a + b) lowers into one ttl.compute with no intermediate
// dataflow buffer for the add result.
// CHECK-LABEL: func.func @fused_add_mul_unary_const
func.func @fused_add_mul_unary_const(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  %reserve = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %add = ttl.add %a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %scaled = ttl.mul_unary_const %add, 5.000000e-01 : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %scaled, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: ttl.compute
  // CHECK-SAME: ins(%{{.*}}, %{{.*}}:
  // CHECK: ttl.tile_add
  // CHECK: ttl.tile_mul_unary_const
  // CHECK: ttl.tile_store
  // CHECK-NOT: ttl.compute

  func.return %scaled : tensor<2x2x!ttcore.tile<32x32, f32>>
}
