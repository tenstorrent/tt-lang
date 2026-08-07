// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute))' | FileCheck %s

// The exp hardware flags (approx / scale / input_clamping / iterations) on the
// tensor-level ttl.exp are forwarded unchanged to the ttl.tile_exp op created
// inside the ttl.compute body.

// CHECK-LABEL: func.func @exp_flags_forwarded
func.func @exp_flags_forwarded(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK: ttl.compute
  // CHECK: %[[EXP:.*]] = ttl.tile_exp %{{.*}} into dst[%{{.*}}] {{[{].*}}approx = true{{.*}}input_clamping = 0 : i32{{.*}}iterations = 4 : i32{{.*}}scale = 5.000000e-01 : f32{{.*[}]}}
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.exp %a {approx = true, scale = 5.000000e-01 : f32,
                   input_clamping = 0 : i32, iterations = 4 : i32}
      : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}

// -----

// Accurate scaled exp materializes a tile multiply before plain exp because the
// current accurate BF16 LLK scale path requires an immediate but accepts scale
// as a runtime argument.

// CHECK-LABEL: func.func @exp_accurate_scale_materialized
func.func @exp_accurate_scale_materialized(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  // CHECK: ttl.compute
  // CHECK: %[[SCALED:.*]] = ttl.tile_mul_unary_const %{{.*}}, 2.000000e+00 into dst[%{{.*}}]
  // CHECK: ttl.tile_exp %[[SCALED]] into dst[%{{.*}}]
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %scaled = ttl.mul_unary_const %a, 2.000000e+00 : tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %exp = ttl.exp %scaled : tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.store %exp, %reserve : tensor<1x1x!ttcore.tile<32x32, f32>>, tensor<1x1x!ttcore.tile<32x32, f32>>
  func.return %exp : tensor<1x1x!ttcore.tile<32x32, f32>>
}
