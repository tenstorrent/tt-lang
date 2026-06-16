// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute))' | FileCheck %s

// The exp hardware flags (approx / scale / input_clamping / iterations) on the
// tensor-level ttl.exp are forwarded unchanged to the ttl.tile_exp op created
// inside the ttl.compute body.

// CHECK-LABEL: func.func @exp_flags_forwarded
func.func @exp_flags_forwarded(%arg0: tensor<4x4x!ttcore.tile<32x32, f32>>) -> tensor<4x4x!ttcore.tile<32x32, f32>> {
  // CHECK: ttl.compute
  // CHECK: %[[EXP:.*]] = ttl.tile_exp %{{.*}} into dst[%{{.*}}] {{[{].*}}approx = true{{.*}}iterations = 4 : i32{{.*}}scale = 5.000000e-01 : f32{{.*[}]}}
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<4x4x!ttcore.tile<32x32, f32>>, !ttl.cb<[4, 4], !ttcore.tile<32x32, f32>, 2>) -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %reserve = ttl.cb_reserve %cb1 : <[4, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  %0 = ttl.exp %a {approx = true, scale = 5.000000e-01 : f32,
                   input_clamping = #ttl.input_clamping<none>, iterations = 4 : i32}
      : tensor<4x4x!ttcore.tile<32x32, f32>> -> tensor<4x4x!ttcore.tile<32x32, f32>>
  ttl.store %0, %reserve : tensor<4x4x!ttcore.tile<32x32, f32>>, tensor<4x4x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<4x4x!ttcore.tile<32x32, f32>>
}
