// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// Verify ttl.exp and ttl.tile_exp parse and print their optional hardware
// flags (approx / scale / input_clamping / iterations). With no flags set the
// ops keep their plain spelling (all flags are default-valued or optional).

// CHECK-LABEL: func.func @exp_no_flags
// CHECK: ttl.exp %{{.*}} : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
func.func @exp_no_flags(
    %arg0: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %0 = ttl.exp %arg0
      : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// CHECK-LABEL: func.func @exp_with_flags
// CHECK: ttl.exp %{{.*}} {{[{].*}}approx = true{{.*}}input_clamping = 0 : i32{{.*}}iterations = 4 : i32{{.*}}scale = 2.000000e+00 : f32{{.*[}]}} : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
func.func @exp_with_flags(
    %arg0: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %0 = ttl.exp %arg0 {approx = true, scale = 2.000000e+00 : f32,
                      input_clamping = 0 : i32,
                      iterations = 4 : i32}
      : tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// CHECK-LABEL: func.func @tile_exp_no_flags
// CHECK: ttl.tile_exp %{{.*}} into dst[%{{.*}}] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
func.func @tile_exp_no_flags(%a: !ttcore.tile<32x32, f32>)
    -> !ttcore.tile<32x32, f32> {
  %c0 = arith.constant 0 : index
  %0 = ttl.tile_exp %a into dst[%c0]
       : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  return %0 : !ttcore.tile<32x32, f32>
}

// -----

// CHECK-LABEL: func.func @tile_exp_with_flags
// CHECK: ttl.tile_exp %{{.*}} into dst[%{{.*}}] {{[{].*}}approx = true{{.*}}input_clamping = 0 : i32{{.*}}iterations = 4 : i32{{.*}}scale = 2.000000e+00 : f32{{.*[}]}} : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
func.func @tile_exp_with_flags(%a: !ttcore.tile<32x32, f32>)
    -> !ttcore.tile<32x32, f32> {
  %c0 = arith.constant 0 : index
  %0 = ttl.tile_exp %a into dst[%c0] {approx = true, scale = 2.000000e+00 : f32,
                                      input_clamping = 0 : i32,
                                      iterations = 4 : i32}
       : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  return %0 : !ttcore.tile<32x32, f32>
}
