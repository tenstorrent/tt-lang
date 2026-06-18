// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// Verify ttl.mul_unary_const and ttl.tile_mul_unary_const parse and print
// correctly. Both ops scale by a compile-time F32Attr; input and result types
// must match.

// CHECK-LABEL: func.func @mul_unary_const_tensor_bf16
// CHECK: ttl.mul_unary_const %{{.*}}, 5.000000e-01 : tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
func.func @mul_unary_const_tensor_bf16(
    %arg0: tensor<2x2x!ttcore.tile<32x32, bf16>>)
    -> tensor<2x2x!ttcore.tile<32x32, bf16>> {
  %0 = ttl.mul_unary_const %arg0, 5.000000e-01
      : tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  return %0 : tensor<2x2x!ttcore.tile<32x32, bf16>>
}

// -----

// CHECK-LABEL: func.func @mul_unary_const_tensor_f32
// CHECK: ttl.mul_unary_const %{{.*}}, -2.500000e-01 : tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
func.func @mul_unary_const_tensor_f32(
    %arg0: tensor<1x1x!ttcore.tile<32x32, f32>>)
    -> tensor<1x1x!ttcore.tile<32x32, f32>> {
  %0 = ttl.mul_unary_const %arg0, -2.500000e-01
      : tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  return %0 : tensor<1x1x!ttcore.tile<32x32, f32>>
}

// -----

// CHECK-LABEL: func.func @tile_mul_unary_const_bf16
// CHECK: ttl.tile_mul_unary_const %{{.*}}, 1.250000e+00 into dst[%{{.*}}] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
func.func @tile_mul_unary_const_bf16(%a: !ttcore.tile<32x32, bf16>)
    -> !ttcore.tile<32x32, bf16> {
  %c0 = arith.constant 0 : index
  %0 = ttl.tile_mul_unary_const %a, 1.250000e+00 into dst[%c0]
       : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
  return %0 : !ttcore.tile<32x32, bf16>
}

// -----

// CHECK-LABEL: func.func @tile_mul_unary_const_f32
// CHECK: ttl.tile_mul_unary_const %{{.*}}, 0.000000e+00 into dst[%{{.*}}] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
func.func @tile_mul_unary_const_f32(%a: !ttcore.tile<32x32, f32>)
    -> !ttcore.tile<32x32, f32> {
  %c0 = arith.constant 0 : index
  %0 = ttl.tile_mul_unary_const %a, 0.000000e+00 into dst[%c0]
       : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  return %0 : !ttcore.tile<32x32, f32>
}
