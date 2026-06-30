// RUN: ttlang-opt %s --canonicalize --split-input-file | FileCheck %s

// Verify ttl.typecast and ttl.tile_typecast parse and print correctly with
// different in/out element types, and identity typecasts fold to their inputs.
// The destination dtype is encoded in the result tile element type only.

// CHECK-LABEL: func.func @typecast_tensor_bf16_to_f32
// CHECK: ttl.typecast %{{.*}} : (tensor<2x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
func.func @typecast_tensor_bf16_to_f32(
    %arg0: tensor<2x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %0 = ttl.typecast %arg0
      : (tensor<2x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// CHECK-LABEL: func.func @typecast_tile_bf16_to_f32
// CHECK: ttl.tile_typecast %{{.*}} into dst[%{{.*}}] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
func.func @typecast_tile_bf16_to_f32(%a: !ttcore.tile<32x32, bf16>)
    -> !ttcore.tile<32x32, f32> {
  %c0 = arith.constant 0 : index
  %0 = ttl.tile_typecast %a into dst[%c0]
       : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
  return %0 : !ttcore.tile<32x32, f32>
}

// -----

// CHECK-LABEL: func.func @typecast_tile_f32_to_bf16
// CHECK: ttl.tile_typecast %{{.*}} into dst[%{{.*}}] : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, bf16>
func.func @typecast_tile_f32_to_bf16(%a: !ttcore.tile<32x32, f32>)
    -> !ttcore.tile<32x32, bf16> {
  %c0 = arith.constant 0 : index
  %0 = ttl.tile_typecast %a into dst[%c0]
       : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, bf16>
  return %0 : !ttcore.tile<32x32, bf16>
}

// -----

// CHECK-LABEL: func.func @typecast_tensor_identity
// CHECK-SAME: (%[[ARG:.*]]: tensor<2x2x!ttcore.tile<32x32, f32>>)
// CHECK-NOT: ttl.typecast
// CHECK: return %[[ARG]] : tensor<2x2x!ttcore.tile<32x32, f32>>
func.func @typecast_tensor_identity(
    %arg0: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %0 = ttl.typecast %arg0
      : (tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// CHECK-LABEL: func.func @typecast_tile_identity
// CHECK-SAME: (%[[ARG:.*]]: !ttcore.tile<32x32, f32>)
// CHECK-NOT: ttl.tile_typecast
// CHECK: return %[[ARG]] : !ttcore.tile<32x32, f32>
func.func @typecast_tile_identity(%a: !ttcore.tile<32x32, f32>)
    -> !ttcore.tile<32x32, f32> {
  %c0 = arith.constant 0 : index
  %0 = ttl.tile_typecast %a into dst[%c0]
       : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
  return %0 : !ttcore.tile<32x32, f32>
}
