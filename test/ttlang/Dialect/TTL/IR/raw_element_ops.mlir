// Verifier acceptance tests for ttl.raw_element_read and ttl.raw_element_write.
// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// -----

// Read a scalar f32 element from a 2D tiled block.
// CHECK-LABEL: func.func @raw_element_read_f32
// CHECK-SAME: (%[[BLOCK:.*]]: tensor<1x1x!ttcore.tile<32x32, f32>>)
// CHECK: %[[VAL:.*]] = ttl.raw_element_read %[[BLOCK]][%{{.*}}, %{{.*}}] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
func.func @raw_element_read_f32(
    %block: tensor<1x1x!ttcore.tile<32x32, f32>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %val = ttl.raw_element_read %block[%c0, %c5] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
  func.return
}

// -----

// Read a scalar bf16 element from a 2D tiled block.
// CHECK-LABEL: func.func @raw_element_read_bf16
// CHECK: %[[VAL:.*]] = ttl.raw_element_read %{{.*}}[%{{.*}}, %{{.*}}] : tensor<2x3x!ttcore.tile<32x32, bf16>> -> bf16
func.func @raw_element_read_bf16(
    %block: tensor<2x3x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %val = ttl.raw_element_read %block[%c0, %c1] : tensor<2x3x!ttcore.tile<32x32, bf16>> -> bf16
  func.return
}

// -----

// Write a scalar f32 element to a 2D tiled block.
// CHECK-LABEL: func.func @raw_element_write_f32
// CHECK-SAME: (%[[BLOCK:.*]]: tensor<1x1x!ttcore.tile<32x32, f32>>, %[[VAL:.*]]: f32)
// CHECK: ttl.raw_element_write %[[BLOCK]][%{{.*}}, %{{.*}}], %[[VAL]] : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
func.func @raw_element_write_f32(
    %block: tensor<1x1x!ttcore.tile<32x32, f32>>, %val: f32)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %c7 = arith.constant 7 : index
  ttl.raw_element_write %block[%c0, %c7], %val : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
  func.return
}

// -----

// Write a scalar bf16 element to a 2D tiled block.
// CHECK-LABEL: func.func @raw_element_write_bf16
// CHECK: ttl.raw_element_write %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : tensor<2x3x!ttcore.tile<32x32, bf16>>, bf16
func.func @raw_element_write_bf16(
    %block: tensor<2x3x!ttcore.tile<32x32, bf16>>, %val: bf16)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttl.raw_element_write %block[%c0, %c1], %val : tensor<2x3x!ttcore.tile<32x32, bf16>>, bf16
  func.return
}

// -----

// Read from a 3D tiled block (higher rank).
// CHECK-LABEL: func.func @raw_element_read_3d
// CHECK: %[[VAL:.*]] = ttl.raw_element_read %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : tensor<2x3x4x!ttcore.tile<32x32, f32>> -> f32
func.func @raw_element_read_3d(
    %block: tensor<2x3x4x!ttcore.tile<32x32, f32>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %val = ttl.raw_element_read %block[%c0, %c1, %c2] : tensor<2x3x4x!ttcore.tile<32x32, f32>> -> f32
  func.return
}
