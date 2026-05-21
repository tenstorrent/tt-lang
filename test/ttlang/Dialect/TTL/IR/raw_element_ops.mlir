// Verifier acceptance tests for ttl.raw_element_read and ttl.raw_element_write.
// Each block operand must trace to a circular buffer (cb_wait or cb_reserve).
// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// -----

// Read a scalar f32 element from a 2D tiled block via cb_wait.
// CHECK-LABEL: func.func @raw_element_read_f32
// CHECK: %[[CB:.*]] = ttl.bind_cb
// CHECK: %[[VIEW:.*]] = ttl.cb_wait %[[CB]]
// CHECK: ttl.raw_element_read %[[VIEW]][%{{.*}}, %{{.*}}] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
func.func @raw_element_read_f32()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %val = ttl.raw_element_read %block[%c0, %c5] : tensor<1x1x!ttcore.tile<32x32, f32>> -> f32
  func.return
}

// -----

// Read a scalar bf16 element from a 2D tiled block via cb_wait.
// CHECK-LABEL: func.func @raw_element_read_bf16
// CHECK: ttl.raw_element_read %{{.*}}[%{{.*}}, %{{.*}}] : tensor<2x3x!ttcore.tile<32x32, bf16>> -> bf16
func.func @raw_element_read_bf16()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>
  %block = ttl.cb_wait %cb : <[2, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %val = ttl.raw_element_read %block[%c0, %c1] : tensor<2x3x!ttcore.tile<32x32, bf16>> -> bf16
  func.return
}

// -----

// Write a scalar f32 element to a 2D tiled block via cb_reserve.
// CHECK-LABEL: func.func @raw_element_write_f32
// CHECK: %[[CB:.*]] = ttl.bind_cb
// CHECK: %[[VIEW:.*]] = ttl.cb_reserve %[[CB]]
// CHECK: ttl.raw_element_write %[[VIEW]][%{{.*}}, %{{.*}}], %{{.*}} : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
func.func @raw_element_write_f32(%val: f32)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %block = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c7 = arith.constant 7 : index
  ttl.raw_element_write %block[%c0, %c7], %val : tensor<1x1x!ttcore.tile<32x32, f32>>, f32
  func.return
}

// -----

// Write a scalar bf16 element to a 2D tiled block via cb_reserve.
// CHECK-LABEL: func.func @raw_element_write_bf16
// CHECK: ttl.raw_element_write %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : tensor<2x3x!ttcore.tile<32x32, bf16>>, bf16
func.func @raw_element_write_bf16(%val: bf16)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>
  %block = ttl.cb_reserve %cb : <[2, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ttl.raw_element_write %block[%c0, %c1], %val : tensor<2x3x!ttcore.tile<32x32, bf16>>, bf16
  func.return
}

// -----

// Read from a 3D tiled block (higher rank).
// CHECK-LABEL: func.func @raw_element_read_3d
// CHECK: ttl.raw_element_read %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : tensor<2x3x4x!ttcore.tile<32x32, f32>> -> f32
func.func @raw_element_read_3d()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 3, 4], !ttcore.tile<32x32, f32>, 2>
  %block = ttl.cb_wait %cb : <[2, 3, 4], !ttcore.tile<32x32, f32>, 2> -> tensor<2x3x4x!ttcore.tile<32x32, f32>>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %val = ttl.raw_element_read %block[%c0, %c1, %c2] : tensor<2x3x4x!ttcore.tile<32x32, f32>> -> f32
  func.return
}

// -----

// Read a scalar f32 element from a row-major (non-tile) 2D block.
// CHECK-LABEL: func.func @raw_element_read_row_major_f32
// CHECK: ttl.raw_element_read %{{.*}}[%{{.*}}, %{{.*}}] : tensor<4x8xf32> -> f32
func.func @raw_element_read_row_major_f32()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 8], f32, 2>
  %block = ttl.cb_wait %cb : <[4, 8], f32, 2> -> tensor<4x8xf32>
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %val = ttl.raw_element_read %block[%c1, %c3] : tensor<4x8xf32> -> f32
  func.return
}

// -----

// Read a scalar bf16 element from a row-major (non-tile) 2D block.
// CHECK-LABEL: func.func @raw_element_read_row_major_bf16
// CHECK: ttl.raw_element_read %{{.*}}[%{{.*}}, %{{.*}}] : tensor<8x16xbf16> -> bf16
func.func @raw_element_read_row_major_bf16()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[8, 16], bf16, 2>
  %block = ttl.cb_wait %cb : <[8, 16], bf16, 2> -> tensor<8x16xbf16>
  %c2 = arith.constant 2 : index
  %c7 = arith.constant 7 : index
  %val = ttl.raw_element_read %block[%c2, %c7] : tensor<8x16xbf16> -> bf16
  func.return
}

// -----

// Write a scalar f32 element to a row-major (non-tile) 2D block.
// CHECK-LABEL: func.func @raw_element_write_row_major_f32
// CHECK: ttl.raw_element_write %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : tensor<4x8xf32>, f32
func.func @raw_element_write_row_major_f32(%val: f32)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4, 8], f32, 2>
  %block = ttl.cb_reserve %cb : <[4, 8], f32, 2> -> tensor<4x8xf32>
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  ttl.raw_element_write %block[%c2, %c5], %val : tensor<4x8xf32>, f32
  func.return
}

// -----

// Write a scalar bf16 element to a row-major (non-tile) 2D block.
// CHECK-LABEL: func.func @raw_element_write_row_major_bf16
// CHECK: ttl.raw_element_write %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : tensor<8x16xbf16>, bf16
func.func @raw_element_write_row_major_bf16(%val: bf16)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[8, 16], bf16, 2>
  %block = ttl.cb_reserve %cb : <[8, 16], bf16, 2> -> tensor<8x16xbf16>
  %c3 = arith.constant 3 : index
  %c10 = arith.constant 10 : index
  ttl.raw_element_write %block[%c3, %c10], %val : tensor<8x16xbf16>, bf16
  func.return
}

// -----

// Read a scalar f32 element from a rank-1 tiled block.
// CHECK-LABEL: func.func @raw_element_read_rank1_tiled
// CHECK: ttl.raw_element_read %{{.*}}[%{{.*}}] : tensor<4x!ttcore.tile<32x32, f32>> -> f32
func.func @raw_element_read_rank1_tiled()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %block = ttl.cb_wait %cb : <[4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x!ttcore.tile<32x32, f32>>
  %c2 = arith.constant 2 : index
  %val = ttl.raw_element_read %block[%c2] : tensor<4x!ttcore.tile<32x32, f32>> -> f32
  func.return
}

// -----

// Read a scalar f32 element from a rank-1 row-major block.
// CHECK-LABEL: func.func @raw_element_read_rank1_row_major
// CHECK: ttl.raw_element_read %{{.*}}[%{{.*}}] : tensor<128xf32> -> f32
func.func @raw_element_read_rank1_row_major()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[128], f32, 2>
  %block = ttl.cb_wait %cb : <[128], f32, 2> -> tensor<128xf32>
  %c42 = arith.constant 42 : index
  %val = ttl.raw_element_read %block[%c42] : tensor<128xf32> -> f32
  func.return
}

// -----

// Write a scalar f32 element to a rank-1 tiled block.
// CHECK-LABEL: func.func @raw_element_write_rank1_tiled
// CHECK: ttl.raw_element_write %{{.*}}[%{{.*}}], %{{.*}} : tensor<4x!ttcore.tile<32x32, f32>>, f32
func.func @raw_element_write_rank1_tiled(%val: f32)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[4], !ttcore.tile<32x32, f32>, 2>
  %block = ttl.cb_reserve %cb : <[4], !ttcore.tile<32x32, f32>, 2> -> tensor<4x!ttcore.tile<32x32, f32>>
  %c1 = arith.constant 1 : index
  ttl.raw_element_write %block[%c1], %val : tensor<4x!ttcore.tile<32x32, f32>>, f32
  func.return
}

// -----

// Write a scalar bf16 element to a rank-1 row-major block.
// CHECK-LABEL: func.func @raw_element_write_rank1_row_major
// CHECK: ttl.raw_element_write %{{.*}}[%{{.*}}], %{{.*}} : tensor<256xbf16>, bf16
func.func @raw_element_write_rank1_row_major(%val: bf16)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[256], bf16, 2>
  %block = ttl.cb_reserve %cb : <[256], bf16, 2> -> tensor<256xbf16>
  %c100 = arith.constant 100 : index
  ttl.raw_element_write %block[%c100], %val : tensor<256xbf16>, bf16
  func.return
}
