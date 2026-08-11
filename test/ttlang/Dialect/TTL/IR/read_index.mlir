// Verifier acceptance tests for ttl.read_index on supported block layouts and
// scalar element types.
// RUN: ttlang-opt %s | FileCheck %s

// Read an f32 element from a tiled block acquired by ttl.cb_wait.
// CHECK-LABEL: func.func @read_index_tiled_f32
// CHECK: %[[INDEX:.*]] = ttl.read_index %{{.*}}[%{{.*}}, %{{.*}}] : tensor<1x1x!ttcore.tile<32x32, f32>> -> index
// CHECK-NEXT: return %[[INDEX]] : index
func.func @read_index_tiled_f32() -> index
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %row = arith.constant 0 : index
  %column = arith.constant 5 : index
  %index = ttl.read_index %block[%row, %column] : tensor<1x1x!ttcore.tile<32x32, f32>> -> index
  func.return %index : index
}

// Compute consumers may read a waited tiled DFB.
// CHECK-LABEL: func.func @read_index_tiled_bf16_compute
// CHECK: %[[COMPUTE_INDEX:.*]] = ttl.read_index %{{.*}}[%{{.*}}, %{{.*}}] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> index
// CHECK-NEXT: return %[[COMPUTE_INDEX]] : index
func.func @read_index_tiled_bf16_compute() -> index
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 5, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %row = arith.constant 0 : index
  %column = arith.constant 0 : index
  %index = ttl.read_index %block[%row, %column] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> index
  func.return %index : index
}

// Read a bf16 element from a row-major block acquired by ttl.cb_wait.
// CHECK-LABEL: func.func @read_index_row_major_bf16
// CHECK: %[[INDEX:.*]] = ttl.read_index %{{.*}}[%{{.*}}] : tensor<128xbf16> -> index
// CHECK-NEXT: return %[[INDEX]] : index
func.func @read_index_row_major_bf16() -> index
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[128], bf16, 2>
  %block = ttl.cb_wait %cb : <[128], bf16, 2> -> tensor<128xbf16>
  %position = arith.constant 42 : index
  %index = ttl.read_index %block[%position] : tensor<128xbf16> -> index
  func.return %index : index
}

// Read unsigned integer elements without losing values whose high bit is set.
// CHECK-LABEL: func.func @read_index_tiled_ui8
// CHECK: %[[UI8_INDEX:.*]] = ttl.read_index %{{.*}}[%{{.*}}, %{{.*}}] : tensor<1x1x!ttcore.tile<1x32, u8>> -> index
// CHECK-NEXT: return %[[UI8_INDEX]] : index
func.func @read_index_tiled_ui8() -> index
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<1x32, u8>, 2>
  %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<1x32, u8>, 2> -> tensor<1x1x!ttcore.tile<1x32, u8>>
  %row = arith.constant 0 : index
  %column = arith.constant 15 : index
  %index = ttl.read_index %block[%row, %column] : tensor<1x1x!ttcore.tile<1x32, u8>> -> index
  func.return %index : index
}

// CHECK-LABEL: func.func @read_index_row_major_ui16
// CHECK: %[[UI16_INDEX:.*]] = ttl.read_index %{{.*}}[%{{.*}}] : tensor<32xui16> -> index
// CHECK-NEXT: return %[[UI16_INDEX]] : index
func.func @read_index_row_major_ui16() -> index
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[32], ui16, 2>
  %block = ttl.cb_wait %cb : <[32], ui16, 2> -> tensor<32xui16>
  %position = arith.constant 7 : index
  %index = ttl.read_index %block[%position] : tensor<32xui16> -> index
  func.return %index : index
}

// CHECK-LABEL: func.func @read_index_tiled_ui32
// CHECK: %[[UI32_INDEX:.*]] = ttl.read_index %{{.*}}[%{{.*}}, %{{.*}}] : tensor<1x1x!ttcore.tile<1x32, u32>> -> index
// CHECK-NEXT: return %[[UI32_INDEX]] : index
func.func @read_index_tiled_ui32() -> index
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 4, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<1x32, u32>, 2>
  %block = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<1x32, u32>, 2> -> tensor<1x1x!ttcore.tile<1x32, u32>>
  %row = arith.constant 0 : index
  %column = arith.constant 3 : index
  %index = ttl.read_index %block[%row, %column] : tensor<1x1x!ttcore.tile<1x32, u32>> -> index
  func.return %index : index
}
