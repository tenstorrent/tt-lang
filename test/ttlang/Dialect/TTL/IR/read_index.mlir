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
