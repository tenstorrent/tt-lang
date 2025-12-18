// RUN: ttlang-opt %s --one-shot-bufferize="bufferize-function-boundaries=false allow-unknown-ops=false" | FileCheck %s
// Summary: Ensure ttl.copy bufferizes by swapping tensor operands for memrefs
// and injecting tensor_type metadata so downstream passes retain TTNN layout.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  // CHECK-LABEL: func.func @tensor_to_cb
  // CHECK-SAME: (%[[ARG0:.*]]: tensor<32x32xf32, #ttnn_layout>)
  // CHECK-NEXT:   %[[BUF:.*]] = bufferization.to_buffer %[[ARG0]] : tensor<32x32xf32, #ttnn_layout> to memref<32x32xf32, strided<[?, ?], offset: ?>>
  // CHECK-NEXT:   %[[CB:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[1, 1], f32, 2>
  // CHECK-NEXT:   %[[XF:.*]] = ttl.copy %[[BUF]], %[[CB]] {tensor_type = tensor<32x32xf32, #ttnn_layout>} : (memref<32x32xf32, strided<[?, ?], offset: ?>>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
  // CHECK-NEXT:   ttl.wait %[[XF]] : !ttl.transfer_handle<read>
  // CHECK-NEXT:   return
  func.func @tensor_to_cb(%arg0: tensor<32x32xf32, #ttnn_layout>) {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %arg0, %cb
        : (tensor<32x32xf32, #ttnn_layout>, !ttl.cb<[1, 1], f32, 2>)
        -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }

  // CHECK-LABEL: func.func @cb_to_tensor
  // CHECK-SAME: (%[[ARG0:.*]]: tensor<32x32xf32, #ttnn_layout>)
  // CHECK-NEXT:   %[[CB:.*]] = ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[1, 1], f32, 2>
  // CHECK-NEXT:   %[[ALLOC:.*]] = memref.alloc(){{.*}} : memref<32x32xf32>
  // CHECK-NEXT:   %[[XF:.*]] = ttl.copy %[[CB]], %[[ALLOC]] {tensor_type = tensor<32x32xf32, #ttnn_layout>} : (!ttl.cb<[1, 1], f32, 2>, memref<32x32xf32>) -> !ttl.transfer_handle<write>
  // CHECK-NEXT:   ttl.wait %[[XF]] : !ttl.transfer_handle<write>
  // CHECK-NEXT:   return
  func.func @cb_to_tensor(%arg0: tensor<32x32xf32, #ttnn_layout>) {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %cb, %arg0
        : (!ttl.cb<[1, 1], f32, 2>, tensor<32x32xf32, #ttnn_layout>)
        -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
    func.return
  }
}

