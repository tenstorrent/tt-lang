// RUN: ttlang-opt %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>,
           memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: func.func @dma_single
// CHECK: %[[CB:.*]] = ttl.create_cb() {buffer_factor = 2 : i64, element_type = f32, shape = [1, 1]} : <[1, 1], f32, 2>
// CHECK: %[[XF:.*]] = ttl.copy %[[T:.*]], %[[CB]] : (tensor<32x32xf32, #ttnn_layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
// CHECK: ttl.wait %[[XF]] : !ttl.transfer_handle<read>
module {
  func.func @dma_single(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}
