// RUN: ttlang-opt --split-input-file %s | FileCheck %s
// RUN: ttlang-opt --convert-ttl-to-ttkernel --split-input-file %s | FileCheck %s --check-prefix=LOWERED
// Summary: MVP DMA lowering tests for tensor<->CB copies (no pipes).

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: func.func @dma_single
// CHECK: ttl.create_cb
// CHECK: ttl.copy
// CHECK: ttl.wait

// LOWERED-LABEL: func.func @dma_single
// LOWERED: ttkernel.TensorAccessorArgs
// LOWERED: ttkernel.TensorAccessor
// LOWERED: ttkernel.noc_async_read_tile
// LOWERED: ttkernel.noc_async_read_barrier
// LOWERED: ttkernel.noc_async_write_tile
// LOWERED: ttkernel.noc_async_write_barrier
module {
  func.func @dma_single(%arg0: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %arg0, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
    ttl.wait %xf
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: func.func @cb_to_tensor
// CHECK: ttl.create_cb
// CHECK: ttl.copy
// CHECK: ttl.wait

// LOWERED-LABEL: func.func @cb_to_tensor
// LOWERED: ttkernel.TensorAccessorArgs
// LOWERED: ttkernel.TensorAccessor
// LOWERED: ttkernel.noc_async_write_tile
// LOWERED: ttkernel.noc_async_write_barrier
module {
  func.func @cb_to_tensor(%arg0: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %cb, %arg0 : (!ttl.cb<[1, 1], f32, 2>, tensor<32x32xf32, #layout>) -> !ttl.xf
    ttl.wait %xf
    func.return
  }
}
