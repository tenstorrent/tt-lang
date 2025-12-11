// RUN: ttlang-opt --split-input-file %s | FileCheck %s
// RUN: ttlang-opt --convert-ttl-to-ttkernel --split-input-file %s | FileCheck %s --check-prefix=LOWERED

// CHECK-LABEL: func.func @dma_single
// CHECK: ttl.create_cb
// CHECK: ttl.copy
// CHECK: ttl.wait

// LOWERED-LABEL: func.func @dma_single
// LOWERED: ttkernel.TensorAccessorArgs
// LOWERED: ttkernel.TensorAccessor
// LOWERED: ttkernel.noc_async_read_tile
// LOWERED: ttkernel.noc_async_read_barrier
module {
  func.func @dma_single(%arg0: tensor<1x1xf32>) {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %arg0, %cb : (tensor<1x1xf32>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.xf
    ttl.wait %xf
    return
  }
}


