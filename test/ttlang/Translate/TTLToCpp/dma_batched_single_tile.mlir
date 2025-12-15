// RUN: ttlang-opt --ttl-to-ttkernel-pipeline --canonicalize %s -o %t.ttkernel.mlir
// RUN: ttmlir-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttmlir-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Test: Batched DMA operations
// Validates multiple async operations with proper barrier placement

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK: // dma_batched
// CHECK-NEXT: #include <cstdint>
// CHECK-NEXT: #include "tools/profiler/kernel_profiler.hpp"
// CHECK-NEXT: #include "dataflow_api.h"
// CHECK-NEXT: void kernel_main() {
// CHECK-DAG:   int32_t [[SIZE:v[0-9]+]] = 32;
// CHECK-DAG:   int32_t [[V1:v[0-9]+]] = 1;
// CHECK-DAG:   int32_t [[ADDR:v[0-9]+]] = 128;
// CHECK-DAG:   int32_t [[ZERO:v[0-9]+]] = 0;
// CHECK:   TensorAccessorArgs [[ARGS0:v[0-9]+]] = TensorAccessorArgs<32, 1>();
// CHECK-NEXT:   TensorAccessor [[ACCESSOR0:v[0-9]+]] = TensorAccessor([[ARGS0]], [[ZERO]], [[ADDR]]);
// CHECK-NEXT:   TensorAccessorArgs [[ARGS1:v[0-9]+]] = TensorAccessorArgs<32, 1>();
// CHECK-NEXT:   TensorAccessor [[ACCESSOR1:v[0-9]+]] = TensorAccessor([[ARGS1]], [[ZERO]], [[ADDR]]);
// CHECK-NEXT:   noc_async_read_tile([[ZERO]], [[ACCESSOR1]], [[ZERO]]);
// CHECK-NEXT:   noc_async_read_tile([[ZERO]], [[ACCESSOR0]], [[ZERO]]);
// Consecutive barriers deduplicated to single barrier.
// CHECK-NEXT:   noc_async_read_barrier();
// CHECK-NEXT:   return;
// CHECK-NEXT: }
module {
  func.func @dma_batched(%t0: tensor<32x32xf32, #layout>, %t1: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb0 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf0 = ttl.copy %t0, %cb0 : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf1 = ttl.copy %t1, %cb1 : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf0 : !ttl.transfer_handle<read>
    ttl.wait %xf1 : !ttl.transfer_handle<read>
    func.return
  }
}
