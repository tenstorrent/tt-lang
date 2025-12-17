// RUN: ttlang-opt --ttl-to-ttkernel-pipeline --canonicalize %s -o %t.ttkernel.mlir
// RUN: ttmlir-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttmlir-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Test: Single DMA write operation (CB → tensor)
// Validates write barrier placement and ensures no read barrier

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK: // cb_to_tensor
// CHECK-NEXT: #include <cstdint>
// CHECK-NEXT: #include "tools/profiler/kernel_profiler.hpp"
// CHECK-NEXT: #include "dataflow_api.h"
// CHECK-NEXT: void kernel_main() {
// CHECK-DAG:   int32_t [[SIZE:v[0-9]+]] = 32;
// CHECK-DAG:   int32_t [[V1:v[0-9]+]] = 1;
// CHECK-DAG:   int32_t [[ADDR:v[0-9]+]] = 128;
// CHECK-DAG:   int32_t [[ZERO:v[0-9]+]] = 0;
// CHECK:   TensorAccessorArgs [[ARGS:v[0-9]+]] = TensorAccessorArgs<32, 1>();
// CHECK:   TensorAccessor [[ACCESSOR:v[0-9]+]] = TensorAccessor([[ARGS]], [[ZERO]], [[ADDR]]);
// CHECK-NEXT:   noc_async_write_tile([[ZERO]], [[ACCESSOR]], [[ZERO]]);
// CHECK-NEXT:   noc_async_write_barrier();
// CHECK-NEXT:   return;
// CHECK-NEXT: }
module {
  func.func @cb_to_tensor(%arg0: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %cb, %arg0 : (!ttl.cb<[1, 1], f32, 2>, tensor<32x32xf32, #layout>) -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
    func.return
  }
}
