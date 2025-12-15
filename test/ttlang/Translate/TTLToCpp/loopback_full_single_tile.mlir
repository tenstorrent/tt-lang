// RUN: ttlang-opt --ttl-to-ttkernel-pipeline --canonicalize %s -o %t.ttkernel.mlir
// RUN: ttmlir-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttmlir-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Test: Full loopback pattern (read from DRAM, write back to DRAM)
// Validates complete pattern matching production kernel structure

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK: // loopback
// CHECK-NEXT: #include <cstdint>
// CHECK-NEXT: #include "tools/profiler/kernel_profiler.hpp"
// CHECK-NEXT: #include "dataflow_api.h"
// CHECK-NEXT: void kernel_main() {
// CHECK-NEXT:   int32_t [[ADDR:v[0-9]+]] = 128;
// CHECK-NEXT:   int32_t [[V1:v[0-9]+]] = 1;
// CHECK-NEXT:   int32_t [[SIZE:v[0-9]+]] = 32;
// CHECK-NEXT:   size_t [[STEP:v[0-9]+]] = 1;
// CHECK-NEXT:   size_t [[UB:v[0-9]+]] = 4;
// CHECK-NEXT:   size_t [[LB:v[0-9]+]] = 0;
// CHECK-NEXT:   int32_t [[ZERO:v[0-9]+]] = 0;
// CHECK-NEXT:   for (size_t [[IV:i[0-9]+]] = [[LB]]; [[IV]] < [[UB]]; [[IV]] += [[STEP]]) {
// CHECK-NEXT:     TensorAccessorArgs [[ARGS0:v[0-9]+]] = TensorAccessorArgs<32, 1>();
// CHECK-NEXT:     TensorAccessor [[ACCESSOR0:v[0-9]+]] = TensorAccessor([[ARGS0]], [[ZERO]], [[ADDR]]);
// CHECK-NEXT:     noc_async_read_tile([[ZERO]], [[ACCESSOR0]], [[ZERO]]);
// CHECK-NEXT:     noc_async_read_barrier();
// CHECK-NEXT:     TensorAccessorArgs [[ARGS1:v[0-9]+]] = TensorAccessorArgs<32, 1>();
// CHECK-NEXT:     TensorAccessor [[ACCESSOR1:v[0-9]+]] = TensorAccessor([[ARGS1]], [[ZERO]], [[ADDR]]);
// CHECK-NEXT:     noc_async_write_tile([[ZERO]], [[ACCESSOR1]], [[ZERO]]);
// CHECK-NEXT:     noc_async_write_barrier();
// CHECK-NEXT:   }
// CHECK-NEXT:   return;
// CHECK-NEXT: }
module {
  func.func @loopback(%src: tensor<32x32xf32, #layout>, %dst: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %c4 step %c1 {
      %xf_read = ttl.copy %src, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
      ttl.wait %xf_read : !ttl.transfer_handle<read>
      %xf_write = ttl.copy %cb, %dst : (!ttl.cb<[1, 1], f32, 2>, tensor<32x32xf32, #layout>) -> !ttl.transfer_handle<write>
      ttl.wait %xf_write : !ttl.transfer_handle<write>
    }
    func.return
  }
}
