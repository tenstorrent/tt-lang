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
// CHECK-DAG:   int32_t [[ZERO:v[0-9]+]] = 0;
// CHECK-DAG:   int32_t [[ADDR:v[0-9]+]] = 128;
// CHECK-DAG:   size_t [[STEP:v[0-9]+]] = 1;
// CHECK-DAG:   size_t [[UB:v[0-9]+]] = 4;
// CHECK-DAG:   size_t [[LB:v[0-9]+]] = 0;
// CHECK:   for (size_t [[IV:i[0-9]+]] = [[LB]]; [[IV]] < [[UB]]; [[IV]] += [[STEP]]) {
// Read: tensor → CB (uses get_write_ptr for CB destination)
// CHECK:     int32_t [[RT_ARG_R:v[0-9]+]] = get_common_arg_val<uint32_t>([[LB]]);
// CHECK:     TensorAccessorArgs [[ARGS_READ:v[0-9]+]] = TensorAccessorArgs<32, 1>();
// CHECK:     TensorAccessor [[ACC_READ:v[0-9]+]] = TensorAccessor([[ARGS_READ]], [[RT_ARG_R]], [[ADDR]]);
// CHECK:     int32_t [[CB_WRITE_PTR:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(0));
// CHECK:     noc_async_read_tile([[ZERO]], [[ACC_READ]], [[CB_WRITE_PTR]]);
// CHECK:     noc_async_read_barrier();
// Write: CB → tensor (uses get_read_ptr for CB source)
// CHECK:     int32_t [[RT_ARG_W:v[0-9]+]] = get_common_arg_val<uint32_t>([[STEP]]);
// CHECK:     TensorAccessorArgs [[ARGS_WRITE:v[0-9]+]] = TensorAccessorArgs<32, 1>();
// CHECK:     TensorAccessor [[ACC_WRITE:v[0-9]+]] = TensorAccessor([[ARGS_WRITE]], [[RT_ARG_W]], [[ADDR]]);
// CHECK:     int32_t [[CB_READ_PTR:v[0-9]+]] = get_read_ptr(get_compile_time_arg_val(0));
// CHECK:     noc_async_write_tile([[ZERO]], [[ACC_WRITE]], [[CB_READ_PTR]]);
// CHECK:     noc_async_write_barrier();
// CHECK:   }
// CHECK:   return;
// CHECK-NEXT: }
module {
  func.func @loopback(%src: tensor<32x32xf32, #layout>, %dst: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
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
