// RUN: ttlang-opt --ttl-to-ttkernel-pipeline --canonicalize %s -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Test: DMA operations inside loop
// Validates scf.for → C++ for loop with DMA operations

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK: // dma_pipelined_loop
// CHECK: void kernel_main() {
// CHECK-DAG:   int32_t [[ZERO:v[0-9]+]] = 0;
// CHECK-DAG:   int32_t [[ADDR:v[0-9]+]] = 4096;
// CHECK-DAG:   size_t [[STEP:v[0-9]+]] = 1;
// CHECK-DAG:   size_t [[UB:v[0-9]+]] = 3;
// CHECK-DAG:   size_t [[LB:v[0-9]+]] = 0;
// Pre-loop copy: create accessor with runtime arg, get CB write ptr
// CHECK:   int32_t [[RT_ARG0:v[0-9]+]] = get_common_arg_val<uint32_t>([[LB]]);
// CHECK:   auto [[ARGS0:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<1, 0>();
// CHECK:   TensorAccessor [[ACCESSOR0:v[0-9]+]] = TensorAccessor([[ARGS0]], [[RT_ARG0]], [[ADDR]]);
// CHECK:   int32_t [[CB_PTR0:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(0));
// CHECK:   noc_async_read_tile([[ZERO]], [[ACCESSOR0]], [[CB_PTR0]]);
// CHECK:   for (size_t [[IV:i[0-9]+]] = [[LB]]; [[IV]] < [[UB]]; [[IV]] += [[STEP]]) {
// In-loop copy: create accessor with runtime arg, get CB write ptr
// CHECK:     int32_t [[RT_ARG1:v[0-9]+]] = get_common_arg_val<uint32_t>([[LB]]);
// CHECK:     auto [[ARGS1:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<1, 0>();
// CHECK:     TensorAccessor [[ACCESSOR1:v[0-9]+]] = TensorAccessor([[ARGS1]], [[RT_ARG1]], [[ADDR]]);
// CHECK:     int32_t [[CB_PTR1:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(0));
// CHECK:     noc_async_read_tile([[ZERO]], [[ACCESSOR1]], [[CB_PTR1]]);
// CHECK:     noc_async_read_barrier();
// CHECK:   }
// CHECK:   noc_async_read_barrier();
// CHECK:   return;
// CHECK-NEXT: }
module {
  func.func @dma_pipelined_loop(%t: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index

    %slice = ttl.tensor_slice %t[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf_init = ttl.copy %slice, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %last = scf.for %i = %c0 to %c3 step %c1 iter_args(%prev = %xf_init) -> (!ttl.transfer_handle<read>) {
      %xf_next = ttl.copy %slice, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
      ttl.wait %prev : !ttl.transfer_handle<read>
      scf.yield %xf_next : !ttl.transfer_handle<read>
    }
    ttl.wait %last : !ttl.transfer_handle<read>
    func.return
  }
}
