// RUN: ttlang-opt --ttl-to-ttkernel-pipeline --canonicalize %s -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Test: Batched DMA operations
// Validates multiple async operations with proper barrier placement

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK: // dma_batched
// CHECK: void kernel_main() {
// CHECK-DAG:   int32_t [[ZERO:v[0-9]+]] = 0;
// CHECK-DAG:   int32_t [[ADDR:v[0-9]+]] = 4096;
// Tensor 0: get runtime arg, create accessor, get CB write ptr, async read
// CHECK:   int32_t [[RT_ARG0:v[0-9]+]] = get_common_arg_val<uint32_t>({{v[0-9]+}});
// CHECK:   auto [[ARGS0:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<2, 0>();
// CHECK:   TensorAccessor [[ACCESSOR0:v[0-9]+]] = TensorAccessor([[ARGS0]], [[RT_ARG0]], [[ADDR]]);
// CHECK:   int32_t [[CB_PTR0:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(0));
// CHECK:   noc_async_read_tile([[ZERO]], [[ACCESSOR0]], [[CB_PTR0]]);
// Tensor 1: get runtime arg, create accessor, get CB write ptr, async read
// CHECK:   int32_t [[RT_ARG1:v[0-9]+]] = get_common_arg_val<uint32_t>({{v[0-9]+}});
// CHECK:   auto [[ARGS1:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<3, 1>();
// CHECK:   TensorAccessor [[ACCESSOR1:v[0-9]+]] = TensorAccessor([[ARGS1]], [[RT_ARG1]], [[ADDR]]);
// CHECK:   int32_t [[CB_PTR1:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(1));
// CHECK:   noc_async_read_tile([[ZERO]], [[ACCESSOR1]], [[CB_PTR1]]);
// Consecutive barriers deduplicated to single barrier.
// CHECK:   noc_async_read_barrier();
// CHECK:   return;
// CHECK-NEXT: }
module {
  func.func @dma_batched(%t0: tensor<32x32xf32, #layout>, %t1: tensor<32x32xf32, #layout>) attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0, 1], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %slice0 = ttl.tensor_slice %t0[%c0, %c0] : tensor<32x32xf32, #layout> -> tensor<32x32xf32, #layout>
    %slice1 = ttl.tensor_slice %t1[%c0, %c0] : tensor<32x32xf32, #layout> -> tensor<32x32xf32, #layout>
    %xf0 = ttl.copy %slice0, %cb0 : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf1 = ttl.copy %slice1, %cb1 : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf0 : !ttl.transfer_handle<read>
    ttl.wait %xf1 : !ttl.transfer_handle<read>
    func.return
  }
}
