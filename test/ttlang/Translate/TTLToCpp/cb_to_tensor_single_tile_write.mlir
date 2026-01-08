// RUN: ttlang-opt --ttl-to-ttkernel-pipeline --canonicalize %s -o %t.ttkernel.mlir
// RUN: ttlang-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Test: Single DMA write operation (CB → tensor)
// Validates write barrier placement and ensures no read barrier

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK: // cb_to_tensor
// CHECK: void kernel_main() {
// CHECK-DAG:   int32_t [[ZERO:v[0-9]+]] = 0;
// CHECK-DAG:   int32_t [[ADDR:v[0-9]+]] = 4096;
// CHECK:   int32_t [[RT_ARG:v[0-9]+]] = get_common_arg_val<uint32_t>([[RT_ARG_IDX:v[0-9]+]]);
// CHECK:   auto [[ARGS:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<1, 0>();
// CHECK:   TensorAccessor [[ACCESSOR:v[0-9]+]] = TensorAccessor([[ARGS]], [[RT_ARG]], [[ADDR]]);
// CHECK:   int32_t [[CB_PTR:v[0-9]+]] = get_read_ptr(get_compile_time_arg_val(0));
// CHECK:   noc_async_write_tile([[ZERO]], [[ACCESSOR]], [[CB_PTR]]);
// CHECK:   noc_async_write_barrier();
// CHECK:   return;
// CHECK-NEXT: }
module {
  func.func @cb_to_tensor(%arg0: tensor<32x32xf32, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %slice = tensor.extract_slice %arg0[0, 0][32, 32][1, 1] : tensor<32x32xf32, #layout> to tensor<32x32xf32, #layout>
    %xf = ttl.copy %cb, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<32x32xf32, #layout>) -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
    func.return
  }
}
