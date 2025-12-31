// RUN: ttlang-opt --ttl-to-ttkernel-pipeline --canonicalize %s -o %t.ttkernel.mlir
// RUN: ttmlir-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Test: Single-tile DMA read operation (tensor → CB)
// Validates TTL→TTKernel→EmitC→C++ pipeline for basic single-tile DMA read

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK: // dma_single
// CHECK: void kernel_main() {
// CHECK-DAG:   int32_t [[ZERO:v[0-9]+]] = 0;
// CHECK-DAG:   int32_t [[ADDR:v[0-9]+]] = 128;
// CHECK:   int32_t [[RT_ARG:v[0-9]+]] = get_common_arg_val<uint32_t>([[RT_ARG_IDX:v[0-9]+]]);
// Placeholder value 42 is a temporary hack, see issue #168
// CHECK:   auto [[ARGS:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<42, 0>();
// CHECK:   TensorAccessor [[ACCESSOR:v[0-9]+]] = TensorAccessor([[ARGS]], [[RT_ARG]], [[ADDR]]);
// CHECK:   int32_t [[CB_PTR:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(0));
// CHECK:   noc_async_read_tile([[ZERO]], [[ACCESSOR]], [[CB_PTR]]);
// CHECK:   noc_async_read_barrier();
// CHECK:   return;
// CHECK-NEXT: }
module {
  func.func @dma_single(%arg0: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %arg0, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}
