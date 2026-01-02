// RUN: ttlang-opt --ttl-to-ttkernel-pipeline --canonicalize %s -o %t.ttkernel.mlir
// RUN: ttmlir-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttmlir-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Test: Binary DMA read with correct TensorAccessorArgs pattern
// Demonstrates the proper way to create accessors for multiple tensors using
// template parameter indexing instead of hardcoded strides

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: void kernel_main() {

// First tensor accessor - uses TensorAccessorArgs<num_cbs, 0>()
// num_cbs = 2 in this test (cb0 and cb1)
// CHECK:   int32_t [[RT_ARG_A:v[0-9]+]] = get_common_arg_val<uint32_t>([[IDX_A:v[0-9]+]]);
// CHECK-NEXT:   auto [[ARGS_A:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<2, 0>();
// CHECK-NEXT:   TensorAccessor [[ACC_A:v[0-9]+]] = TensorAccessor([[ARGS_A]], [[RT_ARG_A]], [[TILE_SIZE:v[0-9]+]]);

// Second tensor accessor - chains from first using .next_compile_time_args_offset()
// CHECK-NEXT:   int32_t [[RT_ARG_B:v[0-9]+]] = get_common_arg_val<uint32_t>([[IDX_B:v[0-9]+]]);
// CHECK-NEXT:   auto [[ARGS_B:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<[[ARGS_A]].next_compile_time_args_offset(), [[ARGS_A]].next_common_runtime_args_offset()>();
// CHECK-NEXT:   TensorAccessor [[ACC_B:v[0-9]+]] = TensorAccessor([[ARGS_B]], [[RT_ARG_B]], [[TILE_SIZE]]);

// Read first tensor
// CHECK:   int32_t [[CB0_PTR:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(0));
// CHECK-NEXT:   noc_async_read_tile([[TILE_IDX:v[0-9]+]], [[ACC_A]], [[CB0_PTR]]);

// Read second tensor
// CHECK:   int32_t [[CB1_PTR:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(1));
// CHECK-NEXT:   noc_async_read_tile([[TILE_IDX]], [[ACC_B]], [[CB1_PTR]]);

// Barrier and return
// CHECK-NEXT:   noc_async_read_barrier();
// CHECK-NEXT:   return;
// CHECK-NEXT: }

module {
  func.func @dma_binary_read(%arg0: tensor<32x32xf32, #layout>, %arg1: tensor<32x32xf32, #layout>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.base_cta_index = 2 : i32} {
    %c0 = arith.constant 0 : index
    %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>

    // Read first tensor
    %xf0 = ttl.copy %arg0, %cb0 : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>

    // Read second tensor
    %xf1 = ttl.copy %arg1, %cb1 : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>

    ttl.wait %xf0 : !ttl.transfer_handle<read>
    ttl.wait %xf1 : !ttl.transfer_handle<read>
    func.return
  }
}
