// RUN: ttlang-opt --convert-ttl-to-ttkernel --canonicalize -cse --split-input-file %s | FileCheck %s

#layout = #ttl.layout<shape = [5, 7168], element_type = bf16,
                      buffer = dram, grid = [1, 1], memory = interleaved>

// CHECK-LABEL: func.func @copy_bf16_tensor_page
// CHECK-DAG: %[[PAGE:.*]] = arith.constant 4 : i32
// CHECK-DAG: %[[SIZE:.*]] = arith.constant 14336 : i32
// CHECK-DAG: %[[NOC:.*]] = arith.constant 0 : i8
// CHECK: %[[DFB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<448, !ttcore.tile<1x32, bf16>>
// CHECK: %[[BASE:.*]] = ttkernel.get_common_arg_val({{.*}}) : (index) -> i32
// CHECK: %[[ARGS:.*]] = ttkernel.TensorAccessorArgs({{.*}})
// CHECK: %[[ACCESSOR:.*]] = ttkernel.TensorAccessor(%[[ARGS]], %[[BASE]]) : (!ttkernel.TensorAccessorArgs, i32) -> !ttkernel.TensorAccessor
// CHECK: %[[DESTINATION:.*]] = ttkernel.get_write_ptr(%[[DFB]])
// CHECK: ttkernel.noc_async_read_tile(%[[PAGE]], %[[ACCESSOR]], %[[DESTINATION]], %[[NOC]], size %[[SIZE]])
// CHECK: ttkernel.noc_async_read_barrier(%[[NOC]])
// CHECK-NOT: ttkernel.opaque_call
module {
  func.func @copy_bf16_tensor_page(
      %source: tensor<5x7168xbf16, #layout>)
      attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                  ttl.kernel_thread = #ttkernel.thread<noc>, ttl.noc_index = 0 : i64} {
    %page = arith.constant 4 : index
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 224], !ttcore.tile<1x32, bf16>, 2>
    %xf = ttl.copy_tensor_page %source[%page], %dfb
        : tensor<5x7168xbf16, #layout>,
          !ttl.cb<[1, 224], !ttcore.tile<1x32, bf16>, 2>
        -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    return
  }
}

// -----

#layout = #ttl.layout<shape = [5, 64], element_type = f32,
                      buffer = l1, grid = [1, 1], memory = interleaved>

// CHECK-LABEL: func.func @copy_fp32_tensor_page
// CHECK-DAG: %[[SIZE:.*]] = arith.constant 256 : i32
// CHECK: %[[ACCESSOR:.*]] = ttkernel.TensorAccessor({{.*}}) : (!ttkernel.TensorAccessorArgs, i32) -> !ttkernel.TensorAccessor
// CHECK: ttkernel.noc_async_read_tile({{.*}}, %[[ACCESSOR]], {{.*}}, {{.*}}, size %[[SIZE]])
// CHECK: ttkernel.noc_async_read_barrier
module {
  func.func @copy_fp32_tensor_page(
      %source: tensor<5x64xf32, #layout>, %page: index)
      attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                  ttl.kernel_thread = #ttkernel.thread<noc>, ttl.noc_index = 1 : i64} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 2], !ttcore.tile<1x32, f32>, 2>
    %xf = ttl.copy_tensor_page %source[%page], %dfb
        : tensor<5x64xf32, #layout>,
          !ttl.cb<[1, 2], !ttcore.tile<1x32, f32>, 2>
        -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    return
  }
}
