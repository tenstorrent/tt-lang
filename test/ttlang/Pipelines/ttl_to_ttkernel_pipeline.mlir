// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline | FileCheck %s
// Summary: Ensures the end-to-end TTL→TTKernel pipeline (compute lowering,
// bufferization, memref cleanup, TTKernel conversion) produces TTKernel NOC
// ops from high level TTL DMA IR.

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: func.func @dma_single_tile_single_copy
// CHECK-SAME: () attributes {ttkernel.arg_spec =
// CHECK-NEXT:   %[[C128:.*]] = arith.constant 128 : i32
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : i32
// CHECK-NEXT:   %[[C32:.*]] = arith.constant 32 : i32
// CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NEXT:   %[[ARGS:.*]] = ttkernel.TensorAccessorArgs(%[[C32]], %[[C1]]) : (i32, i32) -> !ttkernel.TensorAccessorArgs
// CHECK-NEXT:   %[[ACC:.*]] = ttkernel.TensorAccessor(%[[ARGS]], %[[C0]], %[[C128]]) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// CHECK-NEXT:   ttkernel.noc_async_read_tile(%[[C0]], %[[ACC]], %[[C0]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// CHECK-NEXT:   ttkernel.noc_async_read_barrier() : () -> ()
// CHECK-NEXT:   return
// CHECK-NOT: ttl.copy
module {
  func.func @dma_single_tile_single_copy(%arg0: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %arg0, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

