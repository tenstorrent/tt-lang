// RUN: ttlang-opt --convert-ttl-to-ttkernel --canonicalize --cse --split-input-file %s | FileCheck %s --check-prefix=TTKERNEL
// Summary: Lower a loopback DRAM copy (read → wait → write → wait in a loop)
// to TTKernel using global NOC barriers (TRID ops not yet available).

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>,
           memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// TTKERNEL-LABEL: func.func @loopback_dram_copy
// TTKERNEL-DAG: %[[C128:.*]] = arith.constant 128 : i32
// TTKERNEL-DAG: %[[C1:.*]] = arith.constant 1 : i32
// TTKERNEL-DAG: %[[C32:.*]] = arith.constant 32 : i32
// TTKERNEL-DAG: %[[C0:.*]] = arith.constant 0 : i32
// TTKERNEL: %[[ACC0_ARGS:.*]] = ttkernel.TensorAccessorArgs(%[[C32]], %[[C1]]) : (i32, i32) -> !ttkernel.TensorAccessorArgs
// TTKERNEL-NEXT: %[[ACC0:.*]] = ttkernel.TensorAccessor(%[[ACC0_ARGS]], %[[C0]], %[[C128]]) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// TTKERNEL-NEXT: %[[ACC1_ARGS:.*]] = ttkernel.TensorAccessorArgs(%[[C32]], %[[C1]]) : (i32, i32) -> !ttkernel.TensorAccessorArgs
// TTKERNEL-NEXT: %[[ACC1:.*]] = ttkernel.TensorAccessor(%[[ACC1_ARGS]], %[[C0]], %[[C128]]) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// TTKERNEL: scf.for
// TTKERNEL:   ttkernel.noc_async_read_tile(%{{.*}}, %[[ACC1]], %[[C0]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL:   ttkernel.noc_async_read_barrier() : () -> ()
// TTKERNEL:   ttkernel.noc_async_write_tile(%{{.*}}, %[[ACC0]], %[[C0]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL:   ttkernel.noc_async_write_barrier() : () -> ()

module {
  func.func @loopback_dram_copy(%src: tensor<32x32xf32, #layout>,
                                %dst: tensor<32x32xf32, #layout>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %c4 step %c1 {
      %xf_r = ttl.copy %src, %cb
        : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>)
          -> !ttl.transfer_handle<read>
      ttl.wait %xf_r : !ttl.transfer_handle<read>

      %xf_w = ttl.copy %cb, %dst
        : (!ttl.cb<[1, 1], f32, 2>, tensor<32x32xf32, #layout>)
          -> !ttl.transfer_handle<write>
      ttl.wait %xf_w : !ttl.transfer_handle<write>
    }

    func.return
  }
}
