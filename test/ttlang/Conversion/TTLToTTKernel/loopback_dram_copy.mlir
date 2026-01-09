// RUN: ttlang-opt --convert-ttl-to-ttkernel --canonicalize --cse --split-input-file %s | FileCheck %s --check-prefix=TTKERNEL
// Summary: Lower a loopback DRAM copy (read → wait → write → wait in a loop)
// to TTKernel using global NOC barriers (TRID ops not yet available).

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>,
           memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// TTKERNEL-LABEL: func.func @loopback_dram_copy
// Verify function-level tensor accessor materialization: accessors are built ONCE
// at function entry, then reused inside the loop for all copy operations.
// Function entry: src accessor (arg 0)
// TTKERNEL-DAG: %[[C0_IDX:.*]] = arith.constant 0 : index
// TTKERNEL-DAG: %[[C1_IDX:.*]] = arith.constant 1 : index
// TTKERNEL: %[[BANK_BASE_SRC:.*]] = ttkernel.get_common_arg_val(%[[C0_IDX]]) : (index) -> i32
// TTKERNEL-NEXT: %[[SRC_ARGS:.*]] = ttkernel.TensorAccessorArgs({{.*}})
// TTKERNEL-NEXT: %[[ACC_R:.*]] = ttkernel.TensorAccessor(%[[SRC_ARGS]], %[[BANK_BASE_SRC]], {{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// Function entry: dst accessor (arg 1)
// TTKERNEL-NEXT: %[[BANK_BASE_DST:.*]] = ttkernel.get_common_arg_val(%[[C1_IDX]]) : (index) -> i32
// TTKERNEL-NEXT: %[[DST_ARGS:.*]] = ttkernel.TensorAccessorArgs({{.*}})
// TTKERNEL-NEXT: %[[ACC_W:.*]] = ttkernel.TensorAccessor(%[[DST_ARGS]], %[[BANK_BASE_DST]], {{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// Loop: uses pre-materialized accessors
// TTKERNEL: scf.for
// TTKERNEL:   %[[CB_W_PTR:.*]] = ttkernel.get_write_ptr({{.*}}) : (!ttkernel.cb<2, f32>) -> i32
// TTKERNEL:   ttkernel.noc_async_read_tile({{.*}}, %[[ACC_R]], %[[CB_W_PTR]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL:   ttkernel.noc_async_read_barrier() : () -> ()
// TTKERNEL:   %[[CB_R_PTR:.*]] = ttkernel.get_read_ptr({{.*}}) : (!ttkernel.cb<2, f32>) -> i32
// TTKERNEL:   ttkernel.noc_async_write_tile({{.*}}, %[[ACC_W]], %[[CB_R_PTR]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL:   ttkernel.noc_async_write_barrier() : () -> ()

module {
  func.func @loopback_dram_copy(%src: tensor<32x32xf32, #layout>,
                                %dst: tensor<32x32xf32, #layout>)
      attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0, 1], ttl.kernel_thread = #ttkernel.thread<noc>} {
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
