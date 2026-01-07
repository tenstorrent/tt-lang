// RUN: ttlang-opt --convert-ttl-to-ttkernel --canonicalize --cse --split-input-file %s | FileCheck %s --check-prefix=TTKERNEL
// Summary: Lower a loopback DRAM copy (read → wait → write → wait in a loop)
// to TTKernel using global NOC barriers (TRID ops not yet available).

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>,
           memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// TTKERNEL-LABEL: func.func @loopback_dram_copy
// Verify runtime args and CB pointers are used for both read and write operations.
// Tensor accessors are now created at function entry with simple index offsets, not inside loops.
// TTKERNEL-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// TTKERNEL-DAG: %[[C1_I32:.*]] = arith.constant 1 : i32
// TTKERNEL-DAG: %[[C2_I32:.*]] = arith.constant 2 : i32
// TTKERNEL-DAG: %[[PAGE_SIZE:.*]] = arith.constant {{[0-9]+}} : i32
// TTKERNEL-DAG: %[[C0:.*]] = arith.constant 0 : index
// TTKERNEL-DAG: %[[C1:.*]] = arith.constant 1 : index
// First tensor accessor (src) at function entry - CTA starts at 1 (after 1 bind_cb).
// TTKERNEL: %[[BANK0:.*]] = ttkernel.get_common_arg_val(%[[C0]]) : (index) -> i32
// TTKERNEL: %[[ARGS0:.*]] = ttkernel.TensorAccessorArgs(%[[C1_I32]], %[[C0_I32]])
// TTKERNEL: %[[ACC_R:.*]] = ttkernel.TensorAccessor(%[[ARGS0]], %[[BANK0]], %[[PAGE_SIZE]]) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// Second tensor accessor (dst) uses simple index offset: CTA = 1+1=2, CRTA = 0+1=1.
// TTKERNEL: %[[BANK1:.*]] = ttkernel.get_common_arg_val(%[[C1]]) : (index) -> i32
// TTKERNEL: %[[ARGS1:.*]] = ttkernel.TensorAccessorArgs(%[[C2_I32]], %[[C1_I32]])
// TTKERNEL: %[[ACC_W:.*]] = ttkernel.TensorAccessor(%[[ARGS1]], %[[BANK1]], %[[PAGE_SIZE]]) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// Loop uses pre-materialized accessors.
// TTKERNEL: scf.for
// Read: write ptr for CB, then tile read using pre-materialized accessor.
// TTKERNEL:   %[[CB_W_PTR:.*]] = ttkernel.get_write_ptr({{.*}}) : (!ttkernel.cb<2, f32>) -> i32
// TTKERNEL:   ttkernel.noc_async_read_tile(%[[C0_I32]], %[[ACC_R]], %[[CB_W_PTR]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL:   ttkernel.noc_async_read_barrier() : () -> ()
// Write: read ptr for CB, then tile write using pre-materialized accessor.
// TTKERNEL:   %[[CB_R_PTR:.*]] = ttkernel.get_read_ptr({{.*}}) : (!ttkernel.cb<2, f32>) -> i32
// TTKERNEL:   ttkernel.noc_async_write_tile(%[[C0_I32]], %[[ACC_W]], %[[CB_R_PTR]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL:   ttkernel.noc_async_write_barrier() : () -> ()

module {
  func.func @loopback_dram_copy(%src: tensor<32x32xf32, #layout>,
                                %dst: tensor<32x32xf32, #layout>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.base_cta_index = 1 : i32} {
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
