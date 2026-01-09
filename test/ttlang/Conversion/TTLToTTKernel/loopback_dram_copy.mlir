// RUN: ttlang-opt --convert-ttl-to-ttkernel --canonicalize --cse --split-input-file %s | FileCheck %s --check-prefix=TTKERNEL
// Summary: Lower a loopback DRAM copy (read → wait → write → wait in a loop)
// to TTKernel using global NOC barriers (TRID ops not yet available).

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>,
           memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// TTKERNEL-LABEL: func.func @loopback_dram_copy
// Verify runtime args and CB pointers are used for both read and write operations.
// TTKERNEL: scf.for
// Read: runtime arg for src tensor, accessor, write ptr for CB
// TTKERNEL:   ttkernel.get_common_arg_val({{.*}}) : (index) -> i32
// TTKERNEL:   %[[ACC_R:.*]] = ttkernel.TensorAccessor({{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// TTKERNEL:   %[[CB_W_PTR:.*]] = ttkernel.get_write_ptr({{.*}}) : (!ttkernel.cb<2, f32>) -> i32
// TTKERNEL:   ttkernel.noc_async_read_tile({{.*}}, %[[ACC_R]], %[[CB_W_PTR]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL:   ttkernel.noc_async_read_barrier() : () -> ()
// Write: runtime arg for dst tensor, accessor, read ptr for CB
// TTKERNEL:   ttkernel.get_common_arg_val({{.*}}) : (index) -> i32
// TTKERNEL:   %[[ACC_W:.*]] = ttkernel.TensorAccessor({{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
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

    %src_slice = ttl.tensor_slice %src[%c0, %c0] : tensor<32x32xf32, #layout> -> tensor<32x32xf32, #layout>
    %dst_slice = ttl.tensor_slice %dst[%c0, %c0] : tensor<32x32xf32, #layout> -> tensor<32x32xf32, #layout>
    scf.for %i = %c0 to %c4 step %c1 {
      %xf_r = ttl.copy %src_slice, %cb
        : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>)
          -> !ttl.transfer_handle<read>
      ttl.wait %xf_r : !ttl.transfer_handle<read>

      %xf_w = ttl.copy %cb, %dst_slice
        : (!ttl.cb<[1, 1], f32, 2>, tensor<32x32xf32, #layout>)
          -> !ttl.transfer_handle<write>
      ttl.wait %xf_w : !ttl.transfer_handle<write>
    }

    func.return
  }
}
