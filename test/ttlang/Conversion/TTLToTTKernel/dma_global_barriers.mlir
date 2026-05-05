// RUN: ttlang-opt --convert-ttl-to-ttkernel --canonicalize -cse --split-input-file %s | FileCheck %s --check-prefix=GLOBAL
// Summary: Verify the default (non-TRID) code path emits global NOC barriers.
// Companion to dma_single_core.mlir which tests use-trid-barriers=1.

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Single-tile read: default mode emits noc_async_read_barrier (no TRID ops).
// GLOBAL-LABEL: func.func @global_single_tile_read
// GLOBAL: ttkernel.noc_async_read_tile({{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// GLOBAL: ttkernel.noc_async_read_barrier() : () -> ()
// GLOBAL-NOT: ttkernel.noc_async_read_set_trid
// GLOBAL-NOT: ttkernel.noc_async_read_barrier_with_trid
// GLOBAL-NOT: ttkernel.noc_async_write_barrier
module {
  func.func @global_single_tile_read(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %slice = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf = ttl.copy %slice, %cb : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Single-tile write: default mode emits noc_async_write_barrier (no TRID ops).
// GLOBAL-LABEL: func.func @global_single_tile_write
// GLOBAL: ttkernel.noc_async_write_tile({{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// GLOBAL: ttkernel.noc_async_write_barrier() : () -> ()
// GLOBAL-NOT: ttkernel.noc_async_write_set_trid
// GLOBAL-NOT: ttkernel.noc_async_write_barrier_with_trid
// GLOBAL-NOT: ttkernel.noc_async_read_barrier
module {
  func.func @global_single_tile_write(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %slice = ttl.tensor_slice %arg0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf = ttl.copy %cb, %slice : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Batched reads: consecutive global barriers are deduplicated to a single barrier.
// GLOBAL-LABEL: func.func @global_batched_reads
// GLOBAL: ttkernel.noc_async_read_tile({{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// GLOBAL: ttkernel.noc_async_read_tile({{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// GLOBAL: ttkernel.noc_async_read_barrier() : () -> ()
// GLOBAL-NOT: ttkernel.noc_async_read_barrier
// GLOBAL-NOT: ttkernel.noc_async_read_set_trid
// GLOBAL-NOT: ttkernel.noc_async_read_barrier_with_trid
module {
  func.func @global_batched_reads(%t0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, %t1: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0, 1], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %slice0 = ttl.tensor_slice %t0[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %slice1 = ttl.tensor_slice %t1[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %xf0 = ttl.copy %slice0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf1 = ttl.copy %slice1, %cb1 : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf0 : !ttl.transfer_handle<read>
    ttl.wait %xf1 : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Loopback copy: read then write in a loop uses global barriers for both.
// GLOBAL-LABEL: func.func @global_loopback
// GLOBAL: scf.for
// GLOBAL:   ttkernel.noc_async_read_tile({{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// GLOBAL:   ttkernel.noc_async_read_barrier() : () -> ()
// GLOBAL:   ttkernel.noc_async_write_tile({{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// GLOBAL:   ttkernel.noc_async_write_barrier() : () -> ()
// GLOBAL-NOT: noc_async_read_set_trid
// GLOBAL-NOT: noc_async_write_set_trid
// GLOBAL-NOT: barrier_with_trid
module {
  func.func @global_loopback(%src: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>,
                             %dst: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
      attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0, 1], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index

    %src_slice = ttl.tensor_slice %src[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %dst_slice = ttl.tensor_slice %dst[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    scf.for %i = %c0 to %c4 step %c1 {
      %xf_r = ttl.copy %src_slice, %cb
        : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>, !ttl.cb<[1, 1], f32, 2>)
          -> !ttl.transfer_handle<read>
      ttl.wait %xf_r : !ttl.transfer_handle<read>

      %xf_w = ttl.copy %cb, %dst_slice
        : (!ttl.cb<[1, 1], f32, 2>, tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
          -> !ttl.transfer_handle<write>
      ttl.wait %xf_w : !ttl.transfer_handle<write>
    }

    func.return
  }
}
