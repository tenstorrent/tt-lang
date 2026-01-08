// RUN: ttlang-opt --convert-ttl-to-ttkernel --canonicalize -cse --split-input-file %s | FileCheck %s --check-prefix=TTKERNEL
// Summary: MVP DMA lowering tests for tensor<->CB copies (no pipes).

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// TTKERNEL-LABEL: func.func @dma_single_tile_single_copy
// TTKERNEL-DAG: %[[C0_IDX:.*]] = arith.constant 0 : index
// TTKERNEL: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, f32>
// TTKERNEL: %[[BANK_BASE:.*]] = ttkernel.get_common_arg_val(%[[C0_IDX]]) : (index) -> i32
// TTKERNEL: %[[SRC_ARGS:.*]] = ttkernel.TensorAccessorArgs({{.*}})
// TTKERNEL: %[[SRC_ACC:.*]] = ttkernel.TensorAccessor(%[[SRC_ARGS]], %[[BANK_BASE]], {{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// TTKERNEL: %[[CB_PTR:.*]] = ttkernel.get_write_ptr(%[[CB]]) : (!ttkernel.cb<2, f32>) -> i32
// TTKERNEL: ttkernel.noc_async_read_tile({{.*}}, %[[SRC_ACC]], %[[CB_PTR]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL: ttkernel.noc_async_read_barrier() : () -> ()
// TTKERNEL-NOT: ttkernel.noc_async_write_barrier
module {
  func.func @dma_single_tile_single_copy(%arg0: tensor<32x32xf32, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %arg0, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// TTKERNEL-LABEL: func.func @cb_to_tensor
// TTKERNEL-DAG: %[[C0_IDX:.*]] = arith.constant 0 : index
// TTKERNEL: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, f32>
// TTKERNEL: %[[BANK_BASE:.*]] = ttkernel.get_common_arg_val(%[[C0_IDX]]) : (index) -> i32
// TTKERNEL: %[[DST_ARGS:.*]] = ttkernel.TensorAccessorArgs({{.*}})
// TTKERNEL: %[[DST_ACC:.*]] = ttkernel.TensorAccessor(%[[DST_ARGS]], %[[BANK_BASE]], {{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// TTKERNEL: %[[CB_PTR:.*]] = ttkernel.get_read_ptr(%[[CB]]) : (!ttkernel.cb<2, f32>) -> i32
// TTKERNEL: ttkernel.noc_async_write_tile({{.*}}, %[[DST_ACC]], %[[CB_PTR]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL: ttkernel.noc_async_write_barrier() : () -> ()
// TTKERNEL-NOT: ttkernel.noc_async_read_barrier
module {
  func.func @cb_to_tensor(%arg0: tensor<32x32xf32, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %cb, %arg0 : (!ttl.cb<[1, 1], f32, 2>, tensor<32x32xf32, #layout>) -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Batched transfer pattern: issue multiple transfers, then wait on all of them.
// Mirrors TT-Metal kernels that batch NOC async operations for throughput.
// Each tensor arg maps to a runtime arg at its index.
// TTKERNEL-LABEL: func.func @dma_batched
// First tensor: runtime arg, accessor, CB write ptr, tile read.
// TTKERNEL: ttkernel.get_common_arg_val({{.*}}) : (index) -> i32
// TTKERNEL: ttkernel.TensorAccessor({{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// TTKERNEL: ttkernel.get_write_ptr({{.*}}) : (!ttkernel.cb<2, f32>) -> i32
// TTKERNEL: ttkernel.noc_async_read_tile({{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// Second tensor: runtime arg, accessor, CB write ptr, tile read.
// TTKERNEL: ttkernel.get_common_arg_val({{.*}}) : (index) -> i32
// TTKERNEL: ttkernel.TensorAccessor({{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// TTKERNEL: ttkernel.get_write_ptr({{.*}}) : (!ttkernel.cb<2, f32>) -> i32
// TTKERNEL: ttkernel.noc_async_read_tile({{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// Consecutive barriers are deduplicated to a single barrier.
// TTKERNEL: ttkernel.noc_async_read_barrier() : () -> ()
// TTKERNEL-NOT: ttkernel.noc_async_read_barrier
// TTKERNEL-NOT: ttkernel.noc_async_write_barrier
module {
  func.func @dma_batched(%t0: tensor<32x32xf32, #layout>, %t1: tensor<32x32xf32, #layout>) attributes {ttl.base_cta_index = 2 : i32, ttl.crta_indices = [0, 1], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf0 = ttl.copy %t0, %cb0 : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf1 = ttl.copy %t1, %cb1 : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf0 : !ttl.transfer_handle<read>
    ttl.wait %xf1 : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Pipelined loop pattern: wait on the previous transfer while issuing the next.
// This approximates "copies in one loop, waits in another" by separating the wait
// from the copy in time while staying in SSA form.

// TTKERNEL-LABEL: func.func @dma_pipelined_loop
// TTKERNEL: ttkernel.noc_async_read_tile({{.*}}, {{.*}}, {{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL: scf.for {{.*}} {
// TTKERNEL:   ttkernel.noc_async_read_tile({{.*}}, {{.*}}, {{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL:   ttkernel.noc_async_read_barrier() : () -> ()
// TTKERNEL: }
// TTKERNEL: ttkernel.noc_async_read_barrier() : () -> ()
// TTKERNEL-NOT: ttkernel.noc_async_write_barrier
module {
  func.func @dma_pipelined_loop(%t: tensor<32x32xf32, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index

    %xf_init = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %last = scf.for %i = %c0 to %c3 step %c1 iter_args(%prev = %xf_init) -> (!ttl.transfer_handle<read>) {
      %xf_next = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
      ttl.wait %prev : !ttl.transfer_handle<read>
      scf.yield %xf_next : !ttl.transfer_handle<read>
    }
    ttl.wait %last : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Two-phase pattern: issue all copies in one loop, then wait on all handles in
// a second loop. This mirrors TT-Metal kernels that batch NOC async ops and then
// block on a barrier after issuing the batch.
//
// TTKERNEL-LABEL: func.func @dma_single_tile_two_phase_loops
// TTKERNEL: %[[HANDLES0:.*]] = tensor.empty() : tensor<4x!ttl.transfer_handle<read>>
// TTKERNEL: %[[CAST:.*]] = tensor.cast %[[HANDLES0]] : tensor<4x!ttl.transfer_handle<read>> to tensor<?x!ttl.transfer_handle<read>>
// TTKERNEL: %[[HANDLES:.*]] = scf.for {{.*}} iter_args(%[[H:.*]] = %[[CAST]]) -> (tensor<?x!ttl.transfer_handle<read>>) {
// TTKERNEL:   ttkernel.noc_async_read_tile({{.*}}, {{.*}}, {{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL:   %[[XF:.*]] = builtin.unrealized_conversion_cast {{.*}} : i32 to !ttl.transfer_handle<read>
// TTKERNEL:   %[[INS:.*]] = tensor.insert %[[XF]] into %[[H]]{{\[}}{{.*}}{{\]}} : tensor<?x!ttl.transfer_handle<read>>
// TTKERNEL:   scf.yield %[[INS]] : tensor<?x!ttl.transfer_handle<read>>
// TTKERNEL: }
// TTKERNEL: scf.for {{.*}} {
// TTKERNEL:   ttkernel.noc_async_read_barrier() : () -> ()
// TTKERNEL: }
// TTKERNEL-NOT: ttkernel.noc_async_write_barrier
module {
  func.func @dma_single_tile_two_phase_loops(%t: tensor<32x32xf32, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %handles0 = tensor.empty(%c4) : tensor<?x!ttl.transfer_handle<read>>

    %handles = scf.for %i = %c0 to %c4 step %c1 iter_args(%h = %handles0) -> tensor<?x!ttl.transfer_handle<read>> {
      %xf = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
      %h2 = tensor.insert %xf into %h[%i] : tensor<?x!ttl.transfer_handle<read>>
      scf.yield %h2 : tensor<?x!ttl.transfer_handle<read>>
    }

    scf.for %i = %c0 to %c4 step %c1 {
      %xf = tensor.extract %handles[%i] : tensor<?x!ttl.transfer_handle<read>>
      ttl.wait %xf : !ttl.transfer_handle<read>
    }
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Corner case: waiting twice on the same transfer handle is allowed, but
// consecutive barriers are deduplicated to a single barrier.
//
// TTKERNEL-LABEL: func.func @dma_single_tile_double_wait
// TTKERNEL:      ttkernel.noc_async_read_barrier() : () -> ()
// TTKERNEL-NOT: ttkernel.noc_async_read_barrier
// TTKERNEL-NOT: ttkernel.noc_async_write_barrier
module {
  func.func @dma_single_tile_double_wait(%t: tensor<32x32xf32, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Corner case: one-element handle batching via tensor.insert, then waiting
// outside of a loop.
//
// TTKERNEL-LABEL: func.func @dma_single_tile_single_element_container
// TTKERNEL: ttkernel.noc_async_read_tile({{.*}}, {{.*}}, {{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL: ttkernel.noc_async_read_barrier() : () -> ()
// TTKERNEL-NOT: ttkernel.noc_async_write_barrier
// TTKERNEL: return
module {
  func.func @dma_single_tile_single_element_container(%t: tensor<32x32xf32, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c1 = arith.constant 1 : index
    %handles0 = tensor.empty(%c1) : tensor<?x!ttl.transfer_handle<read>>

    %handles = scf.for %i = %c0 to %c1 step %c1 iter_args(%h = %handles0) -> tensor<?x!ttl.transfer_handle<read>> {
      %xf = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
      %h2 = tensor.insert %xf into %h[%i] : tensor<?x!ttl.transfer_handle<read>>
      scf.yield %h2 : tensor<?x!ttl.transfer_handle<read>>
    }

    %xf0 = tensor.extract %handles[%c0] : tensor<?x!ttl.transfer_handle<read>>
    ttl.wait %xf0 : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Multi-tile read should emit nested scf.for over tile grid with correct offset computation.
// Tensor: 64x64xf32 (2x2 tiles), CB: [1,1] (single tile)
// Generated tile loops: for tile_y in 0..2, for tile_x in 0..2
// Tile offset = tile_y * tiles_x + tile_x (row-major ordering)
// TTKERNEL-LABEL: func.func @dma_multi_tile_read
// TTKERNEL-DAG: %[[TILE_LB:.*]] = arith.constant 0 : index
// TTKERNEL-DAG: %[[TILE_STEP:.*]] = arith.constant 1 : index
// TTKERNEL-DAG: %[[TILES_BOUND:.*]] = arith.constant 2 : index
// TTKERNEL: ttkernel.get_common_arg_val({{.*}}) : (index) -> i32
// TTKERNEL: %[[ARGS:.*]] = ttkernel.TensorAccessorArgs({{.*}})
// TTKERNEL: %[[ACC:.*]] = ttkernel.TensorAccessor(%[[ARGS]], {{.*}}, {{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// TTKERNEL: %[[CB_PTR:.*]] = ttkernel.get_write_ptr({{.*}}) : (!ttkernel.cb<{{.*}}>) -> i32
// TTKERNEL: scf.for %[[TILE_Y:.*]] = %[[TILE_LB]] to %[[TILES_BOUND]] step %[[TILE_STEP]]
// TTKERNEL:   scf.for %[[TILE_X:.*]] = %[[TILE_LB]] to %[[TILES_BOUND]] step %[[TILE_STEP]]
// TTKERNEL:     %[[TILE_OFFSET_Y:.*]] = arith.muli %[[TILE_Y]], %[[TILES_BOUND]] : index
// TTKERNEL:     %[[TILE_OFFSET_X:.*]] = arith.addi %[[TILE_OFFSET_Y]], %[[TILE_X]] : index
// TTKERNEL:     %[[TILE_OFFSET_I32:.*]] = arith.index_cast %[[TILE_OFFSET_X]] : index to i32
// TTKERNEL:     ttkernel.noc_async_read_tile(%[[TILE_OFFSET_I32]], %[[ACC]], %[[CB_PTR]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL: ttkernel.noc_async_read_barrier() : () -> ()
// TTKERNEL-NOT: ttkernel.noc_async_write_barrier
module {
  func.func @dma_multi_tile_read(%arg0: tensor<64x64xf32, #layout_tile>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %arg0, %cb : (tensor<64x64xf32, #layout_tile>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Multi-tile write should emit nested scf.for over tile grid with correct offset computation.
// Tensor: 64x64xf32 (2x2 tiles), CB: [1,1] (single tile)
// Generated tile loops: for tile_y in 0..2, for tile_x in 0..2
// Tile offset = tile_y * tiles_x + tile_x (row-major ordering)
// TTKERNEL-LABEL: func.func @dma_multi_tile_write
// TTKERNEL-DAG: %[[TILE_LB:.*]] = arith.constant 0 : index
// TTKERNEL-DAG: %[[TILE_STEP:.*]] = arith.constant 1 : index
// TTKERNEL-DAG: %[[TILES_BOUND:.*]] = arith.constant 2 : index
// TTKERNEL: ttkernel.get_common_arg_val({{.*}}) : (index) -> i32
// TTKERNEL: %[[ARGS:.*]] = ttkernel.TensorAccessorArgs({{.*}})
// TTKERNEL: %[[ACC:.*]] = ttkernel.TensorAccessor(%[[ARGS]], {{.*}}, {{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// TTKERNEL: %[[CB_PTR:.*]] = ttkernel.get_read_ptr({{.*}}) : (!ttkernel.cb<{{.*}}>) -> i32
// TTKERNEL: scf.for %[[TILE_Y:.*]] = %[[TILE_LB]] to %[[TILES_BOUND]] step %[[TILE_STEP]]
// TTKERNEL:   scf.for %[[TILE_X:.*]] = %[[TILE_LB]] to %[[TILES_BOUND]] step %[[TILE_STEP]]
// TTKERNEL:     %[[TILE_OFFSET_Y:.*]] = arith.muli %[[TILE_Y]], %[[TILES_BOUND]] : index
// TTKERNEL:     %[[TILE_OFFSET_X:.*]] = arith.addi %[[TILE_OFFSET_Y]], %[[TILE_X]] : index
// TTKERNEL:     %[[TILE_OFFSET_I32:.*]] = arith.index_cast %[[TILE_OFFSET_X]] : index to i32
// TTKERNEL:     ttkernel.noc_async_write_tile(%[[TILE_OFFSET_I32]], %[[ACC]], %[[CB_PTR]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL: ttkernel.noc_async_write_barrier() : () -> ()
// TTKERNEL-NOT: ttkernel.noc_async_read_barrier
module {
  func.func @dma_multi_tile_write(%arg0: tensor<64x64xf32, #layout_tile>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %cb, %arg0 : (!ttl.cb<[1, 1], f32, 2>, tensor<64x64xf32, #layout_tile>) -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Multi-tile read with larger CB shape still loops over tile grid with correct offset computation.
// Tensor: 64x64xf32 (2x2 tiles), CB: [2,1] (2x1 tiles)
// CB shape does NOT affect tile loop bounds - loops still iterate over tensor tile grid (2x2).
// Generated tile loops: for tile_y in 0..2, for tile_x in 0..2
// Tile offset = tile_y * tiles_x + tile_x (row-major ordering)
// TTKERNEL-LABEL: func.func @dma_multi_tile_read_cb_shape
// TTKERNEL-DAG: %[[TILE_LB:.*]] = arith.constant 0 : index
// TTKERNEL-DAG: %[[TILE_STEP:.*]] = arith.constant 1 : index
// TTKERNEL-DAG: %[[TILES_BOUND:.*]] = arith.constant 2 : index
// TTKERNEL: ttkernel.get_common_arg_val({{.*}}) : (index) -> i32
// TTKERNEL: %[[ARGS:.*]] = ttkernel.TensorAccessorArgs({{.*}})
// TTKERNEL: %[[ACC:.*]] = ttkernel.TensorAccessor(%[[ARGS]], {{.*}}, {{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// TTKERNEL: %[[CB_PTR:.*]] = ttkernel.get_write_ptr({{.*}}) : (!ttkernel.cb<{{.*}}>) -> i32
// TTKERNEL: scf.for %[[TILE_Y:.*]] = %[[TILE_LB]] to %[[TILES_BOUND]] step %[[TILE_STEP]]
// TTKERNEL:   scf.for %[[TILE_X:.*]] = %[[TILE_LB]] to %[[TILES_BOUND]] step %[[TILE_STEP]]
// TTKERNEL:     %[[TILE_OFFSET_Y:.*]] = arith.muli %[[TILE_Y]], %[[TILES_BOUND]] : index
// TTKERNEL:     %[[TILE_OFFSET_X:.*]] = arith.addi %[[TILE_OFFSET_Y]], %[[TILE_X]] : index
// TTKERNEL:     %[[TILE_OFFSET_I32:.*]] = arith.index_cast %[[TILE_OFFSET_X]] : index to i32
// TTKERNEL:     ttkernel.noc_async_read_tile(%[[TILE_OFFSET_I32]], %[[ACC]], %[[CB_PTR]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL: ttkernel.noc_async_read_barrier() : () -> ()
// TTKERNEL-NOT: ttkernel.noc_async_write_barrier
module {
  func.func @dma_multi_tile_read_cb_shape(%arg0: tensor<64x64xf32, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 1], f32, 2>
    %xf = ttl.copy %arg0, %cb : (tensor<64x64xf32, #layout>, !ttl.cb<[2, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Rectangular multi-tile write to exercise non-square tile grids (96x64 = 3x2 tiles) with correct offset computation.
// Tensor: 96x64xf32 (3x2 tiles - 3 rows, 2 columns), CB: [1,1] (single tile)
// Generated tile loops: for tile_y in 0..3, for tile_x in 0..2
// Tile offset = tile_y * tiles_x + tile_x (row-major ordering)
// Examples: (0,0)→0, (0,1)→1, (1,0)→2, (1,1)→3, (2,0)→4, (2,1)→5
// TTKERNEL-LABEL: func.func @dma_multi_tile_write_rect
// TTKERNEL-DAG: %[[TILE_LB:.*]] = arith.constant 0 : index
// TTKERNEL-DAG: %[[TILE_STEP:.*]] = arith.constant 1 : index
// TTKERNEL-DAG: %[[TILES_Y_BOUND:.*]] = arith.constant 3 : index
// TTKERNEL-DAG: %[[TILES_X_BOUND:.*]] = arith.constant 2 : index
// TTKERNEL: ttkernel.get_common_arg_val({{.*}}) : (index) -> i32
// TTKERNEL: %[[ARGS:.*]] = ttkernel.TensorAccessorArgs({{.*}})
// TTKERNEL: %[[ACC:.*]] = ttkernel.TensorAccessor(%[[ARGS]], {{.*}}, {{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// TTKERNEL: %[[CB_PTR:.*]] = ttkernel.get_read_ptr({{.*}}) : (!ttkernel.cb<{{.*}}>) -> i32
// TTKERNEL: scf.for %[[TILE_Y:.*]] = %[[TILE_LB]] to %[[TILES_Y_BOUND]] step %[[TILE_STEP]]
// TTKERNEL:   scf.for %[[TILE_X:.*]] = %[[TILE_LB]] to %[[TILES_X_BOUND]] step %[[TILE_STEP]]
// TTKERNEL:     %[[TILE_OFFSET_Y:.*]] = arith.muli %[[TILE_Y]], %[[TILES_X_BOUND]] : index
// TTKERNEL:     %[[TILE_OFFSET_X:.*]] = arith.addi %[[TILE_OFFSET_Y]], %[[TILE_X]] : index
// TTKERNEL:     %[[TILE_OFFSET_I32:.*]] = arith.index_cast %[[TILE_OFFSET_X]] : index to i32
// TTKERNEL:     ttkernel.noc_async_write_tile(%[[TILE_OFFSET_I32]], %[[ACC]], %[[CB_PTR]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL: ttkernel.noc_async_write_barrier() : () -> ()
// TTKERNEL-NOT: ttkernel.noc_async_read_barrier
module {
  func.func @dma_multi_tile_write_rect(%arg0: tensor<96x64xf32, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %cb, %arg0 : (!ttl.cb<[1, 1], f32, 2>, tensor<96x64xf32, #layout>) -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
    func.return
  }
}
