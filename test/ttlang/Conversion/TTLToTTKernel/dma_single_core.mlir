// RUN: ttlang-opt --convert-ttl-to-ttkernel --canonicalize -cse --split-input-file %s | FileCheck %s
// Summary: MVP DMA lowering tests for tensor<->CB copies (no pipes).
// Tile loop fusion happens during conversion via pre-conversion grouping.

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: func.func @dma_single_tile_single_copy
// CHECK-DAG: %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, f32>
// CHECK: %[[BANK_BASE:.*]] = ttkernel.get_common_arg_val(%[[C0_IDX]]) : (index) -> i32
// CHECK: %[[SRC_ARGS:.*]] = ttkernel.TensorAccessorArgs({{.*}}) : (i32, i32) -> !ttkernel.TensorAccessorArgs
// CHECK: %[[SRC_ACC:.*]] = ttkernel.TensorAccessor(%[[SRC_ARGS]], %[[BANK_BASE]], {{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// CHECK: %[[CB_PTR:.*]] = ttkernel.get_write_ptr(%[[CB]]) : (!ttkernel.cb<2, f32>) -> i32
// CHECK: ttkernel.noc_async_read_tile({{.*}}, %[[SRC_ACC]], %[[CB_PTR]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// CHECK: ttkernel.noc_async_read_barrier() : () -> ()
// CHECK-NOT: ttkernel.noc_async_write_barrier
module {
  func.func @dma_single_tile_single_copy(%arg0: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
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

// CHECK-LABEL: func.func @cb_to_tensor
// CHECK-DAG: %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, f32>
// CHECK: %[[BANK_BASE:.*]] = ttkernel.get_common_arg_val(%[[C0_IDX]]) : (index) -> i32
// CHECK: %[[DST_ARGS:.*]] = ttkernel.TensorAccessorArgs({{.*}}) : (i32, i32) -> !ttkernel.TensorAccessorArgs
// CHECK: %[[DST_ACC:.*]] = ttkernel.TensorAccessor(%[[DST_ARGS]], %[[BANK_BASE]], {{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// CHECK: %[[CB_PTR:.*]] = ttkernel.get_read_ptr(%[[CB]]) : (!ttkernel.cb<2, f32>) -> i32
// CHECK: ttkernel.noc_async_write_tile({{.*}}, %[[DST_ACC]], %[[CB_PTR]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// CHECK: ttkernel.noc_async_write_barrier() : () -> ()
// CHECK-NOT: ttkernel.noc_async_read_barrier
module {
  func.func @cb_to_tensor(%arg0: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
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
// Pre-conversion grouping emits all setup ops first, then both DMA reads together.
// CHECK-LABEL: func.func @dma_batched
// Setup phase: both accessors and CB pointers created first
// CHECK: ttkernel.get_common_arg_val({{.*}}) : (index) -> i32
// CHECK: ttkernel.TensorAccessor({{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// CHECK: ttkernel.get_write_ptr({{.*}}) : (!ttkernel.cb<2, f32>) -> i32
// CHECK: ttkernel.get_common_arg_val({{.*}}) : (index) -> i32
// CHECK: ttkernel.TensorAccessor({{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// CHECK: ttkernel.get_write_ptr({{.*}}) : (!ttkernel.cb<2, f32>) -> i32
// Execution phase: both tile reads together
// CHECK: ttkernel.noc_async_read_tile({{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// CHECK: ttkernel.noc_async_read_tile({{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// Consecutive barriers are deduplicated to a single barrier.
// CHECK: ttkernel.noc_async_read_barrier() : () -> ()
// CHECK-NOT: ttkernel.noc_async_read_barrier
// CHECK-NOT: ttkernel.noc_async_write_barrier
module {
  func.func @dma_batched(%t0: tensor<32x32xf32, #layout>, %t1: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
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

// CHECK-LABEL: func.func @dma_pipelined_loop
// CHECK: ttkernel.noc_async_read_tile({{.*}}, {{.*}}, {{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// CHECK: scf.for {{.*}} {
// CHECK:   ttkernel.noc_async_read_tile({{.*}}, {{.*}}, {{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// CHECK:   ttkernel.noc_async_read_barrier() : () -> ()
// CHECK: }
// CHECK: ttkernel.noc_async_read_barrier() : () -> ()
// CHECK-NOT: ttkernel.noc_async_write_barrier
module {
  func.func @dma_pipelined_loop(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
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
// CHECK-LABEL: func.func @dma_single_tile_two_phase_loops
// CHECK: %[[HANDLES0:.*]] = tensor.empty() : tensor<4x!ttl.transfer_handle<read>>
// CHECK: %[[CAST:.*]] = tensor.cast %[[HANDLES0]] : tensor<4x!ttl.transfer_handle<read>> to tensor<?x!ttl.transfer_handle<read>>
// CHECK: %[[HANDLES:.*]] = scf.for {{.*}} iter_args(%[[H:.*]] = %[[CAST]]) -> (tensor<?x!ttl.transfer_handle<read>>) {
// CHECK:   ttkernel.noc_async_read_tile({{.*}}, {{.*}}, {{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// CHECK:   %[[XF:.*]] = builtin.unrealized_conversion_cast {{.*}} : i32 to !ttl.transfer_handle<read>
// CHECK:   %[[INS:.*]] = tensor.insert %[[XF]] into %[[H]]{{\[}}{{.*}}{{\]}} : tensor<?x!ttl.transfer_handle<read>>
// CHECK:   scf.yield %[[INS]] : tensor<?x!ttl.transfer_handle<read>>
// CHECK: }
// CHECK: scf.for {{.*}} {
// CHECK:   ttkernel.noc_async_read_barrier() : () -> ()
// CHECK: }
// CHECK-NOT: ttkernel.noc_async_write_barrier
module {
  func.func @dma_single_tile_two_phase_loops(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
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
// CHECK-LABEL: func.func @dma_single_tile_double_wait
// CHECK:      ttkernel.noc_async_read_barrier() : () -> ()
// CHECK-NOT: ttkernel.noc_async_read_barrier
// CHECK-NOT: ttkernel.noc_async_write_barrier
module {
  func.func @dma_single_tile_double_wait(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
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
// CHECK-LABEL: func.func @dma_single_tile_single_element_container
// CHECK: ttkernel.noc_async_read_tile({{.*}}, {{.*}}, {{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// CHECK: ttkernel.noc_async_read_barrier() : () -> ()
// CHECK-NOT: ttkernel.noc_async_write_barrier
// CHECK: return
module {
  func.func @dma_single_tile_single_element_container(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
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
// CHECK-LABEL: func.func @dma_multi_tile_read
// CHECK-DAG: %[[TILE_LB:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[TILE_STEP:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[TILES_BOUND:.*]] = arith.constant 2 : index
// CHECK: ttkernel.get_common_arg_val({{.*}}) : (index) -> i32
// CHECK: %[[ARGS:.*]] = ttkernel.TensorAccessorArgs({{.*}}) : (i32, i32) -> !ttkernel.TensorAccessorArgs
// CHECK: %[[ACC:.*]] = ttkernel.TensorAccessor(%[[ARGS]], {{.*}}, {{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// CHECK: %[[CB_PTR:.*]] = ttkernel.get_write_ptr({{.*}}) : (!ttkernel.cb<{{.*}}>) -> i32
// CHECK: scf.for %[[TILE_Y:.*]] = %[[TILE_LB]] to %[[TILES_BOUND]] step %[[TILE_STEP]]
// CHECK:   scf.for %[[TILE_X:.*]] = %[[TILE_LB]] to %[[TILES_BOUND]] step %[[TILE_STEP]]
// CHECK:     %[[TILE_OFFSET_Y:.*]] = arith.muli %[[TILE_Y]], %[[TILES_BOUND]] : index
// CHECK:     %[[TILE_OFFSET_X:.*]] = arith.addi %[[TILE_OFFSET_Y]], %[[TILE_X]] : index
// CHECK:     %[[TILE_OFFSET_I32:.*]] = arith.index_cast %[[TILE_OFFSET_X]] : index to i32
// CHECK:     ttkernel.noc_async_read_tile(%[[TILE_OFFSET_I32]], %[[ACC]], %[[CB_PTR]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// CHECK: ttkernel.noc_async_read_barrier() : () -> ()
// CHECK-NOT: ttkernel.noc_async_write_barrier
module {
  func.func @dma_multi_tile_read(%arg0: tensor<64x64xf32, #layout_tile>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
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
// CHECK-LABEL: func.func @dma_multi_tile_write
// CHECK-DAG: %[[TILE_LB:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[TILE_STEP:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[TILES_BOUND:.*]] = arith.constant 2 : index
// CHECK: ttkernel.get_common_arg_val({{.*}}) : (index) -> i32
// CHECK: %[[ARGS:.*]] = ttkernel.TensorAccessorArgs({{.*}}) : (i32, i32) -> !ttkernel.TensorAccessorArgs
// CHECK: %[[ACC:.*]] = ttkernel.TensorAccessor(%[[ARGS]], {{.*}}, {{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// CHECK: %[[CB_PTR:.*]] = ttkernel.get_read_ptr({{.*}}) : (!ttkernel.cb<{{.*}}>) -> i32
// CHECK: scf.for %[[TILE_Y:.*]] = %[[TILE_LB]] to %[[TILES_BOUND]] step %[[TILE_STEP]]
// CHECK:   scf.for %[[TILE_X:.*]] = %[[TILE_LB]] to %[[TILES_BOUND]] step %[[TILE_STEP]]
// CHECK:     %[[TILE_OFFSET_Y:.*]] = arith.muli %[[TILE_Y]], %[[TILES_BOUND]] : index
// CHECK:     %[[TILE_OFFSET_X:.*]] = arith.addi %[[TILE_OFFSET_Y]], %[[TILE_X]] : index
// CHECK:     %[[TILE_OFFSET_I32:.*]] = arith.index_cast %[[TILE_OFFSET_X]] : index to i32
// CHECK:     ttkernel.noc_async_write_tile(%[[TILE_OFFSET_I32]], %[[ACC]], %[[CB_PTR]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// CHECK: ttkernel.noc_async_write_barrier() : () -> ()
// CHECK-NOT: ttkernel.noc_async_read_barrier
module {
  func.func @dma_multi_tile_write(%arg0: tensor<64x64xf32, #layout_tile>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
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
// CHECK-LABEL: func.func @dma_multi_tile_read_cb_shape
// CHECK-DAG: %[[TILE_LB:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[TILE_STEP:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[TILES_BOUND:.*]] = arith.constant 2 : index
// CHECK: ttkernel.get_common_arg_val({{.*}}) : (index) -> i32
// CHECK: %[[ARGS:.*]] = ttkernel.TensorAccessorArgs({{.*}}) : (i32, i32) -> !ttkernel.TensorAccessorArgs
// CHECK: %[[ACC:.*]] = ttkernel.TensorAccessor(%[[ARGS]], {{.*}}, {{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// CHECK: %[[CB_PTR:.*]] = ttkernel.get_write_ptr({{.*}}) : (!ttkernel.cb<{{.*}}>) -> i32
// CHECK: scf.for %[[TILE_Y:.*]] = %[[TILE_LB]] to %[[TILES_BOUND]] step %[[TILE_STEP]]
// CHECK:   scf.for %[[TILE_X:.*]] = %[[TILE_LB]] to %[[TILES_BOUND]] step %[[TILE_STEP]]
// CHECK:     %[[TILE_OFFSET_Y:.*]] = arith.muli %[[TILE_Y]], %[[TILES_BOUND]] : index
// CHECK:     %[[TILE_OFFSET_X:.*]] = arith.addi %[[TILE_OFFSET_Y]], %[[TILE_X]] : index
// CHECK:     %[[TILE_OFFSET_I32:.*]] = arith.index_cast %[[TILE_OFFSET_X]] : index to i32
// CHECK:     ttkernel.noc_async_read_tile(%[[TILE_OFFSET_I32]], %[[ACC]], %[[CB_PTR]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// CHECK: ttkernel.noc_async_read_barrier() : () -> ()
// CHECK-NOT: ttkernel.noc_async_write_barrier
module {
  func.func @dma_multi_tile_read_cb_shape(%arg0: tensor<64x64xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
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
// CHECK-LABEL: func.func @dma_multi_tile_write_rect
// CHECK-DAG: %[[TILE_LB:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[TILE_STEP:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[TILES_Y_BOUND:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[TILES_X_BOUND:.*]] = arith.constant 2 : index
// CHECK: ttkernel.get_common_arg_val({{.*}}) : (index) -> i32
// CHECK: %[[ARGS:.*]] = ttkernel.TensorAccessorArgs({{.*}}) : (i32, i32) -> !ttkernel.TensorAccessorArgs
// CHECK: %[[ACC:.*]] = ttkernel.TensorAccessor(%[[ARGS]], {{.*}}, {{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// CHECK: %[[CB_PTR:.*]] = ttkernel.get_read_ptr({{.*}}) : (!ttkernel.cb<{{.*}}>) -> i32
// CHECK: scf.for %[[TILE_Y:.*]] = %[[TILE_LB]] to %[[TILES_Y_BOUND]] step %[[TILE_STEP]]
// CHECK:   scf.for %[[TILE_X:.*]] = %[[TILE_LB]] to %[[TILES_X_BOUND]] step %[[TILE_STEP]]
// CHECK:     %[[TILE_OFFSET_Y:.*]] = arith.muli %[[TILE_Y]], %[[TILES_X_BOUND]] : index
// CHECK:     %[[TILE_OFFSET_X:.*]] = arith.addi %[[TILE_OFFSET_Y]], %[[TILE_X]] : index
// CHECK:     %[[TILE_OFFSET_I32:.*]] = arith.index_cast %[[TILE_OFFSET_X]] : index to i32
// CHECK:     ttkernel.noc_async_write_tile(%[[TILE_OFFSET_I32]], %[[ACC]], %[[CB_PTR]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// CHECK: ttkernel.noc_async_write_barrier() : () -> ()
// CHECK-NOT: ttkernel.noc_async_read_barrier
module {
  func.func @dma_multi_tile_write_rect(%arg0: tensor<96x64xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %cb, %arg0 : (!ttl.cb<[1, 1], f32, 2>, tensor<96x64xf32, #layout>) -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Batched multi-tile reads with identical tile grids (2x2).
// Pre-conversion grouping emits all setup ops first, then a single fused tile loop
// with both DMAs in the body.
//
// CHECK-LABEL: func.func @dma_batched_multi_tile_read
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// Setup ops emitted first (before loops)
// CHECK: %[[ACC1:.*]] = ttkernel.TensorAccessor({{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// CHECK: %[[PTR1:.*]] = ttkernel.get_write_ptr({{.*}}) : (!ttkernel.cb<2, f32>) -> i32
// CHECK: %[[ACC2:.*]] = ttkernel.TensorAccessor({{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// CHECK: %[[PTR2:.*]] = ttkernel.get_write_ptr({{.*}}) : (!ttkernel.cb<2, f32>) -> i32
// Single fused tile loop with both DMAs in body
// CHECK: scf.for %[[TY:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:   scf.for %[[TX:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:     %[[OFF_Y:.*]] = arith.muli %[[TY]], %[[C2]] : index
// CHECK-NEXT:     %[[OFF:.*]] = arith.addi %[[OFF_Y]], %[[TX]] : index
// CHECK-NEXT:     %[[OFF_I32:.*]] = arith.index_cast %[[OFF]] : index to i32
// CHECK-NEXT:     ttkernel.noc_async_read_tile(%[[OFF_I32]], %[[ACC1]], %[[PTR1]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// CHECK-NEXT:     ttkernel.noc_async_read_tile(%[[OFF_I32]], %[[ACC2]], %[[PTR2]]) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK: ttkernel.noc_async_read_barrier
module {
  func.func @dma_batched_multi_tile_read(%arg0: tensor<64x64xf32, #layout>, %arg1: tensor<64x64xf32, #layout>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>

    // Batched multi-tile copies: both 64x64 tensors (2x2 tiles each)
    %xf0 = ttl.copy %arg0, %cb0 : (tensor<64x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf1 = ttl.copy %arg1, %cb1 : (tensor<64x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>

    ttl.wait %xf0 : !ttl.transfer_handle<read>
    ttl.wait %xf1 : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Ensures read/write wait lowerings pick the correct barrier after type
// conversion. Expect exactly one read barrier followed by one write barrier,
// no cross-type barriers.
//
// CHECK-LABEL: func.func @wait_barriers_mixed
// CHECK: ttkernel.noc_async_read_barrier() : () -> ()
// CHECK-NOT: ttkernel.noc_async_write_barrier() : () -> ()
// CHECK: ttkernel.noc_async_write_barrier() : () -> ()
// CHECK-NOT: ttkernel.noc_async_read_barrier() : () -> ()
module {
  func.func @wait_barriers_mixed(%t: tensor<32x32xf32, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>

    %xf_read = ttl.copy %t, %cb : (tensor<32x32xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    %xf_write = ttl.copy %cb, %t : (!ttl.cb<[1, 1], f32, 2>, tensor<32x32xf32, #layout>) -> !ttl.transfer_handle<write>

    ttl.wait %xf_read : !ttl.transfer_handle<read>
    ttl.wait %xf_write : !ttl.transfer_handle<write>
    func.return
  }
}
