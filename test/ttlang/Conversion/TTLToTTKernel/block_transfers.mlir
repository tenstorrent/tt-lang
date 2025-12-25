// RUN: ttlang-opt --ttl-to-ttkernel-pipeline --split-input-file %s | FileCheck %s

// Block transfer optimization tests.
//
// Tests verify that TTL copy operations lower to appropriate transfer strategies
// based on tensor layout contiguity:
//   1. FullyContiguous (row-major + interleaved): single noc_async_read/write
//   2. RowContiguous (row-major + sharded): per-row transfers (TODO: #118)
//   3. TileContiguous (tiled layout): per-tile noc_async_read_tile/write_tile

#dram = #ttnn.buffer_type<dram>
#layout_tiled = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Tiled layout (TileContiguous): generates nested loops with per-tile transfers.
// CHECK-LABEL: func.func @tiled_layout_uses_tile_transfers
// CHECK:       %[[CB:.*]] = ttkernel.get_compile_time_arg_val
// CHECK:       %[[ACCESSOR_ARGS:.*]] = ttkernel.TensorAccessorArgs
// CHECK-NEXT:  %[[ACCESSOR:.*]] = ttkernel.TensorAccessor
// CHECK-NEXT:  %[[CB_PTR:.*]] = ttkernel.get_write_ptr(%[[CB]])
// CHECK-NEXT:  scf.for
// CHECK-NEXT:    scf.for
// CHECK:           ttkernel.noc_async_read_tile({{.*}}, %[[ACCESSOR]], %[[CB_PTR]])
// CHECK:       ttkernel.noc_async_read_barrier
// CHECK-NOT:   ttkernel.noc_async_read_barrier
module {
  func.func @tiled_layout_uses_tile_transfers(%arg0: tensor<64x64xf32, #layout_tiled>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %arg0, %cb : (tensor<64x64xf32, #layout_tiled>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout_row_major = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x64xf32, #dram>, <interleaved>>

// Row-major interleaved (FullyContiguous): single block transfer for entire tensor.
// Verifies: no loops, uses noc_async_read (not _tile), single barrier, correct size.
// CHECK-LABEL: func.func @row_major_uses_single_block_transfer
// CHECK:       %[[SIZE:.*]] = arith.constant 16384 : i32
// CHECK:       %[[CB:.*]] = ttkernel.get_compile_time_arg_val
// CHECK:       %[[ACCESSOR_ARGS:.*]] = ttkernel.TensorAccessorArgs
// CHECK-NEXT:  %[[ACCESSOR:.*]] = ttkernel.TensorAccessor
// CHECK-NEXT:  %[[CB_PTR:.*]] = ttkernel.get_write_ptr(%[[CB]])
// CHECK-NEXT:  %[[NOC_ADDR:.*]] = ttkernel.tensor_accessor.get_noc_addr(%[[ACCESSOR]]
// CHECK-NEXT:  ttkernel.noc_async_read(%[[NOC_ADDR]], %[[CB_PTR]], %[[SIZE]])
// CHECK-NEXT:  ttkernel.noc_async_read_barrier
// CHECK-NOT:   scf.for
// CHECK-NOT:   ttkernel.noc_async_read_tile
// CHECK-NOT:   ttkernel.noc_async_read_barrier
module {
  func.func @row_major_uses_single_block_transfer(%arg0: tensor<64x64xf32, #layout_row_major>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %arg0, %cb : (tensor<64x64xf32, #layout_row_major>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout_row_major = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x64xf32, #dram>, <interleaved>>

// Row-major interleaved write (FullyContiguous): single block write.
// Verifies: no loops, uses noc_async_write (not _tile), single barrier.
// CHECK-LABEL: func.func @row_major_write_uses_single_block_transfer
// CHECK:       %[[SIZE:.*]] = arith.constant 16384 : i32
// CHECK:       %[[CB:.*]] = ttkernel.get_compile_time_arg_val
// CHECK:       %[[ACCESSOR_ARGS:.*]] = ttkernel.TensorAccessorArgs
// CHECK-NEXT:  %[[ACCESSOR:.*]] = ttkernel.TensorAccessor
// CHECK-NEXT:  %[[CB_PTR:.*]] = ttkernel.get_read_ptr(%[[CB]])
// CHECK-NEXT:  %[[NOC_ADDR:.*]] = ttkernel.tensor_accessor.get_noc_addr(%[[ACCESSOR]]
// CHECK-NEXT:  ttkernel.noc_async_write(%[[CB_PTR]], %[[NOC_ADDR]], %[[SIZE]])
// CHECK-NEXT:  ttkernel.noc_async_write_barrier
// CHECK-NOT:   scf.for
// CHECK-NOT:   ttkernel.noc_async_write_tile
// CHECK-NOT:   ttkernel.noc_async_write_barrier
module {
  func.func @row_major_write_uses_single_block_transfer(%arg0: tensor<64x64xf32, #layout_row_major>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %cb, %arg0 : (!ttl.cb<[1, 1], f32, 2>, tensor<64x64xf32, #layout_row_major>) -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
    func.return
  }
}

// TODO(#118): Add tests for sharded row-major layouts (RowContiguous) when supported.
