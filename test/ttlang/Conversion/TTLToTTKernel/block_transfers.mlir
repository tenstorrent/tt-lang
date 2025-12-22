// RUN: ttlang-opt --convert-ttl-to-ttkernel --canonicalize -cse --split-input-file %s | FileCheck %s --check-prefix=TTKERNEL
// Summary: Block transfer optimization tests for ttl.copy lowering (Issue #138).
// Tests single block transfers for contiguous row-major tensors.

#dram = #ttnn.buffer_type<dram>
#layout_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xf32, #dram>, <interleaved>>

// Test: Single tile (32x32) row-major should use block transfer
// TTKERNEL-LABEL: func.func @block_transfer_single_tile_read
// TTKERNEL-DAG: %[[SIZE:.*]] = arith.constant 4096 : i32
// TTKERNEL-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// TTKERNEL: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, f32>
// TTKERNEL: %[[SRC_ACC:.*]] = ttkernel.TensorAccessor({{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// TTKERNEL: %[[CB_PTR:.*]] = ttkernel.get_write_ptr(%[[CB]]) : (!ttkernel.cb<2, f32>) -> i32
// TTKERNEL: %[[NOC_ADDR:.*]] = ttkernel.tensor_accessor.get_noc_addr(%[[SRC_ACC]], %[[ZERO]], %[[ZERO]]) : (!ttkernel.TensorAccessor, i32, i32) -> !ttkernel.noc_addr
// TTKERNEL: ttkernel.noc_async_read(%[[NOC_ADDR]], %[[CB_PTR]], %[[SIZE]]) : (!ttkernel.noc_addr, i32, i32) -> ()
// TTKERNEL-NOT: scf.for
// TTKERNEL-NOT: ttkernel.noc_async_read_tile
module {
  func.func @block_transfer_single_tile_read(%arg0: tensor<32x32xf32, #layout_rm>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %arg0, %cb : (tensor<32x32xf32, #layout_rm>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x64xf32, #dram>, <interleaved>>

// Test: Multi-tile tensor (64x64) row-major should use single block transfer
// TTKERNEL-LABEL: func.func @block_transfer_multi_tile_read
// TTKERNEL-DAG: %[[SIZE:.*]] = arith.constant 16384 : i32
// TTKERNEL-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// TTKERNEL: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<8, f32>
// TTKERNEL: %[[SRC_ACC:.*]] = ttkernel.TensorAccessor({{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// TTKERNEL: %[[CB_PTR:.*]] = ttkernel.get_write_ptr(%[[CB]]) : (!ttkernel.cb<8, f32>) -> i32
// TTKERNEL: %[[NOC_ADDR:.*]] = ttkernel.tensor_accessor.get_noc_addr(%[[SRC_ACC]], %[[ZERO]], %[[ZERO]]) : (!ttkernel.TensorAccessor, i32, i32) -> !ttkernel.noc_addr
// TTKERNEL: ttkernel.noc_async_read(%[[NOC_ADDR]], %[[CB_PTR]], %[[SIZE]]) : (!ttkernel.noc_addr, i32, i32) -> ()
// TTKERNEL-NOT: scf.for
// TTKERNEL-NOT: ttkernel.noc_async_read_tile
module {
  func.func @block_transfer_multi_tile_read(%arg0: tensor<64x64xf32, #layout_rm>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>
    %xf = ttl.copy %arg0, %cb : (tensor<64x64xf32, #layout_rm>, !ttl.cb<[2, 2], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xf32, #dram>, <interleaved>>

// Test: CB→Tensor write should also use block transfer
// TTKERNEL-LABEL: func.func @block_transfer_write
// TTKERNEL-DAG: %[[SIZE:.*]] = arith.constant 4096 : i32
// TTKERNEL-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// TTKERNEL: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, f32>
// TTKERNEL: %[[DST_ACC:.*]] = ttkernel.TensorAccessor({{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// TTKERNEL: %[[CB_PTR:.*]] = ttkernel.get_read_ptr(%[[CB]]) : (!ttkernel.cb<2, f32>) -> i32
// TTKERNEL: %[[NOC_ADDR:.*]] = ttkernel.tensor_accessor.get_noc_addr(%[[DST_ACC]], %[[ZERO]], %[[ZERO]]) : (!ttkernel.TensorAccessor, i32, i32) -> !ttkernel.noc_addr
// TTKERNEL: ttkernel.noc_async_write(%[[CB_PTR]], %[[NOC_ADDR]], %[[SIZE]]) : (i32, !ttkernel.noc_addr, i32) -> ()
// TTKERNEL-NOT: scf.for
// TTKERNEL-NOT: ttkernel.noc_async_write_tile
module {
  func.func @block_transfer_write(%arg0: tensor<32x32xf32, #layout_rm>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %xf = ttl.copy %cb, %arg0 : (!ttl.cb<[1, 1], f32, 2>, tensor<32x32xf32, #layout_rm>) -> !ttl.transfer_handle<write>
    ttl.wait %xf : !ttl.transfer_handle<write>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x128xf32, #dram>, <interleaved>>

// Test: Large tensor (128x128 = 16 tiles) should use single block transfer
// TTKERNEL-LABEL: func.func @block_transfer_large_tensor
// TTKERNEL-DAG: %[[SIZE:.*]] = arith.constant 65536 : i32
// TTKERNEL: %[[NOC_ADDR:.*]] = ttkernel.tensor_accessor.get_noc_addr({{.*}}) : (!ttkernel.TensorAccessor, i32, i32) -> !ttkernel.noc_addr
// TTKERNEL: ttkernel.noc_async_read(%[[NOC_ADDR]], {{.*}}, %[[SIZE]]) : (!ttkernel.noc_addr, i32, i32) -> ()
// TTKERNEL-NOT: scf.for
module {
  func.func @block_transfer_large_tensor(%arg0: tensor<128x128xf32, #layout_rm>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[4, 4], f32, 2>
    %xf = ttl.copy %arg0, %cb : (tensor<128x128xf32, #layout_rm>, !ttl.cb<[4, 4], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x64xf16, #dram>, <interleaved>>

// Test: f16 element type - verify correct size (64x64xf16 = 8192 bytes)
// TTKERNEL-LABEL: func.func @block_transfer_f16
// TTKERNEL-DAG: %[[SIZE:.*]] = arith.constant 8192 : i32
// TTKERNEL: ttkernel.noc_async_read({{.*}}, {{.*}}, %[[SIZE]]) : (!ttkernel.noc_addr, i32, i32) -> ()
module {
  func.func @block_transfer_f16(%arg0: tensor<64x64xf16, #layout_rm>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], f16, 2>
    %xf = ttl.copy %arg0, %cb : (tensor<64x64xf16, #layout_rm>, !ttl.cb<[2, 2], f16, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout_tiled = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Test: Tiled layout should fall back to tile-level transfers
// TTKERNEL-LABEL: func.func @tile_fallback_tiled_layout
// TTKERNEL: scf.for
// TTKERNEL: ttkernel.noc_async_read_tile({{.*}}) : (i32, !ttkernel.TensorAccessor, i32) -> ()
// TTKERNEL-NOT: ttkernel.noc_async_read(%
module {
  func.func @tile_fallback_tiled_layout(%arg0: tensor<64x64xf32, #layout_tiled>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>
    %xf = ttl.copy %arg0, %cb : (tensor<64x64xf32, #layout_tiled>, !ttl.cb<[2, 2], f32, 2>) -> !ttl.transfer_handle<read>
    ttl.wait %xf : !ttl.transfer_handle<read>
    func.return
  }
}

// TODO: Add performance benchmarks comparing tile vs block transfers.
