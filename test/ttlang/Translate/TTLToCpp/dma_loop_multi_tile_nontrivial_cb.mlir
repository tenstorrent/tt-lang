// RUN: ttlang-opt --ttl-to-ttkernel-pipeline --canonicalize %s -o %t.ttkernel.mlir
// RUN: ttmlir-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Test: User-written loop with multiple multi-tile DMA operations and nontrivial CB shape.
// This test distinguishes between user loops (from input) and generated tile loops.
//
// Input: scf.for loop (iterations 0..4) containing two multi-tile copies
// Tensor1: 64x64xf32 (2x2 tiles) → CB1: [2,2] (2x2 tiles, matching tensor)
// Tensor2: 96x64xf32 (3x2 tiles) → CB2: [3,1] (3x1 tiles, different from tensor grid)
//
// Expected output structure:
//   for (USER_ITER in 0..4) {              ← User loop from input MLIR
//     for (tile_y in 0..2) {               ← Generated tile loop for Tensor1
//       for (tile_x in 0..2) {             ← Generated tile loop for Tensor1
//         noc_async_read_tile(...)         ← Tensor1 → CB1
//       }
//     }
//     noc_async_read_barrier()
//     for (tile_y in 0..3) {               ← Generated tile loop for Tensor2
//       for (tile_x in 0..2) {             ← Generated tile loop for Tensor2
//         noc_async_read_tile(...)         ← Tensor2 → CB2
//       }
//     }
//     noc_async_read_barrier()
//   }

#dram = #ttnn.buffer_type<dram>
#layout_2x2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_3x2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<3x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: // dma_loop_multi_tile
// CHECK: void kernel_main() {
// CHECK-DAG:   size_t [[TILES_3:v[0-9]+]] = 3;
// CHECK-DAG:   size_t [[TILES_2:v[0-9]+]] = 2;
// CHECK-DAG:   int32_t [[ADDR:v[0-9]+]] = 256;
// CHECK-DAG:   size_t [[TILE_STEP:v[0-9]+]] = 1;
// CHECK-DAG:   size_t [[USER_UB:v[0-9]+]] = 4;
// CHECK-DAG:   size_t [[TILE_LB:v[0-9]+]] = 0;
// CHECK:   for (size_t [[USER_ITER:[a-z][0-9]+]] = [[TILE_LB]]; [[USER_ITER]] < [[USER_UB]]; [[USER_ITER]] += [[TILE_STEP]]) {
// First copy: arg0 (64x64) → CB0, accessor with runtime arg index 0
// CHECK:     int32_t [[RT_ARG1:v[0-9]+]] = get_common_arg_val<uint32_t>([[TILE_LB]]);
// Placeholder value 42 is a temporary hack, see issue #168
// CHECK:     auto [[ACC1_ARGS:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<42, 0>();
// CHECK:     TensorAccessor [[ACC1:v[0-9]+]] = TensorAccessor([[ACC1_ARGS]], [[RT_ARG1]], [[ADDR]]);
// CHECK:     int32_t [[CB_PTR1:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(0));
// CHECK:     for (size_t [[TILE1_Y:[a-z][0-9]+]] = [[TILE_LB]]; [[TILE1_Y]] < [[TILES_2]]; [[TILE1_Y]] += [[TILE_STEP]]) {
// CHECK:       for (size_t [[TILE1_X:[a-z][0-9]+]] = [[TILE_LB]]; [[TILE1_X]] < [[TILES_2]]; [[TILE1_X]] += [[TILE_STEP]]) {
// CHECK:         size_t [[TILE1_OFFSET_Y:v[0-9]+]] = [[TILE1_Y]] * [[TILES_2]];
// CHECK:         size_t [[TILE1_OFFSET_X:v[0-9]+]] = [[TILE1_OFFSET_Y]] + [[TILE1_X]];
// CHECK:         ptrdiff_t [[TILE1_OFFSET_PTR:v[0-9]+]] = (ptrdiff_t) [[TILE1_OFFSET_X]];
// CHECK:         int32_t [[TILE1_OFFSET:v[0-9]+]] = (int32_t) [[TILE1_OFFSET_PTR]];
// CHECK:         noc_async_read_tile([[TILE1_OFFSET]], [[ACC1]], [[CB_PTR1]]);
// CHECK:       }
// CHECK:     }
// CHECK:     noc_async_read_barrier();
// Second copy: arg1 (96x64) → CB1, accessor with runtime arg index 1
// CHECK:     int32_t [[RT_ARG2:v[0-9]+]] = get_common_arg_val<uint32_t>([[TILE_STEP]]);
// Placeholder value 43 is a temporary hack, see issue #168
// CHECK:     auto [[ACC2_ARGS:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<43, 0>();
// CHECK:     TensorAccessor [[ACC2:v[0-9]+]] = TensorAccessor([[ACC2_ARGS]], [[RT_ARG2]], [[ADDR]]);
// CHECK:     int32_t [[CB_PTR2:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(1));
// CHECK:     for (size_t [[TILE2_Y:[a-z][0-9]+]] = [[TILE_LB]]; [[TILE2_Y]] < [[TILES_3]]; [[TILE2_Y]] += [[TILE_STEP]]) {
// CHECK:       for (size_t [[TILE2_X:[a-z][0-9]+]] = [[TILE_LB]]; [[TILE2_X]] < [[TILES_2]]; [[TILE2_X]] += [[TILE_STEP]]) {
// CHECK:         size_t [[TILE2_OFFSET_Y:v[0-9]+]] = [[TILE2_Y]] * [[TILES_2]];
// CHECK:         size_t [[TILE2_OFFSET_X:v[0-9]+]] = [[TILE2_OFFSET_Y]] + [[TILE2_X]];
// CHECK:         ptrdiff_t [[TILE2_OFFSET_PTR:v[0-9]+]] = (ptrdiff_t) [[TILE2_OFFSET_X]];
// CHECK:         int32_t [[TILE2_OFFSET:v[0-9]+]] = (int32_t) [[TILE2_OFFSET_PTR]];
// CHECK:         noc_async_read_tile([[TILE2_OFFSET]], [[ACC2]], [[CB_PTR2]]);
// CHECK:       }
// CHECK:     }
// CHECK:     noc_async_read_barrier();
// CHECK:   }
// CHECK:   return;
// CHECK-NEXT: }

module {
  func.func @dma_loop_multi_tile(%arg0: tensor<64x64xf32, #layout_2x2>, %arg1: tensor<96x64xf32, #layout_3x2>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb1 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], f32, 2>
    %cb2 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[3, 1], f32, 2>
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index

    // User-written loop over 4 iterations
    scf.for %i = %c0 to %c4 step %c1 {
      // First copy: 64x64 (2x2 tiles) → CB [2,2]
      %xf1 = ttl.copy %arg0, %cb1 : (tensor<64x64xf32, #layout_2x2>, !ttl.cb<[2, 2], f32, 2>) -> !ttl.transfer_handle<read>
      ttl.wait %xf1 : !ttl.transfer_handle<read>

      // Second copy: 96x64 (3x2 tiles) → CB [3,1]
      %xf2 = ttl.copy %arg1, %cb2 : (tensor<96x64xf32, #layout_3x2>, !ttl.cb<[3, 1], f32, 2>) -> !ttl.transfer_handle<read>
      ttl.wait %xf2 : !ttl.transfer_handle<read>
    }

    func.return
  }
}
