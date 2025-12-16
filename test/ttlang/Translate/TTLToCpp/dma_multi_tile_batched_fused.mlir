// RUN: ttlang-opt --ttl-to-ttkernel-pipeline="fuse-tile-loops=true" --canonicalize %s -o %t.ttkernel.mlir
// RUN: ttmlir-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttmlir-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Test: Multi-tile copies fused into a single tile loop.
// Same as dma_multi_tile_batched_in_user_loop.mlir but with --fuse-tile-loops=true.
//
// User loop: 0..3
// All three tensors: 64x64xf32 (2x2 tiles) - SAME tile grid
// All three copies issued before barriers
//
// After fusion:
//   for user_iter in 0..3:
//     for tile_y in 0..2:
//       for tile_x in 0..2:
//         noc_async_read_tile(offset, accessor1, ...)
//         noc_async_read_tile(offset, accessor2, ...)
//         noc_async_read_tile(offset, accessor3, ...)
//     noc_async_read_barrier()
//     noc_async_read_barrier()
//     noc_async_read_barrier()

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: // batched_multi_tile_fused
// CHECK-NEXT: #include <cstdint>
// CHECK-NEXT: #include "tools/profiler/kernel_profiler.hpp"
// CHECK-NEXT: #include "dataflow_api.h"
// CHECK-NEXT: void kernel_main() {
// CHECK-DAG:   size_t [[TILES_BOUND:v[0-9]+]] = 2;
// CHECK-DAG:   size_t [[USER_UB:v[0-9]+]] = 3;
// CHECK-DAG:   size_t [[STEP:v[0-9]+]] = 1;
// CHECK-DAG:   size_t [[LB:v[0-9]+]] = 0;
// CHECK-DAG:   int32_t [[ZERO:v[0-9]+]] = 0;

// User loop from input MLIR (0..3)
// CHECK:   for (size_t [[USER_ITER:[a-z][0-9]+]] = [[LB]]; [[USER_ITER]] < [[USER_UB]]; [[USER_ITER]] += [[STEP]]) {

// Fused tile loops: single nested loop with both DMAs
// CHECK:     for (size_t [[TILE_Y:[a-z][0-9]+]] = [[LB]]; [[TILE_Y]] < [[TILES_BOUND]]; [[TILE_Y]] += [[STEP]]) {
// CHECK-NEXT:      for (size_t [[TILE_X:[a-z][0-9]+]] = [[LB]]; [[TILE_X]] < [[TILES_BOUND]]; [[TILE_X]] += [[STEP]]) {

// First DMA in fused loop body
// CHECK:             noc_async_read_tile({{.*}}, {{.*}}, [[ZERO]]);

// Second DMA in same loop body (fused)
// CHECK:             noc_async_read_tile({{.*}}, {{.*}}, [[ZERO]]);

// Third DMA in same loop body (fused)
// CHECK:             noc_async_read_tile({{.*}}, {{.*}}, [[ZERO]]);

// End of fused inner loops
// CHECK:           }
// CHECK-NEXT:    }

// Consecutive barriers deduplicated to single barrier.
// CHECK:         noc_async_read_barrier();
// CHECK:       }
// CHECK:       return;
// CHECK-NEXT: }

module {
  func.func @batched_multi_tile_fused(%arg0: tensor<64x64xf32, #layout>, %arg1: tensor<64x64xf32, #layout>, %arg2: tensor<64x64xf32, #layout>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %cb1 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb2 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb3 = ttl.create_cb() {shape = [1, 1], element_type = f32, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>

    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index

    // User loop (0..3)
    scf.for %i = %c0 to %c3 step %c1 {
      // Batch: issue all three copies before barriers
      // All tensors have same tile grid (2x2) - will be fused into single loop
      %xf1 = ttl.copy %arg0, %cb1 : (tensor<64x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
      %xf2 = ttl.copy %arg1, %cb2 : (tensor<64x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
      %xf3 = ttl.copy %arg2, %cb3 : (tensor<64x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>

      // Barriers wait for all three transfers
      ttl.wait %xf1 : !ttl.transfer_handle<read>
      ttl.wait %xf2 : !ttl.transfer_handle<read>
      ttl.wait %xf3 : !ttl.transfer_handle<read>
    }

    func.return
  }
}
