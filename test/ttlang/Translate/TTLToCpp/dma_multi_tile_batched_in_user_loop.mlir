// RUN: ttlang-opt --ttl-to-ttkernel-pipeline --canonicalize %s -o %t.ttkernel.mlir
// RUN: ttmlir-opt --allow-unregistered-dialect --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttmlir-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp

// Test: Multi-tile copies (same tile grid) within a user loop.
// Multiple copies with SAME tile grid dimensions within a user loop.
//
// User loop: 0..3
// Both tensors: 64x64xf32 (2x2 tiles) - SAME tile grid
// Both copies issued before barriers
//
// Pre-conversion grouping emits setup then fused loop:
//   for user_iter in 0..3:
//     accessor1 = TensorAccessor(...)
//     ptr1 = get_write_ptr(...)
//     accessor2 = TensorAccessor(...)
//     ptr2 = get_write_ptr(...)
//     for tile_y in 0..2:
//       for tile_x in 0..2:
//         noc_async_read_tile(offset, accessor1, ptr1)
//         noc_async_read_tile(offset, accessor2, ptr2)
//     noc_async_read_barrier()

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-LABEL: // batched_multi_tile_user_loop
// CHECK-NEXT: #include <cstdint>
// CHECK-NEXT: #include "tools/profiler/kernel_profiler.hpp"
// CHECK-NEXT: #include "dataflow_api.h"
// CHECK-NEXT: void kernel_main() {
// CHECK-DAG:   size_t [[TILES_BOUND:v[0-9]+]] = 2;
// CHECK-DAG:   size_t [[STEP:v[0-9]+]] = 1;
// CHECK-DAG:   size_t [[USER_UB:v[0-9]+]] = 3;
// CHECK-DAG:   size_t [[LB:v[0-9]+]] = 0;

// User loop from input MLIR (0..3)
// CHECK:   for (size_t [[USER_ITER:[a-z][0-9]+]] = [[LB]]; [[USER_ITER]] < [[USER_UB]]; [[USER_ITER]] += [[STEP]]) {

// Setup: all tensor accessors and CB pointers created before tile loop
// CHECK:     TensorAccessor [[ACC1:v[0-9]+]] = TensorAccessor(
// CHECK:     int32_t [[PTR1:v[0-9]+]] = get_write_ptr(
// CHECK:     TensorAccessor [[ACC2:v[0-9]+]] = TensorAccessor(
// CHECK:     int32_t [[PTR2:v[0-9]+]] = get_write_ptr(

// Fused tile loops: single nested loop with both DMAs
// CHECK:     for (size_t [[TILE_Y:[a-z][0-9]+]] = [[LB]]; [[TILE_Y]] < [[TILES_BOUND]]; [[TILE_Y]] += [[STEP]]) {
// CHECK-NEXT:      for (size_t [[TILE_X:[a-z][0-9]+]] = [[LB]]; [[TILE_X]] < [[TILES_BOUND]]; [[TILE_X]] += [[STEP]]) {
// CHECK:             noc_async_read_tile({{.*}}, [[ACC1]], [[PTR1]]);
// CHECK-NEXT:        noc_async_read_tile({{.*}}, [[ACC2]], [[PTR2]]);
// CHECK:           }
// CHECK-NEXT:    }

// Consecutive barriers deduplicated to single barrier.
// CHECK:         noc_async_read_barrier();
// CHECK:       }
// CHECK:       return;
// CHECK-NEXT: }

module {
  func.func @batched_multi_tile_user_loop(%arg0: tensor<64x64xf32, #layout>, %arg1: tensor<64x64xf32, #layout>)
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %cb1 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %cb2 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index

    // User loop (0..3)
    scf.for %i = %c0 to %c3 step %c1 {
      // Batch: issue both copies before barrier
      // Both tensors have same tile grid (2x2) - potential for shared tile loop
      %xf1 = ttl.copy %arg0, %cb1 : (tensor<64x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>
      %xf2 = ttl.copy %arg1, %cb2 : (tensor<64x64xf32, #layout>, !ttl.cb<[1, 1], f32, 2>) -> !ttl.transfer_handle<read>

      // Single barrier waits for both transfers
      ttl.wait %xf1 : !ttl.transfer_handle<read>
      ttl.wait %xf2 : !ttl.transfer_handle<read>
    }

    func.return
  }
}
